import torch
import torch.nn as nn
import psutil

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from utils.builder import get_optimizer, get_lr_scheduler
from utils.metrics import MetricsRecorder
import utils.misc as utils
from model.ae_adapter import ForecastAutoEncoder
import time
import datetime
from pathlib import Path
import torch.cuda.amp as amp
import os
from collections import OrderedDict
from torch.functional import F
from dfm_networks.transformer import LGUnet_all, ExtraLogVar
import numpy as np


class basemodel(nn.Module):
    def __init__(self, logger, **params) -> None:
        super().__init__()
        self.model = {}
        self.sub_model_name = []
        self.params = params
        self.logger = logger
        self.save_best_param = self.params.get("save_best", "MSE")
        self.metric_best = None

        self.begin_epoch = 0
        self.metric_best = 1000

        self.gscaler = amp.GradScaler(init_scale=1024, growth_interval=2000)
        
        # self.whether_final_test = self.params.get("final_test", False)
        self.pred_len = self.params.get("max_pred_len", 12)

        self.min_pred_len = self.params.get("min_pred_len", 1)


        network_type = params.get('type', "basenetwork")
        network_params = params.get("network_params", None)
        optimizer_params = params.get("optimizer_params", None)
        lr_params = params.get("lr_params", None)
        self.AE_loss = params.get("AE_loss", False)
        self.save_epoch = params.get("save_epoch", False)
        self.AE_only = params.get("AE_only", False)

        self.scheduler = lr_params.get("sched","cosine")

        self.model = LGUnet_all(**network_params)


        self.use_checkpoint = network_params.get("use_checkpoint",False)
        
        self.optimizer = get_optimizer(self.model, optimizer_params)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_params)


        # load metrics
        eval_metrics_list = params.get('metrics_list', [])
        if len(eval_metrics_list) > 0:
            self.eval_metrics = MetricsRecorder(eval_metrics_list)
        else:
            self.eval_metrics = None

        self.model.eval()

        self.extra_params = params.get("extra_params", {})
        self.two_step_training = self.extra_params.get("two_step_training", False)
        self.AE_checkpoint_path = params.get("AE_checkpoint_path", self.extra_params.get("AE_checkpoint_path", None))
        self.AE_model_version = params.get("AE_model_version", self.extra_params.get("AE_model_version", "34_4"))
        self.AE_stats_dir = params.get("AE_stats_dir", self.extra_params.get("AE_stats_dir", None))

        self.loss_type = self.extra_params.get("loss_type", "LpLoss")

        if self.loss_type == "LpLoss":
            self.loss = self.LpLoss
        elif self.loss_type == "Possloss":
            self.loss = self.Possloss

        if self.two_step_training or self.AE_loss:
            checkpoint_path = self.extra_params.get('checkpoint_path', None)
            if checkpoint_path is None:
                self.logger.info("finetune checkpoint path not exist")
            else:
                self.load_checkpoint(checkpoint_path, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)
        
        if self.AE_loss:
            self.AE = ForecastAutoEncoder(model_version=self.AE_model_version, stats_dir=self.AE_stats_dir)
            
            self.AE_weight=params.get("AE_weight", 0)
            self.AE_logvar_static=params.get("AE_logvar_static", True)
            self.AE.eval()
            
            # Optional: initialize AE_logvar from a precomputed NMC/background-error variance.
            # self.AE_logvar = torch.from_numpy(((Back_err**2).sum(axis=0)/(Back_err.shape[0]-1)).flatten())*0.5
            
            if self.AE_logvar_static:
                self.AE_logvar = self.model.AE_logvar = torch.nn.Parameter((torch.ones((1,69,32,64)).float()))
            else:
                self.logvar_head = self.model.logvar_head = ExtraLogVar(69)

            self.AE_max_logvar = self.model.max_logvarr = torch.nn.Parameter((torch.ones((1,69,32,64)).float() / 2))
            self.AE_min_logvar = self.model.min_logvarr = torch.nn.Parameter((-torch.ones((1,69,32,64)).float() * 10))

        if self.loss_type == "Possloss":
            output_dim = self.params['network_params']["out_chans"]
            img_size = self.params['network_params'].get("img_size", [32, 64])
            self.max_logvar = self.model.max_logvar = torch.nn.Parameter((torch.ones((1, output_dim*img_size[-2]*img_size[-1]//2)).float() / 2))
            self.min_logvar = self.model.min_logvar = torch.nn.Parameter((-torch.ones((1, output_dim*img_size[-2]*img_size[-1]//2)).float() * 10))


    def to(self, device):
        self.device = device
        self.model.to(device)
        # self.AE_max_logvar.to(device)
        # self.AE_min_logvar.to(device)
        if self.AE_loss:
            self.AE.to(device)
            if self.AE_checkpoint_path is None:
                raise ValueError("AE_loss=True requires model.AE_checkpoint_path or model.extra_params.AE_checkpoint_path.")
            self.AE.load_state_dict(torch.load(self.AE_checkpoint_path, map_location=device))
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def data_preprocess(self, data,):
        inp = [data[0]]
        if self.two_step_training == False:
            for i in range(1, len(data)-1):
                inp.append(data[i])
            inp = torch.cat(inp, dim=1).float().to(self.device, non_blocking=True)

            tar_step1 = data[-1].float().to(self.device, non_blocking=True)
            # inp, tar_step1 = [x.float().to(self.device, non_blocking=True) for x in data]
            # tar_step1 = tar_step1[:,self.constants_len:]
            tar_step2 = None

        else:
            inp = torch.cat(inp, dim=1).float().to(self.device, non_blocking=True)
            tar_step1 = data[1].float().to(self.device, non_blocking=True)
            tar_step2 = []
            for i in range(1, 1+self.epoch_pred_len):
                tar_step2.append(data[i].float().to(self.device, non_blocking=True))

        tar_step1 = tar_step1
        return inp, tar_step1, tar_step2


    def clear_child_process(self):
        current_process = psutil.Process()
        for child in current_process.children():
            child.terminate()

    def loss_for_training(self, predict, target):
        if self.AE_loss and self.AE_only:
            AE_loss = self.latent_loss(predict, target)
            return AE_loss

        if self.AE_loss and not self.AE_only:
            AE_loss = self.latent_loss(predict, target)
            return self.loss(predict, target) + AE_loss
        else:
            return self.loss(predict, target)

    def latent_loss(self, pred, target):
        mean, log_var = pred.chunk(2, dim = 1)

        latent_pred = self.AE.encode(mean).mode()
        latent_target = self.AE.encode(target).mode()

        # if utils.get_world_size() > 1 and utils.get_rank() == 0:
        #     print('--------------------------')
        #     pred_recon = self.AE.decode(latent_pred)
        #     target_recon = self.AE.decode(latent_target)

        #     # print(((pred_recon[0,0]-target_recon[0][0])/target_recon[0][0]).mean())
        #     # print(((pred_recon[0,3]-target_recon[0][3])/target_recon[0][3]).mean())
        #     np.save('target_recon.npy', pred_recon.detach().cpu().numpy())
            

        # latent_pred = checkpoint(self.AE.encode, mean).mode()
        # latent_target = checkpoint(self.AE.encode, target).mode()

        if self.AE_logvar_static:
            AE_logvar =self.AE_logvar
        else:
            AE_logvar =self.logvar_head(latent_target)

        # latent_pred = latent_pred.reshape(latent_pred.shape[0],-1)
        # latent_target = latent_target.reshape(latent_target.shape[0],-1)

        log_var = self.AE_max_logvar - F.softplus(self.AE_max_logvar - AE_logvar)
        log_var = self.AE_min_logvar + F.softplus(log_var - self.AE_min_logvar)


        inv_var = torch.exp(-1* log_var.to(self.device))

        mse_loss = torch.mean(torch.pow(latent_pred - latent_target, 2) * inv_var, dim=(-1, -2, -3))
        var_loss = torch.mean(log_var, dim=(-1, -2, -3))
        total_loss = mse_loss + var_loss

        total_loss += 0.01 * torch.mean(self.AE_max_logvar) - 0.01 * torch.mean(self.AE_min_logvar)
        return self.AE_weight * torch.mean(total_loss)

    # def latent_loss(self, pred, target):
    #     mean, log_var = pred.chunk(2, dim = 1)
    #     latent_pred = self.AE.encode(mean).mode()
    #     latent_target = self.AE.encode(target).mode()
    #     # latent_pred = checkpoint(self.AE.encode, mean).mode()
    #     # latent_target = checkpoint(self.AE.encode, target).mode()

    #     latent_pred = latent_pred.reshape(latent_pred.shape[0],-1)
    #     latent_target = latent_target.reshape(latent_pred.shape[0],-1)
    #     return self.AE_weight * torch.mean((torch.pow(latent_pred - latent_target, 2)/(self.B_flatten.unsqueeze(0).to(self.device)+1e-6))/2)

    def LpLoss(self, pred, target):
        num_examples = pred.size()[0]

        ppred, _ = pred.chunk(2, dim = 1)
        diff_norms = torch.norm(ppred.reshape(num_examples,-1) - target.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(target.reshape(num_examples,-1), 2, 1)

        return torch.mean(diff_norms/y_norms)


    def Possloss(self, pred, target, **kwargs):
        # print(pred.shape, target.shape, self.max_logvar.shape, self.min_logvar.shape)
        inc_var_loss = kwargs.get("inc_var_loss", True)
        loss_weight = kwargs.get("weight", None)
        
        num_examples = pred.size()[0]

        mean, log_var = pred.chunk(2, dim = 1)
        # log_var = torch.tanh(log_var)


        # mean = mean.reshape(num_examples, -1)
        log_var = log_var.reshape(num_examples, -1)
        # target = target.reshape(num_examples, -1)


        log_var = self.max_logvar - F.softplus(self.max_logvar - log_var)
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)

        log_var = log_var.reshape(*(target.shape))

        inv_var = torch.exp(-log_var)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.pow(mean - target, 2) * inv_var, dim=(-1, -2, -3))
            var_loss = torch.mean(log_var, dim=(-1, -2, -3))
            # mse_loss = torch.mean(torch.pow(mean - target, 2) * inv_var * weight)
            # var_loss = torch.mean(log_var * weight)

            # mse_loss = torch.mean(torch.mean(torch.pow(mean - target, 2) * inv_var, dim=-1), dim=-1)
            # var_loss = torch.mean(torch.mean(log_var, dim=-1), dim=-1)
            total_loss = mse_loss + var_loss
        else:
            mse_loss = torch.mean(torch.pow(mean - target, 2), dim=(-1,-2,-3))
            # mse_loss = torch.mean(torch.pow(mean - target, 2), dim=(1, 2))
            total_loss = mse_loss
            
        total_loss += 0.01 * torch.mean(self.max_logvar) - 0.01 * torch.mean(self.min_logvar)

        if loss_weight is not None:
            total_loss = total_loss * (2 ** (torch.tensor(loss_weight.astype(np.float32)).to(self.device) / 8))

        return torch.mean(total_loss)

    def train_one_step(self, batch_data, step, epoch):
        self.epoch_pred_len = np.min([epoch//(self.max_epoches//(self.pred_len-self.min_pred_len+1)) + self.min_pred_len, (self.pred_len)])
        # self.epoch_pred_len = 12
        # self.epoch_pred_len = self.pred_len

        inp, tar_step1, tar_step2 = self.data_preprocess(batch_data,)

        if self.two_step_training:
            step_two_loss = 0.0
            pred = self.model.module.multi_forward(inp, self.epoch_pred_len)
            # pred = self.model.multi_forward(inp, self.epoch_pred_len)
            # pred = checkpoint(self.model.module.multi_forward, predict, self.epoch_pred_len - 1)
            step_one_loss = None
            step_two_loss = self.loss_for_training(pred, torch.cat(tar_step2,axis=0))

            predict = pred[:tar_step2[0].shape[0]]
            step_one_loss = self.loss_for_training(predict, tar_step1)
            loss = step_two_loss
        else:
            predict = self.model(inp)
            step_one_loss = self.loss_for_training(predict, tar_step1)
            step_two_loss = None
            loss = step_one_loss

        # if self.AE_loss:
        #     AE_loss = self.latent_loss(predict, tar_step1)
        #     # print(loss.item(),AE_loss.item())
        #     loss += AE_loss # *(epoch/self.max_epoches)

        self.optimizer.zero_grad()

        loss.backward()

        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: grad mean = {param.grad.abs().mean().item():.4e}")
        #     else:
        #         print(f"{name}: grad is None")

        self.optimizer.step()

        return {'Possloss': loss.item(), "step_one_loss": step_one_loss.item(), "step_two_loss": step_two_loss.item() if self.two_step_training else 0}

    def test_one_step(self, batch_data,epoch):
        self.epoch_pred_len = self.pred_len

        inp, tar_step1, tar_step2 = self.data_preprocess(batch_data,)

        if self.two_step_training:
            step_two_loss = 0.0
            pred = self.model.module.multi_forward(inp, self.epoch_pred_len)
            # pred = checkpoint(self.model.module.multi_forward, predict, self.epoch_pred_len - 1)
            step_one_loss = None

            step_two_loss = self.loss(pred, torch.cat(tar_step2,axis=0))

            predict = pred[:tar_step2[0].shape[0]]
            step_one_loss = self.loss(predict, tar_step1)

            loss = step_two_loss
        else:
            predict = self.model(inp)
            step_one_loss = self.loss(predict, tar_step1)
            step_two_loss = None
            loss = step_one_loss

        data_dict = {}
        data_dict['gt'] = tar_step1
        data_dict['pred'] = predict[:,:tar_step1.shape[1]]

        metrics_loss = self.eval_metrics.evaluate_batch(data_dict)
        metrics_loss.update({'lp_loss': loss.item(), "step_one_loss": step_one_loss.item(), "step_two_loss": step_two_loss.item() if step_two_loss !=None else 0})
        # return metrics_loss
        return {'Possloss': loss.item(), "step_one_loss": step_one_loss.item(), "step_two_loss": step_two_loss.item() if step_two_loss !=None else 0}

    def train_one_epoch(self, train_data_loader, epoch, max_epoches):
        if self.scheduler != "constant":
           self.lr_scheduler.step(epoch)
        # test_logger = {}

        end_time = time.time()
        self.model.train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
        data_time = utils.SmoothedValue(fmt='{avg:.3f}')
        max_step = len(train_data_loader)

        header = 'Epoch [{epoch}/{max_epoches}][{step}/{max_step}]'
        for step, batch in enumerate(train_data_loader):
            if self.scheduler != "constant":
                self.lr_scheduler.step(epoch*max_step+step)
            # record data read time
            data_time.update(time.time() - end_time)
            
            loss = self.train_one_step(batch, step, epoch)

            if utils.get_world_size() > 1 and self.use_checkpoint is False:
                utils.check_ddp_consistency(self.model)

            # record loss and time
            metric_logger.update(**loss)
            iter_time.update(time.time() - end_time)
            end_time = time.time()

            # output to logger
            if (step+1) % 100 == 0 or step+1 == max_step:
                eta_seconds = iter_time.global_avg * (max_step - step - 1 + max_step * (max_epoches-epoch-1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.logger.info(
                    metric_logger.delimiter.join(
                        [header,
                        "lr: {lr}",
                        "eta: {eta}",
                        "time: {time}",
                        "data: {data}",
                        "memory: {memory:.0f}",
                        "{meters}"
                        ]
                    ).format(
                        epoch=epoch+1, max_epoches=max_epoches, step=step+1, max_step=max_step,
                        lr=self.optimizer.param_groups[0]["lr"],
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                        meters=str(metric_logger)
                    ))
                
    def load_checkpoint(self, checkpoint_path, load_model=True, load_optimizer=True, load_scheduler=True, load_epoch=True, load_metric_best=True):
        if os.path.exists(checkpoint_path):
            checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            self.logger.info("checkpoint is not exist")
            return
        checkpoint_model = checkpoint_dict['model']

        print('--------------------------------')
        print('loading from checkpoint')
        if load_model:
            new_state_dict = OrderedDict()
            for k, v in checkpoint_model.items():
                if "module" == k[:6]:
                    name = k[7:]
                elif "_orig_mod." == k[:10]:
                    name = k[10:]
                else:
                    name = k
                # if not name == "max_logvar" and not name == "min_logvar":
                #     new_state_dict[name] = v
                if "logvar" not in name:
                    new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
            self.model = torch.compile(self.model, fullgraph=False, dynamic=False)
        
        if load_optimizer:
            checkpoint_optimizer = checkpoint_dict['optimizer']
            self.optimizer.load_state_dict(checkpoint_optimizer)
        if load_scheduler:
            checkpoint_lr_scheduler = checkpoint_dict['lr_scheduler']
            self.lr_scheduler.load_state_dict(checkpoint_lr_scheduler)
        if load_epoch:
            self.begin_epoch = checkpoint_dict['epoch']
        if load_metric_best and 'metric_best' in checkpoint_dict:
            self.metric_best = checkpoint_dict['metric_best']
        if 'amp_scaler' in checkpoint_dict:
            self.gscaler.load_state_dict(checkpoint_dict['amp_scaler'])
        self.logger.info("last epoch:{epoch}, metric best:{metric_best}".format(epoch=checkpoint_dict['epoch'], metric_best=checkpoint_dict['metric_best']))


    def save_checkpoint(self, epoch, checkpoint_savedir, save_type='save_best'): 
        checkpoint_savedir = Path(checkpoint_savedir)
        # checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_best.pth' \
        #                     if save_type == 'save_best' else 'checkpoint_latest.pth')

        if save_type == "save_epoch":
            checkpoint_path = checkpoint_savedir / 'checkpoint_epoch{:02d}.pth'.format(epoch)
        elif save_type == "save_best":
            checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_best.pth')
        else:
            checkpoint_path = checkpoint_savedir / '{}'.format('checkpoint_latest.pth')



        # print(save_type, checkpoint_path)

        if utils.get_world_size() > 1 and utils.get_rank() == 0:
            state_dict = self.model.module.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('logvar_head')}
            if self.scheduler != "constant":
                torch.save(
                    {
                    'epoch':            epoch+1,
                    'model':            filtered_state_dict,
                    'optimizer':        self.optimizer.state_dict(),
                    'lr_scheduler':     self.lr_scheduler.state_dict(),
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                    }, checkpoint_path
                )
            else:
                torch.save(
                    {
                    'epoch':            epoch+1,
                    'model':            filtered_state_dict,
                    'optimizer':        self.optimizer.state_dict(),
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                    }, checkpoint_path
                )
        elif utils.get_world_size() == 1:
            state_dict = self.model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('logvar_head')}
            if self.scheduler != "constant":
                torch.save(
                    {
                    'epoch':            epoch+1,
                    'model':            filtered_state_dict,
                    'optimizer':        self.optimizer.state_dict(),
                    'lr_scheduler':     self.lr_scheduler.state_dict(),
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                    }, checkpoint_path
                )
            else:
                torch.save(
                    {
                    'epoch':            epoch+1,
                    'model':            filtered_state_dict,
                    'optimizer':        self.optimizer.state_dict(),
                    'metric_best':      self.metric_best,
                    'amp_scaler':       self.gscaler.state_dict(),
                    }, checkpoint_path
                )

    def whether_save_best(self, metric_logger):
        metric_now = metric_logger.meters[self.save_best_param].global_avg
        if self.metric_best is None:
            self.metric_best = metric_now
            return True
        if metric_now < self.metric_best:
            self.metric_best = metric_now
            return True
        return False

    def trainer(self, builder, max_epoches, checkpoint_savedir=None, save_ceph=False, resume=False):

        for epoch in range(self.begin_epoch, max_epoches):
            self.clear_child_process()
            train_data_loader, train_set, train_sampler = builder.get_dataloader(split = 'train')
            self.max_epoches = max_epoches
            train_data_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(train_data_loader, epoch, max_epoches)
            del train_data_loader, train_set, train_sampler
   
            self.clear_child_process()
            test_data_loader, test_set, test_sampler = builder.get_dataloader(split = 'test')
            metric_logger = self.test(test_data_loader, epoch)

            del test_data_loader, test_set, test_sampler

            # save model
            if checkpoint_savedir is not None:
                if self.save_epoch:
                    self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_epoch')
                else:
                    if self.whether_save_best(metric_logger):
                        self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_best')
                    if (epoch + 1) % 1 == 0:
                        self.save_checkpoint(epoch, checkpoint_savedir, save_type='save_latest')
            # end_time = time.time()
            # print("save model time", end_time - begin_time2)


    @torch.no_grad()
    def test(self, test_data_loader, epoch):
        metric_logger = utils.MetricLogger(delimiter="  ")
        # set model to eval
        self.model.eval()

        for step, batch in enumerate(test_data_loader):
            loss = self.test_one_step(batch,epoch)
            metric_logger.update(**loss)
            # print(self.save_best_param)
            # print(metric_logger.meters)
            # print(metric_logger.meters[self.save_best_param].count)
        
        self.logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))

        return metric_logger


    def test_final(self, valid_data_loader, predict_length):
        metric_logger = []
        for i in range(predict_length):
            metric_logger.append(utils.MetricLogger(delimiter="  "))
        # set model to eval
        self.model.eval()

        index = 0
        total_step = len(valid_data_loader)

        for step, batch in enumerate(valid_data_loader):
            batch_len = batch.shape[0]
            losses = self.multi_step_predict(batch, index, batch_len)
            for i in range(len(losses)):
                metric_logger[i].update(**losses[i])
            index += batch_len

            self.logger.info("#"*80)
            self.logger.info(step)
            if step % 10 == 0 or step == total_step-1:
                for i in range(predict_length):
                    self.logger.info('  '.join(
                            [f'final valid {i}th step predict (val stats)',
                            "{meters}"]).format(
                                meters=str(metric_logger[i])
                            ))

        return None


    def multi_step_predict(self, batch_data, index, batch_len):
        # last_inp = tensor_data[:inp_length-1].float().to(self.device, non_blocking=True).transpose(0,1).transpose(1,2)
        # pred = tensor_data[inp_length-1:inp_length].float().to(self.device, non_blocking=True).transpose(0,1).transpose(1,2)

        inp = batch_data[:, :2].float().to(self.device, non_blocking=True)
        metrics_losses = []

        for i in range(2, batch_data.shape[1]):
            tar = batch_data[:, i].float().to(self.device, non_blocking=True)

            pred = self.model(inp.flatten(1,2))
            data_dict = {}
            data_dict['gt'] = tar
            data_dict['pred'] = pred
            data_dict['clim_mean'] = None
            data_dict['std'] = None
            metrics_losses.append(self.eval_metrics.evaluate_batch(data_dict))
            inp = torch.cat((inp[:,1:], pred.unsqueeze(1)), dim=1)
        
        return metrics_losses
