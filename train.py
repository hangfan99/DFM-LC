import argparse
import os
import psutil
import torch
from utils.builder import ConfigBuilder
import utils.misc as utils
import yaml
from utils.logger import get_logger


#----------------------------------------------------------------------------


def clear_child_process():
    current_process = psutil.Process()
    for child in current_process.children():
        child.terminate()

def subprocess_fn(args):
    utils.setup_seed(args.seed * args.world_size + args.rank)

    print('rank:', args.rank)

    logger = get_logger("train", args.run_dir, utils.get_rank(), filename='iter.log', resume=args.resume)

    # args.logger = logger
    args.cfg_params["logger"] = logger

    # build config
    logger.info('Building config ...')
    builder = ConfigBuilder(**args.cfg_params)

    logger.info('Building dataloaders ...')


    train_dataloader, train_set, train_sampler = builder.get_dataloader(split = 'train')
    logger.info('Train dataloaders build complete')

    steps_per_epoch = len(train_dataloader)

    del train_dataloader, train_set, train_sampler
    clear_child_process()

    model_params = args.cfg_params['model']
    if 'lr_params' in model_params:
        lr_scheduler_params = model_params['lr_params']
        if 'by_step' in lr_scheduler_params:
            if lr_scheduler_params['by_step']:
                for key1 in lr_scheduler_params:
                    if "epochs" in key1:
                        lr_scheduler_params[key1] *= steps_per_epoch
    
    # print(lr_scheduler_params)

    # build model
    logger.info('Building models ...')
    model = builder.get_model()

    model_checkpoint = os.path.join(args.run_dir, 'checkpoint_latest.pth')
    if args.resume:
        model.load_checkpoint(model_checkpoint)

    model_without_ddp = utils.DistributedParallel_Model(model, args.local_rank)

    if args.world_size > 1:
        utils.check_ddp_consistency(model_without_ddp.model)

    params = [p for p in model_without_ddp.model.parameters() if p.requires_grad]
    cnt_params = sum([p.numel() for p in params])
    # print("params {key}:".format(key=key), cnt_params)
    logger.info("params: {cnt_params}".format(cnt_params=cnt_params))



    # valid_dataloader = builder.get_dataloader(split = 'valid')
    # logger.info('valid dataloaders build complete')
    logger.info('begin training ...')

    # model_without_ddp.stat()
    model_without_ddp.trainer(builder, builder.get_max_epoch(), checkpoint_savedir= args.run_dir, resume=args.resume)


def main(args):
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        args.distributed = False
        args.local_rank = 0
        torch.cuda.set_device(args.local_rank)
    desc = f'world_size{args.world_size:d}'

    if args.desc is not None:
        desc += f'-{args.desc}'

    alg_dir = args.cfg.split("/")[-1].split(".")[0]
    args.outdir = args.outdir + "/" + alg_dir
    run_dir = os.path.join(args.outdir, f'{desc}')
    relative_checkpoint_dir = alg_dir + "/" + f'{desc}'
    args.relative_checkpoint_dir = relative_checkpoint_dir
    print(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    train_config_file = os.path.join(run_dir, 'training_options.yaml')

    if (not args.resume) or args.resume_from_config or (not os.path.exists(train_config_file)):
        print("load yaml from config")
        with open(args.cfg, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

        
    else:
        print("load yaml from resume")
        with open(train_config_file, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
        del_keys = []
        for key in cfg_params:
            if key in args:
                del_keys.append(key)
        for key in del_keys:
            del cfg_params[key]

    cfg_params['dataloader']['num_workers'] = args.per_cpus

    if args.ae_weight >0 :
        cfg_params['model']['AE_weight'] = args.ae_weight
    if args.lr > 0:
        cfg_params['model']['optimizer_params']['params']['lr'] = args.lr

    if args.rank == 0:
        with open(os.path.join(run_dir, 'training_options.yaml'), 'wt') as f:
            yaml.dump(vars(args), f, indent=2, sort_keys=False)
            yaml.dump(cfg_params, f, indent=2, sort_keys=False)

    args.cfg_params = cfg_params
    args.run_dir = run_dir

    print('Launching processes...')
    subprocess_fn(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',                     action = "store_true",                                                  help = 'resume')
    parser.add_argument('--resume_from_config',         action = "store_true",                                                  help = 'resume from config')
    parser.add_argument('--seed',                       type = int,     default = 0,                                            help = 'seed')
    parser.add_argument('--cuda',                       type = int,     default = 0,                                            help = 'cuda id')
    parser.add_argument('--world_size',                 type = int,     default = 1,                                            help = 'Number of progress')
    parser.add_argument('--per_cpus',                   type = int,     default = 1,                                            help = 'Number of perCPUs to use')
    # parser.add_argument('--world_size',     type = int,     default = -1,                                           help = 'number of distributed processes')
    parser.add_argument('--init_method',                type = str,     default='tcp://127.0.0.1:23456',                        help = 'multi process init method')
    parser.add_argument('--outdir',                     type = str,     default='output',  help = 'Where to save the results')
    parser.add_argument('--cfg', '-c',                  type = str,     default = './configs/DFM-LC.yaml',      help = 'path to the configuration file')
    parser.add_argument('--desc',                       type=str,       default='test',                                          help = 'String to include in result dir name')
    parser.add_argument('--ae_weight',                       type=float,       default=-1.0,                                          help = 'the weight for AE constrain')
    parser.add_argument('--lr',                         type=float,       default=-1.0,                                          help = 'the learning rate')


    args = parser.parse_args()

    main(args)
