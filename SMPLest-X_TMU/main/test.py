import argparse
import torch
import torch.backends.cudnn as cudnn
from main.config import Config
import os.path as osp
import datetime
from pathlib import Path
import time

from main.base import Tester
from human_models.human_models import SMPL, SMPLX
from tqdm import tqdm
import tmu

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus', default=1)
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--result_path', type=str, default='output/test')
    parser.add_argument('--ckpt_idx', type=int, default=0)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--testset', type=str, default='EHF')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--r', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cudnn.benchmark = True
    
    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'{args.exp_name}_{time_str}'
    root_dir = ""
    config_path = "pretrained_models/smplest_x_h/config_base.py"
    cfg = Config.load_config(config_path)
    checkpoint_path = "pretrained_models/smplest_x_h/smplest_x_h.pth.tar"

    if args.testset in ['AGORA_test', 'BEDLAM_test']:
        print(f'Test on {args.testset} set...')
    
    new_config = {
        "data": {
            "testset": str(args.testset),
            "use_cache": args.use_cache,
        },
        "test":{
            "test_batch_size": int(args.test_batch_size),
        },
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'output_dir': osp.join(root_dir, 'outputs', exp_name),
            'model_dir': osp.join(root_dir, 'outputs', exp_name, 'model_dump'),
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),
            'result_dir': osp.join(root_dir, 'outputs', exp_name, 'result'),
        }
    }

    cfg.update_config(new_config)
    cfg.prepare_log()
    cfg.dump_config()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    tester = Tester(cfg)
    tester.logger.info(f"Using 1 GPU with bs={cfg.test.test_batch_size} per GPU.")
    tester.logger.info(f'Testing [{checkpoint_path}] on datasets [{cfg.data.testset}]')

    tester._make_batch_generator()
    tester._make_model()
    print(f"Model is on device: {next(tester.model.parameters()).device}.")

    """ ------------------------ Apply TMU unmerge ------------------------ """
    tmu.patch.timm(tester.model.module.encoder, trace_source=True)
    tester.model.module.encoder.r = args.r
    print(f"Applied TMU Unmerge. r is: {tester.model.module.encoder.r}")
    """ end """

    total_time = 0
    total_count = 0

    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        # Timed run
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            model_out = tester.model(inputs, targets, meta_info, 'test')
        torch.cuda.synchronize()
        end_time = time.time()

        batch_time = end_time - start_time
        total_time += batch_time

        batch_size = model_out['img'].shape[0]
        total_count += batch_size

        out = {}
        for k, v in model_out.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.cpu().numpy()
            elif isinstance(v, list):
                out[k] = v
            else:
                raise ValueError('Undefined type in out. Key: {}; Type: {}.'.format(k, type(v)))

        out = [{k: v[bid] for k, v in out.items()} for bid in range(batch_size)]

        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx, smpl_x)
        for k, v in cur_eval_result.items():
            if k in eval_result:
                eval_result[k] += v
            else:
                eval_result[k] = v
        cur_sample_idx += len(out)

    tester._print_eval_result(eval_result)
    images_per_second = total_count / total_time
    print(f"Total samples processed: {total_count}")
    print(f"Throughput: {images_per_second:.4f} image/s")


if __name__ == "__main__":
    main()