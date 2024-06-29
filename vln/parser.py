import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--dataset', type=str, default='r2r')
    parser.add_argument('--output_dir', type=str, default='default', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    # Data preparation
    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')
    parser.add_argument('--max_instr_len', type=int, default=200)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=1)  # only support bach_size=1

    # Submision configuration
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=False)
    parser.add_argument("--save_pred", action='store_true', default=False)

    # LLM
    parser.add_argument('--llm', type=str, default='')
    parser.add_argument('--response_format', type=str, default='str', choices=['str', 'json'])
    parser.add_argument('--img_root', type=str, default=None)
    parser.add_argument("--split", type=str, default='MapGPT_72_scenes_processed')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--stop_after', type=int, default=3)
    parser.add_argument('--max_tokens', type=int, default=1000)

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    if args.dataset == 'r2r':
        args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')
    elif args.dataset == 'reverie':
        args.anno_dir = os.path.join(ROOTDIR, 'REVERIE', 'annotations')

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')
    args.vis_dir = os.path.join(args.output_dir, 'vis')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    return args

