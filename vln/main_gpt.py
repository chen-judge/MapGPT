import os
import json
import time
from collections import defaultdict

from vln.data_utils import construct_instrs
from vln.env import R2RNavBatch
from vln.parser import parse_args

from utils.data import set_random_seed
from utils.logger import write_to_record_file

from vln.gpt_agent import GPTNavAgent


def build_dataset(args, rank=0, is_test=True):
    dataset_class = R2RNavBatch
    split = args.split
    val_envs = {}

    if 'processed' in split:
        with open(os.path.join(args.anno_dir, split+'.json'), 'r') as f:
            val_instr_data = json.load(f)

        if args.end is None:
            args.end = len(val_instr_data)
        val_instr_data = val_instr_data[args.start:args.end]
        print(f'------------------ Evaluate {args.start}-{args.end} in {split} ------------------')

    else:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split],
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )

    val_env = dataset_class(
        val_instr_data, args.connectivity_dir, batch_size=args.batch_size,
        seed=args.seed+rank,
        name=split, args=args,
    )   # evaluation using all objects
    val_envs[split] = val_env

    return val_envs


def valid(args, val_envs, rank=0):

    default_gpu = None
    agent_class = GPTNavAgent

    agent = agent_class(args, list(val_envs.values())[0], rank=rank)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():

        print(f"Start evaluating {env_name}")
        prefix = 'submit' if args.detailed_output is False else 'detail'
        if os.path.exists(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name))):
            print('Path already exist...')
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        start_time = time.time()
        print('running...')
        agent.test(args=args)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds, args.dataset)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)

            if args.submit or args.save_pred:
                json.dump(
                    preds,
                    open(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name)), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
                

def main():
    args = parse_args()
    set_random_seed(args.seed)
    val_envs = build_dataset(args)
    valid(args, val_envs)


if __name__ == '__main__':
    main()
