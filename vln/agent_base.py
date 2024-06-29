import json
import os
import time
from utils.logger import write_to_record_file


class BaseAgent(object):
    ''' Base class for an agent to generate and save trajectories. '''

    def __init__(self, env):
        self.env = env
        self.results = {}

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'trajectory': v['path'], 'a_t': v['a_t']})
            if detailed_output:
                output[-1]['details'] = v['details']
        return output

    def rollout(self, **args):
        raise NotImplementedError

    def test(self, iters=None, args=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.results = {}
        looped = False

        while True:
            for traj in self.rollout(**kwargs):
                if traj is None:
                    looped = True
                else:
                    self.loss = 0
                    self.results[traj['instr_id']] = traj

            if looped:
                break

            preds = self.get_results(detailed_output=args.detailed_output)
            current_pred = [preds[-1]]

            # evaluating current case
            score_summary, current_metrics = self.env.eval_metrics(current_pred, args.dataset)
            loss_str = "Current case  -"
            for metric, val in score_summary.items():
                loss_str += '  %s: %.2f' % (metric, val)
            print(loss_str)

            # add evaluation result
            instr_id = preds[-1]['instr_id']
            scan, gt_traj = self.env.gt_trajs[instr_id]

            preds[-1]['scan'] = scan
            preds[-1]['gt_traj'] = gt_traj
            preds[-1]['evaluation'] = current_metrics

            if args.save_pred:
                json.dump(
                    preds[-1],
                    open(os.path.join(args.pred_dir, "case_InstrID_%s.json" % instr_id), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )

        # evaluating all cases
        score_summary, _ = self.env.eval_metrics(preds, args.dataset)
        loss_str = "All cases  -"
        for metric, val in score_summary.items():
            loss_str += '  %s: %.2f' % (metric, val)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(loss_str + '\n', record_file)





