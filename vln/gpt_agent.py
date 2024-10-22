import sys
import numpy as np
from collections import defaultdict
from GPT.one_stage_prompt_manager import OneStagePromptManager
from .agent_base import BaseAgent
from GPT.api import gpt_infer
import json


class GPTNavAgent(BaseAgent):
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args

        self._build_prompt_manager()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
    
    def _build_prompt_manager(self):
        self.prompt_manager = OneStagePromptManager(self.args)
        print('Model version:', self.args.llm)

    def make_equiv_action(self, a_t, obs, traj=None):

        def take_action(i, name):
            if type(name) is int:       # Go to the next viewpoint
                self.env.env.sims[i].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, 'down')
                    src_level -= 1
                while self.env.env.sims[i].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, select_candidate['idx']) # j+1: idx for navigable location

                state = self.env.env.sims[i].getState()[0]
                if traj is not None:
                    traj[i]['path'].append([state.location.viewpointId])

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        batch_size = len(obs)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'a_t': {},
        } for ob in obs]

        if traj[0]['instr_id'] in self.results:
            return [None]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]

        self.prompt_manager.history = ['' for _ in range(self.args.batch_size)]
        self.prompt_manager.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.graph = [{} for _ in range(self.args.batch_size)]
        self.prompt_manager.trajectory = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.planning = [["Navigation has just started, with no planning yet."] for _ in range(self.args.batch_size)]

        for t in range(self.args.max_action_len):
            if t == self.args.max_action_len:
                break

            cand_inputs = self.prompt_manager.make_action_prompt(obs, previous_angle)
            if self.args.response_format == 'str':
                nav_input = self.prompt_manager.make_r2r_prompts(cand_inputs=cand_inputs, obs=obs, t=t)
            elif self.args.response_format == 'json':
                nav_input = self.prompt_manager.make_r2r_json_prompts(cand_inputs=cand_inputs, obs=obs, t=t)
            else:
                raise NotImplemented

            image_list = self.prompt_manager.node_imgs[0]
            environment_prompts = nav_input["prompts"][0]
            print('-------------------- Environment Prompts --------------------')
            print(environment_prompts)

            if self.args.llm == 'gpt-4-vision-preview' and self.args.response_format == 'str':
                # GPT-4V only supports string mode output
                nav_output, tokens = gpt_infer(nav_input["task_description"], environment_prompts, image_list,
                                               self.args.llm, self.args.max_tokens)
                print('-------------------- Output --------------------')
                print(nav_output)
                nav_output = [nav_output]
                a_t = self.prompt_manager.parse_action(nav_output=nav_output,
                                                       only_options_batch=nav_input["only_options"],
                                                       t=t)
                self.prompt_manager.parse_planning(nav_output=nav_output)

            elif self.args.llm == 'gpt-4o-2024-05-13' and self.args.response_format == 'json':
                if len(image_list) > 20:
                    # GPT-4o currently does not support queries with more than 20 images
                    a_t = [0]
                    print('Exceed image limit and stop!')
                else:
                    nav_output, tokens = gpt_infer(nav_input["task_description"], environment_prompts, image_list,
                                                   self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
                    json_output = json.loads(nav_output)
                    a_t = self.prompt_manager.parse_json_action(json_output, nav_input["only_options"], t)
                    self.prompt_manager.parse_json_planning(json_output)
                    print('-------------------- Output --------------------')
                    print(nav_output)

            else:
                raise NotImplemented

            for i in range(batch_size):
                traj[i]['a_t'][t] = a_t[i]

            # Determine stop actions
            a_t_stop = [a_t_i == 0 for a_t_i in a_t]

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i]:
                    cpu_a_t.append(-1)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(a_t[i] - 1)

            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs()

            previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]

            # we only implement batch_size=1
            if a_t[0] == 0:
                break

            self.prompt_manager.make_history(a_t, nav_input, t)

        return traj
