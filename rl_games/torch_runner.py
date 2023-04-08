import os
import time
import numpy as np
import random
from copy import deepcopy
import torch
#import yaml
from gym import Env
from typing import Optional, Dict
from dataclasses import asdict

#from rl_games import envs
from rl_games.common import object_factory
from rl_games.common import tr_helpers

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent

from rl_games.configs.definitions import RunnerConfig, AlgorithmConfig, PolicyConfig


def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        agent.restore(args['checkpoint'])

def _override_sigma(agent, args):
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Print cannot set new sigma because fixed_sigma is False')


class Runner:
    def __init__(self,
        env: Optional[Env] = None,
        runner_cfg: Optional[RunnerConfig] = None,
        algorithm_cfg: Optional[AlgorithmConfig] = None,
        policy_cfg: Optional[PolicyConfig] = None,
        algo_observer: Optional[DefaultAlgoObserver] = None,
    ):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()
        torch.backends.cudnn.benchmark = True
        ### it didnot help for lots for openai gym envs anyway :(
        #torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)

        self._init_params_from_dataclass_configs(runner_cfg, algorithm_cfg, policy_cfg)
        self._update_params_from_env(env)


    def _init_params_from_dataclass_configs(self, runner_cfg=None, algorithm_cfg=None, policy_cfg=None):
        params = {}
        params['config'] = {}

        if runner_cfg:
            self.seed = runner_cfg.seed
            self._process_seed()
            if runner_cfg.multi_gpu:
                self.seed += int(os.getenv("LOCAL_RANK", "0"))
                print(f"self.seed = {self.seed}")
            params['seed'] = self.seed

            params['load_checkpoint'] = runner_cfg.load_checkpoint
            params['load_path'] = runner_cfg.load_path

            params['config'].update(asdict(runner_cfg))

        if algorithm_cfg:
            params['algo'] = asdict(algorithm_cfg)
            params['algo']['name'] = algorithm_cfg.name
            self.algo_name = algorithm_cfg.name

            params['config'].update(asdict(algorithm_cfg))
            self.algo_params = params['algo']

        if policy_cfg:
            params['model'] = {
                'name': policy_cfg.name,
            }
            params['network'] = {
                'name': policy_cfg.network.name,
                'separate': policy_cfg.network.separate,
                'space': {
                    'continuous': {
                        'mu_activation': policy_cfg.network.space.mu_activation,
                        'sigma_activation': policy_cfg.network.space.mu_activation,
                        'mu_init': {
                            'name': policy_cfg.network.space.mu_init,
                        },
                        'sigma_init': {
                            'name': policy_cfg.network.space.sigma_init,
                            'val': policy_cfg.network.space.sigma_val,
                        },
                        'fixed_sigma': policy_cfg.network.space.fixed_sigma,
                    },
                },
                'mlp': {
                    'units': policy_cfg.network.mlp.units,
                    'activation': policy_cfg.network.mlp.activation,
                    'd2rl': policy_cfg.network.mlp.d2rl,
                    'initializer': {
                        'name': policy_cfg.network.mlp.initializer,
                    },
                    'regularizer': {
                        'name': policy_cfg.network.mlp.regularizer,
                    },
                },
            }
                        
            # params[config] seems like it just gets evertything
            params['config'].update(asdict(policy_cfg))

        params['config']['reward_shaper'] = {'scale_value': algorithm_cfg.reward_shaper_scale_val}
        #TODO: possibly necessary
        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def _update_params_from_env(self, env):
        
        self.params['config']['vec_env'] = env
        self.params['config']['env_info'] = env.get_env_info()
        self.params['config']['num_actors'] = env.get_number_of_agents()
        self.params['config']['env_name'] = 'rlgpu' 

        #if env.num_envs:
        #    self.params['config']['num_actors'] = env.get_number_of_agents()
        #elif env.env.num_envs:
        #    self.params['config']['num_actors'] = env.env.num_envs
        #    self.params['config']['env_name'] = env.
        #else:
        #    raise ValueError('Env is missing num_actors attribute!')

    def _process_seed(self):
        if self.seed is None:
            self.seed = int(time.time())

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def reset(self):
        pass

    def load_config(self, params):
        self.seed = params.get('seed', None)
        self._process_seed()

        if params["config"].get('multi_gpu', False):
            self.seed += int(os.getenv("LOCAL_RANK", "0"))
        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        # deal with environment specific seed if applicable
        if 'env_config' in params['config']:
            if not 'seed' in params['config']['env_config']:
                params['config']['env_config']['seed'] = self.seed
            else:
                if params["config"].get('multi_gpu', False):
                    params['config']['env_config']['seed'] += int(os.getenv("LOCAL_RANK", "0"))

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_config):
        config = deepcopy(yaml_config)
        self.default_config = deepcopy(config['params'])
        self.load_config(params=self.default_config)

    def run_train(self, args):
        print('Started to train')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        _restore(agent, args)
        _override_sigma(agent, args)
        agent.train()

    def run_play(self, args):
        print('Started to play')
        player = self.create_player()
        _restore(player, args)
        _override_sigma(player, args)
        player.run()

    def create_player(self):
        return self.player_factory.create(self.algo_name, params=self.params)

    def reset(self):
        pass

    def run(self, args: Optional[Dict] = None, play: bool = False):
        load_path = None

        if not args:
            args = {}

        args.setdefault("checkpoint", None)

        if play:
            self.run_play(args)
        else:
            self.run_train(args)
