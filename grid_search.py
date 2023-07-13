import os
import numpy as np


# example
"""
python cs285/scripts/run_dqn.py \
--exp_name test_pruned_sparse \
--env_name LunarLander-Customizable \
--pruning_file_prefix test_LunarLander-Customizable \
--pruning_eps 0.5 \
--env_rew_weights 0 0 0 0 1 \
--double_q --seed 1
"""


def weights_generator(length, normalize=False):
    weights = np.random.rand(length) * 11
    if not normalize:
        return weights.astype(int)
    weights /= np.sum(weights)
    return weights.astype(int)


def train_pruned(eps_list):
    cmd = "python cs285/scripts/run_dqn.py \
    --exp_name 42_eps_{}_pruned_sparse \
    --env_name LunarLander-Customizable \
    --pruning_file_prefix grid \
    --pruning_eps {} \
    --env_rew_weights 0 0 0 0 1 \
    --double_q --seed 1"

    for i, eps in enumerate(eps_list):
        exp = cmd.format(eps, eps)
        print("Iteration {}".format(i + 1))
        print(exp)
        os.system(exp)


def train_dqn(n):
    dqn_cmd = "python cs285/scripts/run_dqn.py \
    --exp_name {}_default \
    --env_name LunarLander-Customizable \
    --env_rew_weights {} \
    --double_q --seed 1"

    for i in range(n):
        weights = weights_generator(5, normalize=False)
        w = "{} {} {} {} {}".format(np.round(weights[0], decimals=2), np.round(weights[1], decimals=2),
                                    np.round(weights[2], decimals=2), np.round(weights[3], decimals=2),
                                    np.round(weights[4], decimals=2))
        exp_name = "{}_{}_{}_{}_{}".format(np.round(weights[0], decimals=2), np.round(weights[1], decimals=2),
                                           np.round(weights[2], decimals=2), np.round(weights[3], decimals=2),
                                           np.round(weights[4], decimals=2))

        exp = dqn_cmd.format(exp_name, w)
        print("Iteration {}".format(i + 1))
        print(exp)
        os.system(exp)


# train_dqn(30)


train_pruned([0.9])


"""
python cs285/scripts/run_dqn.py \
--exp_name 42_pruned_sparse \
--env_name LunarLander-Customizable \
--pruning_file_prefix grid_ \
--pruning_eps 0.25 \
--env_rew_weights 0 0 0 0 1 \
--double_q --seed 1
"""

"""
python cs285/scripts/run_dqn.py \
--exp_name 42_pruned_sparse \
--env_name LunarLander-Customizable \
--pruning_file_prefix grid_ \
--pruning_eps 0.15 \
--env_rew_weights 0 0 0 0 1 \
--double_q --seed 1
"""