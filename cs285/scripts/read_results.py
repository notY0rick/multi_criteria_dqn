import glob
import tensorflow as tf


def get_section_tags(file):
    all_tags = set()
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            all_tags.add(v.tag)
    return all_tags


def get_data(file):
    for e in tf.train.summary_iterator(file):
        print(e.summary.value.tag)


print(get_section_tags("/Users/yorio/Documents/cs/berkeley/multi-criteria-dqn/data/42_eps_0.5_pruned_sparse_LunarLander-Cu\
            stomizable0.0-0.0-0.0-0.0-1.0_11-12-2022_04-42-46/events.out.tfevents.1670762567.Yoricks-MacBook-Pro.local"))

# def get_section_results(file):
#     """
#         requires tensorflow==1.12.0
#     """
#     X = []
#     Y = []
#     for e in tf.train.summary_iterator(file):
#         for v in e.summary.value:
#             if v.tag == 'Train_EnvstepsSoFar':
#                 X.append(v.simple_value)
#             elif v.tag == 'Eval_AverageReturn':
#                 Y.append(v.simple_value)
#     return X, Y
#
# if __name__ == '__main__':
#     import glob
#
#     logdir = 'data/q1_lb_rtg_na_CartPole-v0_13-09-2020_23-32-10/events*'
#     eventfile = glob.glob(logdir)[0]
#
#     X, Y = get_section_results(eventfile)
#     for i, (x, y) in enumerate(zip(X, Y)):
#         print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))