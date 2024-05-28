'''
消除TD3时延违反对成本的影响
'''

import numpy as np

TD3_delay = []
TD3 = []
TD3.append(np.load("../../TD3/dmax/10_20_0.1_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/dmax/10_20_0.15_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/dmax/10_20_0.2_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/dmax/10_20_0.25_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/dmax/10_20_0.3_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/dmax/10_20_0.35_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/dmax/10_20_0.4_0.05/latency_reward.npy"))

for i in range(7):
    TD3_delay.append(np.mean(TD3[i][-50:]))

SAC_delay = []
SAC = []
SAC.append(np.load("../../SAC/dmax/10_20_0.1_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_20_0.15_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_20_0.2_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_20_0.25_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_20_0.3_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_20_0.35_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_20_0.4_0.05/latency_reward.npy"))

for i in range(7):
    SAC_delay.append(np.mean(SAC[i][-50:]))

dmax_gap = [x - y for x, y in zip(TD3_delay, SAC_delay)]


TD3_delay = []
TD3 = []
TD3.append(np.load("../../TD3/prob/10_20_0.2_0.0/latency_reward.npy"))
TD3.append(np.load("../../TD3/prob/10_20_0.2_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/prob/10_20_0.2_0.1/latency_reward.npy"))
TD3.append(np.load("../../TD3/prob/10_20_0.2_0.15/latency_reward.npy"))
TD3.append(np.load("../../TD3/prob/10_20_0.2_0.2/latency_reward.npy"))
TD3.append(np.load("../../TD3/prob/10_20_0.2_0.25/latency_reward.npy"))
TD3.append(np.load("../../TD3/prob/10_20_0.2_0.3/latency_reward.npy"))

for i in range(7):
    TD3_delay.append(np.mean(TD3[i][-50:]))

SAC_delay = []
SAC = []
SAC.append(np.load("../../SAC/prob/10_20_0.2_0.0/latency_reward.npy"))
SAC.append(np.load("../../SAC/prob/10_20_0.2_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/prob/10_20_0.2_0.1/latency_reward.npy"))
SAC.append(np.load("../../SAC/prob/10_20_0.2_0.15/latency_reward.npy"))
SAC.append(np.load("../../SAC/prob/10_20_0.2_0.2/latency_reward.npy"))
SAC.append(np.load("../../SAC/prob/10_20_0.2_0.25/latency_reward.npy"))
SAC.append(np.load("../../SAC/prob/10_20_0.2_0.3/latency_reward.npy"))

for i in range(7):
    SAC_delay.append(np.mean(SAC[i][-50:]))

prob_gap = [x - y for x, y in zip(TD3_delay, SAC_delay)]


TD3_delay = []
TD3 = []
TD3.append(np.load("../../TD3/user_num/8_20_0.2_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/user_num/10_20_0.2_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/user_num/12_20_0.2_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/user_num/14_20_0.2_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/user_num/16_20_0.2_0.05/latency_reward.npy"))

for i in range(5):
    TD3_delay.append(np.mean(TD3[i][-50:]))

SAC_delay = []
SAC = []
SAC.append(np.load("../../SAC/user_num/8_20_0.2_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/user_num/10_20_0.2_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/user_num/12_20_0.2_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/user_num/14_20_0.2_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/user_num/16_20_0.2_0.05/latency_reward.npy"))

for i in range(5):
    SAC_delay.append(np.mean(SAC[i][-50:]))

user_num_gap = [x - y for x, y in zip(TD3_delay, SAC_delay)]
user_num_gap = np.array(user_num_gap)
user_num_gap[user_num_gap < -50] = -50


TD3_delay = []
TD3 = []
TD3.append(np.load("../../TD3/task_num/10_10_0.2_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/task_num/10_15_0.2_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/task_num/10_20_0.2_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/task_num/10_25_0.2_0.05/latency_reward.npy"))
TD3.append(np.load("../../TD3/task_num/10_30_0.2_0.05/latency_reward.npy"))

for i in range(5):
    TD3_delay.append(np.mean(TD3[i][-50:]))

SAC_delay = []
SAC = []
SAC.append(np.load("../../SAC/task_num/10_10_0.2_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/task_num/10_15_0.2_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/task_num/10_20_0.2_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/task_num/10_25_0.2_0.05/latency_reward.npy"))
SAC.append(np.load("../../SAC/task_num/10_30_0.2_0.05/latency_reward.npy"))

for i in range(5):
    SAC_delay.append(np.mean(SAC[i][-50:]))

task_num_gap = [x - y for x, y in zip(TD3_delay, SAC_delay)]
task_num_gap = np.array(task_num_gap)
task_num_gap[task_num_gap < -50] = -50