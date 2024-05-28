import numpy as np
import matplotlib.pyplot as plt

worst_cost_user_num = []
worst_cost = np.mean(np.load("../../SAC/dmax/10_40_0.2_0.05/worst.npy"))

for i in range(5):
    worst_cost_user_num.append(worst_cost)

worst_cost_user_num = np.array(worst_cost_user_num)

SAC_user_num = []
SAC = []
SAC.append(np.load("../../SAC/user_num/8_40_0.2_0.05/cost_reward.npy"))
SAC.append(np.load("../../SAC/user_num/10_40_0.2_0.05/cost_reward.npy"))
SAC.append(np.load("../../SAC/user_num/12_40_0.2_0.05/cost_reward.npy"))
SAC.append(np.load("../../SAC/user_num/14_40_0.2_0.05/cost_reward.npy"))
SAC.append(np.load("../../SAC/user_num/16_40_0.2_0.05/cost_reward.npy"))

for i in range(5):
    SAC_user_num.append(np.mean(SAC[i][-50:]))

SAC_user_num = np.array(SAC_user_num)
SAC_user_num = worst_cost_user_num - SAC_user_num

TD3_user_num = []
TD3 = []
TD3.append(np.load("../../TD3/user_num/8_40_0.2_0.05/cost_reward.npy"))
TD3.append(np.load("../../TD3/user_num/10_40_0.2_0.05/cost_reward.npy"))
TD3.append(np.load("../../TD3/user_num/12_40_0.2_0.05/cost_reward.npy"))
TD3.append(np.load("../../TD3/user_num/14_40_0.2_0.05/cost_reward.npy"))
TD3.append(np.load("../../TD3/user_num/16_40_0.2_0.05/cost_reward.npy"))

for i in range(5):
    TD3_user_num.append(np.mean(TD3[i][-50:]))

TD3_user_num = np.array(TD3_user_num)
TD3_user_num = worst_cost_user_num - TD3_user_num

PB_cache_user_num = []
PB_cache = []
PB_cache.append(np.load("../../SAC_PB_cache/user_num/8_40_0.2_0.05/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/user_num/10_40_0.2_0.05/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/user_num/12_40_0.2_0.05/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/user_num/14_40_0.2_0.05/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/user_num/16_40_0.2_0.05/cost_reward.npy"))

for i in range(5):
    PB_cache_user_num.append(np.mean(PB_cache[i][-50:]))

PB_cache_user_num = np.array(PB_cache_user_num)
PB_cache_user_num = worst_cost_user_num - PB_cache_user_num

random_cache_user_num = []
random_cache = []
random_cache.append(np.load("../../SAC_random_cache/user_num/8_40_0.2_0.05/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/user_num/10_40_0.2_0.05/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/user_num/12_40_0.2_0.05/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/user_num/14_40_0.2_0.05/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/user_num/16_40_0.2_0.05/cost_reward.npy"))

for i in range(5):
    random_cache_user_num.append(np.mean(random_cache[i][-50:]))

random_cache_user_num = np.array(random_cache_user_num)
random_cache_user_num = worst_cost_user_num - random_cache_user_num

without_cache_user_num = []
without_cache = []
without_cache.append(np.load("../../SAC_without_cache/user_num/8_40_0.2_0.05/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/user_num/10_40_0.2_0.05/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/user_num/12_40_0.2_0.05/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/user_num/14_40_0.2_0.05/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/user_num/16_40_0.2_0.05/cost_reward.npy"))

for i in range(5):
    without_cache_user_num.append(np.mean(without_cache[i][-50:]))

without_cache_user_num = np.array(without_cache_user_num)
without_cache_user_num = worst_cost_user_num - without_cache_user_num

x = np.array([8, 10, 12, 14, 16])
l1, = plt.plot(x, SAC_user_num, marker='o', markerfacecolor='white')
l2, = plt.plot(x, TD3_user_num, marker='v', markerfacecolor='white')
l3, = plt.plot(x, PB_cache_user_num, marker='D', markerfacecolor='white')
l4, = plt.plot(x, random_cache_user_num, marker='x', markerfacecolor='white')
l5, = plt.plot(x, without_cache_user_num, marker='s', markerfacecolor='white')
l6, = plt.plot(x, worst_cost_user_num, marker='*', markerfacecolor='white')

plt.xlabel("Number of MUs", fontsize=15)
plt.ylabel("Average system cost per MU", fontsize=15)
plt.xticks(np.arange(8, 17, 2), fontsize=15)
plt.yticks(fontsize=15)
plt.legend([l1, l2, l3, l4, l5, l6], ['DRA-SAC', 'TD3', 'PB cache', 'Random cache', 'Without cache', 'Worst cost'], loc='best', bbox_to_anchor=(0.4, 0.5))
plt.grid(color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("user_num.pdf", format="pdf")
plt.show()
