import numpy as np
import matplotlib.pyplot as plt

worst_cost_dmax = []
worst_cost = np.mean(np.load("../../SAC/dmax/10_40_0.2_0.05/worst.npy"))

for i in range(7):
    worst_cost_dmax.append(worst_cost)

worst_cost_dmax = np.array(worst_cost_dmax)

SAC_dmax = []
SAC = []
SAC.append(np.load("../../SAC/dmax/10_40_0.1_0.05/cost_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_40_0.15_0.05/cost_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_40_0.2_0.05/cost_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_40_0.25_0.05/cost_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_40_0.3_0.05/cost_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_40_0.35_0.05/cost_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_40_0.4_0.05/cost_reward.npy"))

for i in range(7):
    SAC_dmax.append(np.mean(SAC[i][-50:]))

SAC_dmax = np.array(SAC_dmax)
SAC_dmax = worst_cost_dmax - SAC_dmax

TD3_dmax = []
TD3 = []
TD3.append(np.load("../../TD3/dmax/10_40_0.1_0.05/cost_reward.npy"))
TD3.append(np.load("../../TD3/dmax/10_40_0.15_0.05/cost_reward.npy"))
TD3.append(np.load("../../TD3/dmax/10_40_0.2_0.05/cost_reward.npy"))
TD3.append(np.load("../../TD3/dmax/10_40_0.25_0.05/cost_reward.npy"))
TD3.append(np.load("../../TD3/dmax/10_40_0.3_0.05/cost_reward.npy"))
TD3.append(np.load("../../TD3/dmax/10_40_0.35_0.05/cost_reward.npy"))
TD3.append(np.load("../../TD3/dmax/10_40_0.4_0.05/cost_reward.npy"))

for i in range(7):
    TD3_dmax.append(np.mean(TD3[i][-50:]))

TD3_dmax = np.array(TD3_dmax)
TD3_dmax = worst_cost_dmax - TD3_dmax

PB_cache_dmax = []
PB_cache = []
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_40_0.1_0.05/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_40_0.15_0.05/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_40_0.2_0.05/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_40_0.25_0.05/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_40_0.3_0.05/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_40_0.35_0.05/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_40_0.4_0.05/cost_reward.npy"))

for i in range(7):
    PB_cache_dmax.append(np.mean(PB_cache[i][-50:]))

PB_cache_dmax = np.array(PB_cache_dmax)
PB_cache_dmax = worst_cost_dmax - PB_cache_dmax

random_cache_dmax = []
random_cache = []
random_cache.append(np.load("../../SAC_random_cache/dmax/10_40_0.1_0.05/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/dmax/10_40_0.15_0.05/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/dmax/10_40_0.2_0.05/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/dmax/10_40_0.25_0.05/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/dmax/10_40_0.3_0.05/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/dmax/10_40_0.35_0.05/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/dmax/10_40_0.4_0.05/cost_reward.npy"))

for i in range(7):
    random_cache_dmax.append(np.mean(random_cache[i][-50:]))

random_cache_dmax = np.array(random_cache_dmax)
random_cache_dmax = worst_cost_dmax - random_cache_dmax

without_cache_dmax = []
without_cache = []
without_cache.append(np.load("../../SAC_without_cache/dmax/10_40_0.1_0.05/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/dmax/10_40_0.15_0.05/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/dmax/10_40_0.2_0.05/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/dmax/10_40_0.25_0.05/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/dmax/10_40_0.3_0.05/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/dmax/10_40_0.35_0.05/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/dmax/10_40_0.4_0.05/cost_reward.npy"))

for i in range(7):
    without_cache_dmax.append(np.mean(without_cache[i][-50:]))

without_cache_dmax = np.array(without_cache_dmax)
without_cache_dmax = worst_cost_dmax - without_cache_dmax

x = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
l1, = plt.plot(x, SAC_dmax, marker='o', markerfacecolor='white')
l2, = plt.plot(x, TD3_dmax, marker='v', markerfacecolor='white')
l3, = plt.plot(x, PB_cache_dmax, marker='D', markerfacecolor='white')
l4, = plt.plot(x, random_cache_dmax, marker='x', markerfacecolor='white')
l5, = plt.plot(x, without_cache_dmax, marker='s', markerfacecolor='white')
l6, = plt.plot(x, worst_cost_dmax, marker='*', markerfacecolor='white')

plt.xlabel("Tolerant delay (s)", fontsize=15)
plt.ylabel("Average system cost per MU", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([l1, l2, l3, l4, l5, l6], ['DRA-SAC', 'TD3', 'PB cache', 'Random cache', 'Without cache', 'Worst cost'], loc='best')
plt.grid(color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("dmax.pdf", format="pdf")
plt.show()
