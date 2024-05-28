import numpy as np
import matplotlib.pyplot as plt

SAC_dmax = []
SAC = []
SAC.append(np.load("../../SAC/dmax/10_20_0.1_0.05/cache_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_20_0.15_0.05/cache_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_20_0.2_0.05/cache_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_20_0.25_0.05/cache_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_20_0.3_0.05/cache_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_20_0.35_0.05/cache_reward.npy"))
SAC.append(np.load("../../SAC/dmax/10_20_0.4_0.05/cache_reward.npy"))

for i in range(7):
    SAC_dmax.append(np.mean(SAC[i][-50:]))

TD3_dmax = []
TD3 = []
TD3.append(np.load("../../TD3/dmax/10_20_0.1_0.05/hit_ratio.npy"))
TD3.append(np.load("../../TD3/dmax/10_20_0.15_0.05/hit_ratio.npy"))
TD3.append(np.load("../../TD3/dmax/10_20_0.2_0.05/hit_ratio.npy"))
TD3.append(np.load("../../TD3/dmax/10_20_0.25_0.05/hit_ratio.npy"))
TD3.append(np.load("../../TD3/dmax/10_20_0.3_0.05/hit_ratio.npy"))
TD3.append(np.load("../../TD3/dmax/10_20_0.35_0.05/hit_ratio.npy"))
TD3.append(np.load("../../TD3/dmax/10_20_0.4_0.05/hit_ratio.npy"))

# TD3.append(np.load("../../TD3/dmax/10_20_0.1_0.05/cache_reward.npy"))
# TD3.append(np.load("../../TD3/dmax/10_20_0.15_0.05/cache_reward.npy"))
# TD3.append(np.load("../../TD3/dmax/10_20_0.2_0.05/cache_reward.npy"))
# TD3.append(np.load("../../TD3/dmax/10_20_0.25_0.05/cache_reward.npy"))
# TD3.append(np.load("../../TD3/dmax/10_20_0.3_0.05/cache_reward.npy"))
# TD3.append(np.load("../../TD3/dmax/10_20_0.35_0.05/cache_reward.npy"))
# TD3.append(np.load("../../TD3/dmax/10_20_0.4_0.05/cache_reward.npy"))

for i in range(7):
    TD3_dmax.append(np.mean(TD3[i][-50:]))

PB_cache_dmax = []
PB_cache = []
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_20_0.1_0.05/cache_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_20_0.15_0.05/cache_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_20_0.2_0.05/cache_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_20_0.25_0.05/cache_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_20_0.3_0.05/cache_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_20_0.35_0.05/cache_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/dmax/10_20_0.4_0.05/cache_reward.npy"))

for i in range(7):
    PB_cache_dmax.append(np.mean(PB_cache[i][-50:]))

random_cache_dmax = []
random_cache = []
random_cache.append(np.load("../../SAC_random_cache/dmax/10_20_0.1_0.05/cache_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/dmax/10_20_0.15_0.05/cache_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/dmax/10_20_0.2_0.05/cache_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/dmax/10_20_0.25_0.05/cache_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/dmax/10_20_0.3_0.05/cache_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/dmax/10_20_0.35_0.05/cache_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/dmax/10_20_0.4_0.05/cache_reward.npy"))

for i in range(7):
    random_cache_dmax.append(np.mean(random_cache[i][-50:]))

x = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
l1, = plt.plot(x, SAC_dmax, marker='o', markerfacecolor='white')
l2, = plt.plot(x, TD3_dmax, marker='v', markerfacecolor='white')
l3, = plt.plot(x, PB_cache_dmax, marker='h', markerfacecolor='white')
l4, = plt.plot(x, random_cache_dmax, marker='x', markerfacecolor='white')

plt.xlabel("Tolerant delay")
plt.ylabel("Cache hit ratio")
plt.legend([l1, l2, l3, l4], ['DRA-SAC', 'TD3', 'PB cache', 'Random cache'], loc='best')
plt.grid()
plt.savefig("dmax_cache.pdf", format="pdf")
plt.show()
