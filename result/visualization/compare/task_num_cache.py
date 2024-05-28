import numpy as np
import matplotlib.pyplot as plt

SAC_dmax = []
SAC = []
SAC.append(np.load("../../SAC/task_num/10_20_0.2_0.05/hit_ratio.npy"))
SAC.append(np.load("../../SAC/task_num/10_30_0.2_0.05/hit_ratio.npy"))
SAC.append(np.load("../../SAC/task_num/10_40_0.2_0.05/hit_ratio.npy"))
SAC.append(np.load("../../SAC/task_num/10_50_0.2_0.05/hit_ratio.npy"))
SAC.append(np.load("../../SAC/task_num/10_60_0.2_0.05/hit_ratio.npy"))

for i in range(5):
    SAC_dmax.append(np.mean(SAC[i][-50:]))

TD3_dmax = []
TD3 = []
TD3.append(np.load("../../TD3/task_num/10_20_0.2_0.05/hit_ratio.npy"))
TD3.append(np.load("../../TD3/task_num/10_30_0.2_0.05/hit_ratio.npy"))
TD3.append(np.load("../../TD3/task_num/10_40_0.2_0.05/hit_ratio.npy"))
TD3.append(np.load("../../TD3/task_num/10_50_0.2_0.05/hit_ratio.npy"))
TD3.append(np.load("../../TD3/task_num/10_60_0.2_0.05/hit_ratio.npy"))

for i in range(5):
    TD3_dmax.append(np.mean(TD3[i][-50:]))

PB_cache_dmax = []
PB_cache = []
PB_cache.append(np.load("../../SAC_PB_cache/task_num/10_20_0.2_0.05/hit_ratio.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/task_num/10_30_0.2_0.05/hit_ratio.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/task_num/10_40_0.2_0.05/hit_ratio.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/task_num/10_50_0.2_0.05/hit_ratio.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/task_num/10_60_0.2_0.05/hit_ratio.npy"))


for i in range(5):
    PB_cache_dmax.append(np.mean(PB_cache[i][-50:]))

random_cache_dmax = []
random_cache = []
random_cache.append(np.load("../../SAC_random_cache/task_num/10_20_0.2_0.05/hit_ratio.npy"))
random_cache.append(np.load("../../SAC_random_cache/task_num/10_30_0.2_0.05/hit_ratio.npy"))
random_cache.append(np.load("../../SAC_random_cache/task_num/10_40_0.2_0.05/hit_ratio.npy"))
random_cache.append(np.load("../../SAC_random_cache/task_num/10_50_0.2_0.05/hit_ratio.npy"))
random_cache.append(np.load("../../SAC_random_cache/task_num/10_60_0.2_0.05/hit_ratio.npy"))

for i in range(5):
    random_cache_dmax.append(np.mean(random_cache[i][-50:]))

x = np.array([20, 30, 40, 50, 60])
l1, = plt.plot(x, SAC_dmax, marker='o', markerfacecolor='white')
l2, = plt.plot(x, TD3_dmax, marker='v', markerfacecolor='white')
l3, = plt.plot(x, PB_cache_dmax, marker='D', markerfacecolor='white')
l4, = plt.plot(x, random_cache_dmax, marker='x', markerfacecolor='white')

plt.xlabel("Number of tasks", fontsize=15)
plt.ylabel("Cache hit ratio", fontsize=15)
plt.xticks(np.arange(20, 61, 10), fontsize=15)
plt.yticks(fontsize=15)
plt.legend([l1, l2, l3, l4], ['DRA-SAC', 'TD3', 'PB cache', 'Random cache'], loc='best')
plt.grid(color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("task_num_cache.pdf", format="pdf")
plt.show()
