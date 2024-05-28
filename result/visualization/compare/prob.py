import numpy as np
import matplotlib.pyplot as plt
# from TD3_delay_penalty import prob_gap

worst_cost_prob = []
worst_cost = np.mean(np.load("../../SAC/dmax/10_40_0.2_0.05/worst.npy"))

for i in range(7):
    worst_cost_prob.append(worst_cost)

worst_cost_prob = np.array(worst_cost_prob)

SAC_prob = []
SAC = []
SAC.append(np.load("../../SAC/prob/10_40_0.2_0.0/cost_reward.npy"))
SAC.append(np.load("../../SAC/prob/10_40_0.2_0.05/cost_reward.npy"))
SAC.append(np.load("../../SAC/prob/10_40_0.2_0.1/cost_reward.npy"))
SAC.append(np.load("../../SAC/prob/10_40_0.2_0.15/cost_reward.npy"))
SAC.append(np.load("../../SAC/prob/10_40_0.2_0.2/cost_reward.npy"))
SAC.append(np.load("../../SAC/prob/10_40_0.2_0.25/cost_reward.npy"))
SAC.append(np.load("../../SAC/prob/10_40_0.2_0.3/cost_reward.npy"))

for i in range(7):
    SAC_prob.append(np.mean(SAC[i][-50:]))

SAC_prob = np.array(SAC_prob)
SAC_prob = worst_cost_prob - SAC_prob

TD3_prob = []
TD3 = []
TD3.append(np.load("../../TD3/prob/10_40_0.2_0.0/cost_reward.npy"))
TD3.append(np.load("../../TD3/prob/10_40_0.2_0.05/cost_reward.npy"))
TD3.append(np.load("../../TD3/prob/10_40_0.2_0.1/cost_reward.npy"))
TD3.append(np.load("../../TD3/prob/10_40_0.2_0.15/cost_reward.npy"))
TD3.append(np.load("../../TD3/prob/10_40_0.2_0.2/cost_reward.npy"))
TD3.append(np.load("../../TD3/prob/10_40_0.2_0.25/cost_reward.npy"))
TD3.append(np.load("../../TD3/prob/10_40_0.2_0.3/cost_reward.npy"))

for i in range(7):
    TD3_prob.append(np.mean(TD3[i][-50:]))

TD3_prob = np.array(TD3_prob)
# prob_gap = np.array(prob_gap)
# TD3_prob = TD3_prob + prob_gap
TD3_prob = worst_cost_prob - TD3_prob

ratio1 = (TD3_prob[1] - SAC_prob[1]) / TD3_prob[1]
print(ratio1)

PB_cache_prob = []
PB_cache = []
PB_cache.append(np.load("../../SAC_PB_cache/prob/10_40_0.2_0.0/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/prob/10_40_0.2_0.05/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/prob/10_40_0.2_0.1/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/prob/10_40_0.2_0.15/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/prob/10_40_0.2_0.2/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/prob/10_40_0.2_0.25/cost_reward.npy"))
PB_cache.append(np.load("../../SAC_PB_cache/prob/10_40_0.2_0.3/cost_reward.npy"))

for i in range(7):
    PB_cache_prob.append(np.mean(PB_cache[i][-50:]))

PB_cache_prob = np.array(PB_cache_prob)
PB_cache_prob = worst_cost_prob - PB_cache_prob

ratio2 = (PB_cache_prob[1] - SAC_prob[1]) / PB_cache_prob[1]
print(ratio2)

random_cache_prob = []
random_cache = []
random_cache.append(np.load("../../SAC_random_cache/prob/10_40_0.2_0.0/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/prob/10_40_0.2_0.05/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/prob/10_40_0.2_0.1/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/prob/10_40_0.2_0.15/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/prob/10_40_0.2_0.2/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/prob/10_40_0.2_0.25/cost_reward.npy"))
random_cache.append(np.load("../../SAC_random_cache/prob/10_40_0.2_0.3/cost_reward.npy"))

for i in range(7):
    random_cache_prob.append(np.mean(random_cache[i][-50:]))

random_cache_prob = np.array(random_cache_prob)
random_cache_prob = worst_cost_prob - random_cache_prob

without_cache_prob = []
without_cache = []
without_cache.append(np.load("../../SAC_without_cache/prob/10_40_0.2_0.0/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/prob/10_40_0.2_0.05/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/prob/10_40_0.2_0.1/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/prob/10_40_0.2_0.15/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/prob/10_40_0.2_0.2/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/prob/10_40_0.2_0.25/cost_reward.npy"))
without_cache.append(np.load("../../SAC_without_cache/prob/10_40_0.2_0.3/cost_reward.npy"))

for i in range(7):
    without_cache_prob.append(np.mean(without_cache[i][-50:]))

without_cache_prob = np.array(without_cache_prob)
without_cache_prob = worst_cost_prob - without_cache_prob

x = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
l1, = plt.plot(x, SAC_prob, marker='o', markerfacecolor='white')
l2, = plt.plot(x, TD3_prob, marker='v', markerfacecolor='white')
l3, = plt.plot(x, PB_cache_prob, marker='D', markerfacecolor='white')
l4, = plt.plot(x, random_cache_prob, marker='x', markerfacecolor='white')
l5, = plt.plot(x, without_cache_prob, marker='s', markerfacecolor='white')
l6, = plt.plot(x, worst_cost_prob, marker='*', markerfacecolor='white')

plt.xlabel("Probability bound", fontsize=15)
plt.ylabel("Average system cost per MU", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([l1, l2, l3, l4, l5, l6], ['DRA-SAC', 'TD3', 'PB cache', 'Random cache', 'Without cache', 'Worst cost'], loc='best')
plt.grid(color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("prob.pdf", format="pdf")
plt.show()
