import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes

cost_rewards = []
cost_rewards.append(np.load("../../SAC/prob/10_40_0.2_0.3/cost_reward.npy"))
cost_rewards.append(np.load("../../SAC/prob/10_40_0.2_0.25/cost_reward.npy"))
cost_rewards.append(np.load("../../SAC/prob/10_40_0.2_0.2/cost_reward.npy"))
cost_rewards.append(np.load("../../SAC/prob/10_40_0.2_0.15/cost_reward.npy"))
cost_rewards.append(np.load("../../SAC/prob/10_40_0.2_0.1/cost_reward.npy"))
cost_rewards.append(np.load("../../SAC/prob/10_40_0.2_0.05/cost_reward.npy"))
cost_rewards.append(np.load("../../SAC/prob/10_40_0.2_0.0/cost_reward.npy"))

length = len(cost_rewards[0])
x = np.arange(1, length+1)
legends = ['$\\beta = 0.3$', '$\\beta = 0.25$', '$\\beta = 0.2$', '$\\beta = 0.15$', '$\\beta = 0.1$', '$\\beta = 0.05$', '$\\beta = 0.0$']

fig, ax = plt.subplots()
for i in range(7):
    ax.plot(x, cost_rewards[i], label=legends[i])

ax.set_xlabel("Episode", fontsize=15)
ax.set_ylabel("Cost reward", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax.grid(color='gray', alpha=0.3)

# 创建局部放大图的子图
axins = inset_axes(ax, width="33%", height="28%", loc='lower left', bbox_to_anchor=(0.3, 0.27, 1, 1), bbox_transform=ax.transAxes)

for i in range(7):
    axins.plot(x, cost_rewards[i], label=legends[i])

axins.set_xlim(650, 700)
axins.set_ylim(613, 748)
axins.tick_params(axis='both', labelsize=13)    # 设置子图的字体大小

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

ax.legend(loc="best")
plt.tight_layout()
plt.savefig("merge_prob_cost_reward.pdf", format="pdf")
plt.show()