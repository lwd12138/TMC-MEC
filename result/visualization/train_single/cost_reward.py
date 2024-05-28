import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes

cost_reward = np.load("../../SAC/dmax/10_40_0.2_0.05/cost_reward.npy")
length = len(cost_reward)
x = np.arange(1, length+1)

fig, ax = plt.subplots()
ax.plot(x, cost_reward, color='red')
ax.set_xlabel("Episode", fontsize=15)
ax.set_ylabel("Cost reward", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax.grid(color='gray', alpha=0.3)

# 创建局部放大图的子图
axins = inset_axes(ax, width="15%", height="35%", loc='lower left', bbox_to_anchor=(0.3, 0.3, 1, 1), bbox_transform=ax.transAxes)

axins.plot(x, cost_reward, color='red')

axins.set_xlim(0, 20)
axins.set_ylim(330, 520)
axins.tick_params(axis='both', labelsize=13)    # 设置子图的字体大小

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# ax.legend(loc="best")
# plt.tight_layout()
plt.tight_layout()
plt.savefig("train_cost_reward.pdf", format="pdf")
plt.show()