import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes

cost_rewards = []
cost_rewards.append(np.load("../../SAC/dmax/10_40_0.4_0.05/average_off_ratio.npy"))
cost_rewards.append(np.load("../../SAC/dmax/10_40_0.35_0.05/average_off_ratio.npy"))
cost_rewards.append(np.load("../../SAC/dmax/10_40_0.3_0.05/average_off_ratio.npy"))
cost_rewards.append(np.load("../../SAC/dmax/10_40_0.25_0.05/average_off_ratio.npy"))
cost_rewards.append(np.load("../../SAC/dmax/10_40_0.2_0.05/average_off_ratio.npy"))
cost_rewards.append(np.load("../../SAC/dmax/10_40_0.15_0.05/average_off_ratio.npy"))
cost_rewards.append(np.load("../../SAC/dmax/10_40_0.1_0.05/average_off_ratio.npy"))

length = len(cost_rewards[0])
x = np.arange(1, length+1)
legends = ['$T^{max} = 0.4$', '$T^{max} = 0.35$', '$T^{max} = 0.3$', '$T^{max} = 0.25$', '$T^{max} = 0.2$', '$T^{max} = 0.15$', '$T^{max} = 0.1$']

fig, ax = plt.subplots()
for i in range(7):
    ax.plot(x, cost_rewards[i], label=legends[i])

ax.set_xlabel("Episode", fontsize=15)
ax.set_ylabel("Offloading ratio", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax.grid(color='gray', alpha=0.3)

# # 创建局部放大图的子图
# axins = inset_axes(ax, width="33%", height="28%", loc='lower left', bbox_to_anchor=(0.3, 0.15, 1, 1), bbox_transform=ax.transAxes)

# for i in range(7):
#     axins.plot(x, cost_rewards[i], label=legends[i])

# axins.set_xlim(650, 700)
# axins.set_ylim(0.4, 0.7)
# axins.tick_params(axis='both', labelsize=11)    # 设置子图的字体大小

# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

ax.legend(loc="best")
plt.tight_layout()
plt.savefig("merge_dmax_off_ratio.pdf", format="pdf")
plt.show()