import numpy as np
import matplotlib.pyplot as plt

total_reward = np.load("../../SAC/dmax/10_40_0.2_0.05/total_reward.npy")
length = len(total_reward)
x = np.arange(1, length+1)

plt.plot(x, total_reward, color='red')
plt.xlabel("Episode", fontsize=15)
plt.ylabel("Total reward", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("train_total_reward.pdf", format="pdf")
plt.show()