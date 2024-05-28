import numpy as np
import matplotlib.pyplot as plt

delay_reward = np.load("../../SAC/dmax/10_40_0.2_0.05/latency_reward.npy")
length = len(delay_reward)
x = np.arange(1, length+1)

plt.plot(x, delay_reward, color='red')
plt.xlabel("Episode", fontsize=15)
plt.ylabel("Delay reward", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("train_delay_reward.pdf", format="pdf")
plt.show()