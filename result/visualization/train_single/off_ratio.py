import numpy as np
import matplotlib.pyplot as plt

off_ratio = np.load("../../SAC/dmax/10_40_0.2_0.05/average_off_ratio.npy")
length = len(off_ratio)
x = np.arange(1, length+1)

plt.plot(x, off_ratio, color='blue')
plt.xlabel("Episode", fontsize=15)
plt.ylabel("Offloading ratio", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("train_off_ratio.pdf", format="pdf")
plt.show()