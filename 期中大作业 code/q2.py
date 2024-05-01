import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False # 显示负号

a = np.array([0.015, 0.03, 0.05, 0.08]).reshape(-1, 1)  # 将a变为列向量
b = np.array([0, 0.1, 0.2, 0.3]).reshape(-1, 1)  # 将b变为列向量
k = np.arange(0, np.pi, 0.01)
Re_k = a * np.sin(3 * k) + (-4 * a - 1 / 6) * np.sin(2 * k) + (5 * a + 4 / 3) * np.sin(k)
Im_k = b * np.cos(3 * k) - 6 * b * np.cos(2 * k) + 15 * b * np.cos(k) - 10 * b * np.ones_like(k)

plt.figure(1)
for i in range(len(a)):
    plt.plot(k, Re_k[i], linewidth=1)
plt.gca().tick_params(width=1)
plt.legend(['α=0.015', 'α=0.03', 'α=0.05', 'α=0.08'], loc='upper left')
plt.xlim([0, np.pi])
plt.xlabel('k')
plt.ylabel("Re(k'')")
plt.title('Dispersion curve', fontsize=12)  # 色散曲线

plt.figure(2)
for i in range(len(b)):
    plt.plot(k, Im_k[i], linewidth=1)
plt.gca().tick_params(width=1)
plt.legend(['β=0', 'β=0.1', 'β=0.2', 'β=0.3'], loc='lower left')
plt.xlim([0, np.pi])
plt.ylim([-10, 1])
plt.xlabel('k')
plt.ylabel("Im(k'')")
plt.title('Dissipation curve', fontsize=12)  # 耗散曲线

plt.show()
