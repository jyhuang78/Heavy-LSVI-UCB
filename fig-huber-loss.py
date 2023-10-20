import numpy as np
import matplotlib.pyplot as plt

def huber_loss(x):
    """Huber loss with \tau = 1
    """
    res = np.zeros_like(x)
    for i, xi in enumerate(x):
        if abs(xi) <= 1:
            res[i] = xi**2 / 2
        else:
            res[i] = abs(xi) - 1/2
    return res

def pseudo_huber_loss(x):
    """pseudo Huber loss with \tau = 1
    """
    return np.sqrt(1 + np.power(x, 2)) - 1

x = np.linspace(-3, 3, 1000)
plt.figure()
plt.plot(x, np.power(x, 2) / 2, color='blue', label="Square Loss")
plt.plot(x, huber_loss(x), color='red', label="Huber Loss")
plt.plot(x, pseudo_huber_loss(x), color='green', label="pseudo-Huber loss")
# plt.ylabel("Density", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.savefig("./figures/huber-loss.pdf")
plt.show()
