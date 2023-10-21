import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = 3

    for scale in [0,2]:
        round = 10000

        # draw
        x = np.arange(0, round)
        plt.figure(figsize=(10, 6))


        with open("./data/menu_" + str(round) + "_noise_scale_" + str(scale) + ".txt", "r") as f:
            y_menu = list(map(float, f.readline().strip().split()))
        plt.plot(x, y_menu, label="MENU", linewidth='3')

        with open("./data/tofu_" + str(round) + "_noise_scale_" + str(scale) + ".txt", "r") as f:
            y_tofu = list(map(float, f.readline().strip().split()))
        plt.plot(x, y_tofu, label="TOFU", linewidth='3')

        with open("./data/heavy_oful_" + str(round) + "_noise_scale_" + str(scale) + ".txt", "r") as f:
            y_heavy_oful = list(map(float, f.readline().strip().split()))
        plt.plot(x, y_heavy_oful, label="Heavy-OFUL", linewidth='3')

        plt.xlabel("Iteration", fontsize=22)
        plt.ylabel("Regret", fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(linestyle=":")
        plt.savefig("./figures/noise_scale_" + str(scale) + ".pdf")
        plt.show()
