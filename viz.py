import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def visualize_dropout_rates():
    folder_path = os.path.expanduser("~/Downloads/checkpoints_dropout/")

    dr_02 = torch.load(os.path.join(folder_path, "checkpoint_dr_0.2.tar"), map_location=torch.device('cpu'))["loss"]
    dr_03 = torch.load(os.path.join(folder_path, "checkpoint_dr_0.3.tar"), map_location=torch.device('cpu'))["loss"]
    dr_04 = torch.load(os.path.join(folder_path, "checkpoint_dr_0.4.tar"), map_location=torch.device('cpu'))["loss"]
    dr_05 = torch.load(os.path.join(folder_path, "checkpoint_dr_0.5.tar"), map_location=torch.device('cpu'))["loss"]
    drs = [dr_02, dr_03, dr_04, dr_05]

    dr_02_n = torch.load(os.path.join(folder_path, "checkpoint_dr_0.2_noisy.tar"), map_location=torch.device('cpu'))["loss"]
    dr_03_n = torch.load(os.path.join(folder_path, "checkpoint_dr_0.3_noisy.tar"), map_location=torch.device('cpu'))["loss"]
    dr_04_n = torch.load(os.path.join(folder_path, "checkpoint_dr_0.4_noisy.tar"), map_location=torch.device('cpu'))["loss"]
    dr_05_n = torch.load(os.path.join(folder_path, "checkpoint_dr_0.5_noisy.tar"), map_location=torch.device('cpu'))["loss"]
    drs_n = [dr_02_n, dr_03_n, dr_04_n, dr_05_n]

    x = np.arange(4)

    bar_width = 0.35

    bars1 = plt.bar(x - bar_width/2, drs, bar_width, label='dropout')
    bars2 = plt.bar(x + bar_width/2, drs_n, bar_width, label='dropout and noise')

    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
            f"{bar.get_height():0.2f}", ha='center', va='bottom')

    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
            f"{bar.get_height():0.2f}", ha='center', va='bottom')

    plt.xlabel('dropout rate')
    plt.ylabel('Loss after 50 epochs')
    plt.title('Loss for different dropout rates with and without noise')
    plt.xticks(x, ['0.2', '0.3', '0.4', '0.5'])
    plt.legend()

    plt.show()


if __name__ == "__main__":
    visualize_dropout_rates()