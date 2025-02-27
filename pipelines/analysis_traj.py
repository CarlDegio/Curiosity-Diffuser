import hydra
from utils import set_seed
import matplotlib.pyplot as plt
import numpy as np  # Make sure to import numpy


def draw_ratio():
    # Success rates for different RND coefficients
    rnd_coefficients = [0.1, 1, 5, 10, 20, 50]  # Adjusted to avoid log(0)
    success_rates = [10/50, 12/50, 14/50, 18/50, 13/50, 15/50]  # Average success rates from the log

    plt.plot(rnd_coefficients, success_rates, marker='o', label='Success Rate')
    plt.axhline(y=success_rates[0], color='r', linestyle='--', label='λ=0 Success Rate')
    plt.xlabel('RND Coefficient', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.ylim(0, 0.5)
    # plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.title('Success Rate of AntMaze Medium Play', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.show()

@hydra.main(config_path="../configs/diffuser/antmaze", config_name="antmaze_rnd", version_base=None)
def pipeline(args):
    set_seed(args.seed)
    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/rnd_10_90000/'
    obs_path = save_path + f"episode_{0}_obs.npy"
    act_path = save_path + f"episode_{0}_act.npy"
    obs = np.load(obs_path)
    act = np.load(act_path)
    path = obs[0,:,:2]
    plt.plot(path[:, 0], path[:, 1], marker='o', label='Path')  # Plotting the path
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Path Visualization')
    plt.grid(True)
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    # draw_ratio()
    pipeline()