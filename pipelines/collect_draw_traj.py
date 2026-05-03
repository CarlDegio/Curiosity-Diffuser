import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D


TRAJ_START = 0
TRAJ_END = None
STRIDE = 5


METHODS = [
    {
        "key": "curiosity_diffuser",
        "title": "Curiosity-Diffuser",
        "pipeline": "diffuser_d4rl_antmaze_rnd",
        "subdir": "curiosity_diffuser_ckpt_{ckpt}",
        "filename": "curiosity_diffuser_seed_{data_id}.npz",
    },
    {
        "key": "rediffuser",
        "title": "ReDiffuser",
        "pipeline": "rediffuser_minari_antmaze",
        "subdir": "rediffuser_ckpt_{ckpt}",
        "filename": "rediffuser_seed_{data_id}.npz",
    },
    {
        "key": "diffuser",
        "title": "Diffuser",
        "pipeline": "diffuser_d4rl_antmaze",
        "subdir": "diffuser_ckpt_{ckpt}",
        "filename": "diffuser_seed_{data_id}.npz",
    },
]


MEDIUM_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, "r", 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, "g", 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]


LARGE_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, "r", 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, "g", 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


def env_fragment(maze):
    if maze == "medium":
        return "D4RL/antmaze/medium-play-v1"
    if maze == "large":
        return "D4RL/antmaze/large-play-v1"
    raise ValueError(f"Unsupported maze: {maze}")


def load_method_data(method, maze, data_id, ckpt):
    env_name = env_fragment(maze)
    path = (
        Path("results") / method["pipeline"] / env_name /
        "collected_test_data" / method["subdir"].format(ckpt=ckpt) /
        method["filename"].format(data_id=data_id)
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")
    data = np.load(path)
    obs = data["obs"]
    novelty = data["curiosity_rnd_novelty"]
    return {
        "path": path,
        "obs": obs,
        "xy": obs[:, :2],
        "goal": obs[0, -2:],
        "start": obs[0, :2],
        "novelty": novelty,
        "return": float(data["episode_return"][0]),
    }


def slice_traj(xy, novelty):
    xy_sliced = xy[TRAJ_START:TRAJ_END:STRIDE]
    novelty_sliced = novelty[TRAJ_START:TRAJ_END:STRIDE]
    step_sliced = np.arange(len(xy))[TRAJ_START:TRAJ_END:STRIDE]
    return xy_sliced, novelty_sliced, step_sliced


def draw_map(ax, maze):
    maze_layout = MEDIUM_MAZE if maze == "medium" else LARGE_MAZE
    maze_scale = 4.0
    block_size = 3.7
    height = len(maze_layout)
    width = len(maze_layout[0])
    x_center = width / 2 * maze_scale
    y_center = height / 2 * maze_scale

    ax.set_axis_off()
    ax.set_aspect("equal")

    for i in range(height):
        for j in range(width):
            cell = maze_layout[i][j]
            x = (j + 0.5) * maze_scale - x_center
            y = y_center - (i + 0.5) * maze_scale

            if cell == 1:
                rect = plt.Rectangle(
                    (x - maze_scale / 2, y - maze_scale / 2),
                    maze_scale, maze_scale, color="black", fill=True)
            else:
                rect = plt.Rectangle(
                    (x - block_size / 2, y - block_size / 2),
                    block_size, block_size, color="lightgray", fill=True)
            ax.add_patch(rect)


def mark_start_goal(ax, start, goal):
    ax.plot(
        start[0], start[1], marker="^", color="#2ca02c", markeredgecolor="black",
        markeredgewidth=0.5, markersize=7, linestyle="None", zorder=5)
    ax.plot(
        goal[0], goal[1], marker="*", color="#ff7f0e", markeredgecolor="black",
        markeredgewidth=0.5, markersize=10, linestyle="None", zorder=5)


def plot_order_traj(ax, xy, steps, start, goal, norm):
    if len(xy) == 0:
        return None
    ax.plot(xy[:, 0], xy[:, 1], color="#7f0000", linewidth=0.8, alpha=0.35, zorder=2)
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=steps, cmap=plt.cm.Reds, norm=norm, s=18, zorder=3)
    mark_start_goal(ax, start, goal)
    return scatter


def plot_novelty_traj(ax, xy, novelty, start, goal, norm):
    if len(xy) == 0:
        return None
    ax.plot(xy[:, 0], xy[:, 1], color="#333333", linewidth=0.7, alpha=0.25, zorder=2)
    scatter = ax.scatter(
        xy[:, 0], xy[:, 1], c=novelty, cmap="viridis", norm=norm, s=18, zorder=3)
    mark_start_goal(ax, start, goal)
    return scatter


def draw_figure(args):
    plt.rc("font", family="Times New Roman")
    method_data = [load_method_data(method, args.maze, args.data_id, args.ckpt) for method in METHODS]

    sliced = []
    all_novelty = []
    for data in method_data:
        xy, novelty, steps = slice_traj(data["xy"], data["novelty"])
        sliced.append((xy, novelty, steps))
        if len(novelty) > 0:
            all_novelty.append(novelty)

    if not all_novelty:
        raise ValueError("No novelty values found after trajectory slicing.")
    novelty_values = np.concatenate(all_novelty)
    novelty_norm = colors.Normalize(vmin=float(np.min(novelty_values)), vmax=np.percentile(novelty_values, 80))
    step_norm = colors.Normalize(vmin=0, vmax=1000)

    fig, axs = plt.subplots(2, 3, figsize=(13.5, 6.7), constrained_layout=False)
    fig.subplots_adjust(left=0.04, right=0.90, bottom=0.11, top=0.90, wspace=0.08, hspace=-0.0)
    for row in range(2):
        for col in range(3):
            draw_map(axs[row, col], args.maze)

    order_scatter = None
    novelty_scatter = None
    for col, (method, data, (xy, novelty, steps)) in enumerate(zip(METHODS, method_data, sliced)):
        axs[0, col].set_title(method["title"], fontsize=17)
        order_scatter = plot_order_traj(axs[0, col], xy, steps, data["start"], data["goal"], step_norm)
        novelty_scatter = plot_novelty_traj(
            axs[1, col], xy, novelty, data["start"], data["goal"], novelty_norm)

    axs[0, 0].set_ylabel("Time order", fontsize=15)
    axs[1, 0].set_ylabel("RND novelty", fontsize=15)
    if order_scatter is not None:
        cbar = fig.colorbar(order_scatter, ax=axs[0, :], shrink=0.8, pad=0.02)
        cbar.set_label("Timestep", fontsize=14)
        cbar.set_ticks([0, 250, 500, 750, 1000])
    if novelty_scatter is not None:
        cbar = fig.colorbar(novelty_scatter, ax=axs[1, :], shrink=0.8, pad=0.02)
        cbar.set_label("Curiosity Value", fontsize=14)

    legend_handles = [
        Line2D(
            [0], [0], marker="^", color="none", markerfacecolor="#2ca02c",
            markeredgecolor="black", markeredgewidth=0.5, markersize=9, label="Start"),
        Line2D(
            [0], [0], marker="*", color="none", markerfacecolor="#ff7f0e",
            markeredgecolor="black", markeredgewidth=0.5, markersize=13, label="Goal"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=2, frameon=False,
        fontsize=14, bbox_to_anchor=(0.4, 0.0))

    fig.suptitle(f"AntMaze {args.maze.title()} Play", fontsize=14)
    output = Path(args.output) if args.output else Path("results") / "trajectory_figures" / f"antmaze_{args.maze}_play_seed_{args.data_id}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to {output}")
    if args.show:
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Draw collected AntMaze trajectories with order and RND novelty colors.")
    parser.add_argument("--data-id", type=int, default=0, help="Seed/data id shared by all three methods.")
    parser.add_argument("--maze", choices=["medium", "large"], default="medium")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    draw_figure(parse_args())
