import argparse
import json
import time
from pathlib import Path

import minari
import numpy as np
from sklearn.cluster import KMeans


METHODS = {
    "curiosity-diffuser": {
        "pipeline": "diffuser_d4rl_antmaze_rnd",
        "subdir": "curiosity_diffuser_ckpt_{ckpt}",
        "pattern": "curiosity_diffuser_seed_*.npz",
    },
    "curiosity_diffuser": {
        "pipeline": "diffuser_d4rl_antmaze_rnd",
        "subdir": "curiosity_diffuser_ckpt_{ckpt}",
        "pattern": "curiosity_diffuser_seed_*.npz",
    },
    "diffuser": {
        "pipeline": "diffuser_d4rl_antmaze",
        "subdir": "diffuser_ckpt_{ckpt}",
        "pattern": "diffuser_seed_*.npz",
    },
    "rediffuser": {
        "pipeline": "rediffuser_minari_antmaze",
        "subdir": "rediffuser_ckpt_{ckpt}",
        "pattern": "rediffuser_seed_*.npz",
    },
}


def env_name(group):
    if group == "medium":
        return "D4RL/antmaze/medium-play-v1"
    if group == "large":
        return "D4RL/antmaze/large-play-v1"
    raise ValueError(f"Unsupported group: {group}")


def method_dir(method, group, ckpt):
    spec = METHODS[method]
    return (
        Path("results") / spec["pipeline"] / env_name(group) /
        "collected_test_data" / spec["subdir"].format(ckpt=ckpt)
    )


def seed_from_path(path):
    return int(path.stem.rsplit("_", 1)[-1])


def load_collected_trajectories(method, group, ckpt, n):
    spec = METHODS[method]
    root = method_dir(method, group, ckpt)
    paths = sorted(root.glob(spec["pattern"]), key=seed_from_path)
    if n is not None:
        paths = paths[:n]
    if not paths:
        raise FileNotFoundError(f"No collected trajectories found under {root}")

    trajectories = []
    for path in paths:
        data = np.load(path)
        trajectories.append({
            "path": path,
            "seed": int(data["seed"]),
            "obs": data["obs"].astype(np.float64),
            "act": data["act"].astype(np.float64),
        })
    return trajectories


def load_minari_reference(group, download):
    dataset = minari.load_dataset(env_name(group), download=download)
    obs_list = []
    act_list = []
    for episode in dataset:
        obs = np.concatenate([
            episode.observations["achieved_goal"],
            episode.observations["observation"],
            episode.observations["desired_goal"],
        ], axis=-1)
        obs_list.append(obs[:-1])
        act_list.append(episode.actions)

    observations = np.concatenate(obs_list, axis=0).astype(np.float64)
    actions = np.concatenate(act_list, axis=0).astype(np.float64)
    return observations, actions


def build_cluster_index(observations, n_clusters, random_state):
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(observations)
    elapsed = time.time() - start_time

    cluster_indices = {}
    for cluster_id in range(n_clusters):
        cluster_indices[cluster_id] = np.where(labels == cluster_id)[0]

    return kmeans, cluster_indices, elapsed


def trajectory_distances(trajectory, reference_obs, reference_act, kmeans, cluster_indices, nearest_set_size):
    obs = trajectory["obs"]
    act = trajectory["act"]
    if len(obs) != len(act):
        length = min(len(obs), len(act))
        obs = obs[:length]
        act = act[:length]

    clusters = kmeans.predict(obs)
    distances = []
    for timestep, cluster_id in enumerate(clusters):
        candidate_indices = cluster_indices[int(cluster_id)]
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(len(reference_obs))

        candidate_obs = reference_obs[candidate_indices]
        obs_distance = np.linalg.norm(candidate_obs - obs[timestep], axis=1)
        nearest_local = np.argsort(obs_distance)[:nearest_set_size]
        nearest_global = candidate_indices[nearest_local]

        reference_obs_act = np.concatenate(
            [reference_obs[nearest_global], reference_act[nearest_global]], axis=1)
        traj_obs_act = np.concatenate([obs[timestep], act[timestep]], axis=0)
        total_distance = np.linalg.norm(reference_obs_act - traj_obs_act, axis=1)
        distances.append(float(np.min(total_distance)))

    return np.asarray(distances, dtype=np.float64)


def ksim_score(distances, gamma):
    eps = 1e-12
    return float(np.minimum(1.0, gamma / np.maximum(distances, eps)).mean())


def calc_ksim(args):
    trajectories = load_collected_trajectories(args.method, args.group, args.ckpt, args.n)
    reference_obs, reference_act = load_minari_reference(args.group, args.download_dataset)
    if reference_obs.shape[1] != trajectories[0]["obs"].shape[1]:
        raise ValueError(
            f"Reference obs dim {reference_obs.shape[1]} does not match trajectory obs dim "
            f"{trajectories[0]['obs'].shape[1]}.")
    if reference_act.shape[1] != trajectories[0]["act"].shape[1]:
        raise ValueError(
            f"Reference act dim {reference_act.shape[1]} does not match trajectory act dim "
            f"{trajectories[0]['act'].shape[1]}.")

    print(f"Method: {args.method}")
    print(f"Group: {args.group}")
    print(f"Collected trajectories: {len(trajectories)}")
    print(f"Reference transitions: {len(reference_obs)}")
    print(f"Obs dim: {reference_obs.shape[1]}, act dim: {reference_act.shape[1]}")

    kmeans, cluster_indices, elapsed = build_cluster_index(
        reference_obs, args.n_clusters, args.random_state)
    print(f"KMeans clustering took {elapsed:.2f} seconds")
    if args.verbose:
        for cluster_id in range(args.n_clusters):
            print(f"Cluster {cluster_id}: {len(cluster_indices[cluster_id])} observations")

    results = []
    all_distances = []
    for trajectory in trajectories:
        distances = trajectory_distances(
            trajectory, reference_obs, reference_act, kmeans, cluster_indices, args.nearest_set_size)
        score = ksim_score(distances, args.gamma)
        all_distances.append(distances)
        result = {
            "seed": trajectory["seed"],
            "path": str(trajectory["path"]),
            "steps": int(len(distances)),
            "score": score,
            "mean_distance": float(np.mean(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
        }
        results.append(result)
        print(
            f"seed={result['seed']} score={result['score']:.6f} "
            f"mean_distance={result['mean_distance']:.6f} steps={result['steps']}")

    scores = np.asarray([item["score"] for item in results], dtype=np.float64)
    concat_distances = np.concatenate(all_distances)
    summary = {
        "method": args.method,
        "group": args.group,
        "ckpt": args.ckpt,
        "n_trajectories": len(results),
        "n_clusters": args.n_clusters,
        "nearest_set_size": args.nearest_set_size,
        "gamma": args.gamma,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "overall_score": ksim_score(concat_distances, args.gamma),
        "mean_distance": float(np.mean(concat_distances)),
        "results": results,
    }
    print("\nSummary")
    print(f"mean_score={summary['mean_score']:.6f}")
    print(f"std_score={summary['std_score']:.6f}")
    print(f"overall_score={summary['overall_score']:.6f}")
    print(f"mean_distance={summary['mean_distance']:.6f}")

    if args.output is not None:
        output = Path(args.output)
    else:
        output = method_dir(args.method, args.group, args.ckpt) / "ksim_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute K-sim for collected AntMaze trajectories.")
    parser.add_argument("--method", choices=sorted(METHODS.keys()), required=True)
    parser.add_argument("--group", choices=["medium", "large"], required=True)
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--n", type=int, default=None, help="Number of collected trajectories to use. Default: all.")
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--nearest-set-size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--download-dataset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    calc_ksim(parse_args())
