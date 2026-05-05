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

STATE_METRIC_DIMS = slice(0, 2)
STATE_METRIC_NAME = "xy"


def env_name(group):
    aliases = {
        "medium": "medium-play",
        "large": "large-play",
        "medium-play": "medium-play",
        "large-play": "large-play",
        "medium-diverse": "medium-diverse",
        "large-diverse": "large-diverse",
    }
    if group not in aliases:
        raise ValueError(f"Unsupported group: {group}")
    return f"D4RL/antmaze/{aliases[group]}-v1"


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


def state_metric(observations):
    return observations[:, STATE_METRIC_DIMS]


def fit_sa_standardizer(reference_obs, reference_act):
    reference_state = state_metric(reference_obs)
    reference_sa = np.concatenate([reference_state, reference_act], axis=1)
    mean = reference_sa.mean(axis=0)
    std = reference_sa.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    state_dim = reference_state.shape[1]
    return {
        "mean": mean,
        "std": std,
        "obs_mean": mean[:state_dim],
        "obs_std": std[:state_dim],
        "act_mean": mean[state_dim:],
        "act_std": std[state_dim:],
    }


def normalize_reference(reference_obs, reference_act, standardizer):
    reference_state = state_metric(reference_obs)
    return (
        (reference_state - standardizer["obs_mean"]) / standardizer["obs_std"],
        (reference_act - standardizer["act_mean"]) / standardizer["act_std"],
    )


def normalize_trajectory(trajectory, standardizer):
    normalized = dict(trajectory)
    trajectory_state = state_metric(trajectory["obs"])
    normalized["obs"] = (trajectory_state - standardizer["obs_mean"]) / standardizer["obs_std"]
    normalized["act"] = (trajectory["act"] - standardizer["act_mean"]) / standardizer["act_std"]
    return normalized


DISTANCE_MODES = {
    "state": "state",
    "action": "action",
    "state_action": "state_action",
}


def distance_scales(state_dim, act_dim):
    return {
        "state": float(np.sqrt(state_dim)),
        "action": float(np.sqrt(act_dim)),
        "state_action": float(np.sqrt(state_dim + act_dim)),
    }


def trajectory_distances(trajectory, reference_obs, reference_act, kmeans, cluster_indices, nearest_set_size):
    obs = trajectory["obs"]
    act = trajectory["act"]
    if len(obs) != len(act):
        length = min(len(obs), len(act))
        obs = obs[:length]
        act = act[:length]

    clusters = kmeans.predict(obs)
    distances = {mode: [] for mode in DISTANCE_MODES}
    for timestep, cluster_id in enumerate(clusters):
        candidate_indices = cluster_indices[int(cluster_id)]
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(len(reference_obs))

        candidate_obs = reference_obs[candidate_indices]
        obs_distance = np.linalg.norm(candidate_obs - obs[timestep], axis=1)
        nearest_local = np.argsort(obs_distance)[:nearest_set_size]
        nearest_global = candidate_indices[nearest_local]

        candidate_act = reference_act[nearest_global]
        action_distance = np.linalg.norm(candidate_act - act[timestep], axis=1)
        nearest_action_local = int(np.argmin(action_distance))
        nearest_index = nearest_global[nearest_action_local]

        state_distance = np.linalg.norm(reference_obs[nearest_index] - obs[timestep])
        selected_action_distance = np.linalg.norm(reference_act[nearest_index] - act[timestep])
        state_action_distance = np.sqrt(state_distance ** 2 + selected_action_distance ** 2)

        distances["state"].append(float(state_distance))
        distances["action"].append(float(selected_action_distance))
        distances["state_action"].append(float(state_action_distance))

    return {
        mode: np.asarray(mode_distances, dtype=np.float64)
        for mode, mode_distances in distances.items()
    }


def ksim_score(distances, gamma, distance_scale):
    eps = 1e-12
    return float(np.minimum(1.0, gamma * distance_scale / np.maximum(distances, eps)).mean())


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
    standardizer = fit_sa_standardizer(reference_obs, reference_act)
    reference_obs_norm, reference_act_norm = normalize_reference(reference_obs, reference_act, standardizer)
    state_dim = reference_obs_norm.shape[1]
    scales = distance_scales(state_dim, reference_act.shape[1])
    print(f"State metric: {STATE_METRIC_NAME} dims {STATE_METRIC_DIMS.start}:{STATE_METRIC_DIMS.stop}")
    print(f"Using dataset z-score standardization over ({STATE_METRIC_NAME}, a)")
    print(
        "Distance numerator scales: "
        f"sqrt(dim_{STATE_METRIC_NAME})={scales['state']:.6f}, "
        f"sqrt(dim_a)={scales['action']:.6f}, "
        f"sqrt(dim_{STATE_METRIC_NAME} + dim_a)={scales['state_action']:.6f}")

    kmeans, cluster_indices, elapsed = build_cluster_index(
        reference_obs_norm, args.n_clusters, args.random_state)
    print(f"KMeans clustering took {elapsed:.2f} seconds")
    if args.verbose:
        for cluster_id in range(args.n_clusters):
            print(f"Cluster {cluster_id}: {len(cluster_indices[cluster_id])} observations")

    results = []
    all_distances = {mode: [] for mode in DISTANCE_MODES}
    for trajectory in trajectories:
        normalized_trajectory = normalize_trajectory(trajectory, standardizer)
        distances = trajectory_distances(
            normalized_trajectory, reference_obs_norm, reference_act_norm,
            kmeans, cluster_indices, args.nearest_set_size)
        scores = {
            mode: ksim_score(mode_distances, args.gamma, scales[mode])
            for mode, mode_distances in distances.items()
        }
        for mode, mode_distances in distances.items():
            all_distances[mode].append(mode_distances)
        result = {
            "seed": trajectory["seed"],
            "path": str(trajectory["path"]),
            "steps": int(len(distances["state_action"])),
            "score": scores["state_action"],
            "scores": scores,
            "mean_distance": float(np.mean(distances["state_action"])),
            "mean_distances": {
                mode: float(np.mean(mode_distances))
                for mode, mode_distances in distances.items()
            },
            "min_distances": {
                mode: float(np.min(mode_distances))
                for mode, mode_distances in distances.items()
            },
            "max_distances": {
                mode: float(np.max(mode_distances))
                for mode, mode_distances in distances.items()
            },
        }
        results.append(result)
        print(
            f"seed={result['seed']} "
            f"score_state={scores['state']:.6f} "
            f"score_action={scores['action']:.6f} "
            f"score_state_action={scores['state_action']:.6f} "
            f"mean_distance_state_action={result['mean_distance']:.6f} "
            f"steps={result['steps']}")

    metric_summaries = {}
    for mode in DISTANCE_MODES:
        mode_scores = np.asarray([item["scores"][mode] for item in results], dtype=np.float64)
        mode_distances = np.concatenate(all_distances[mode])
        metric_summaries[mode] = {
            "distance_scale": scales[mode],
            "mean_score": float(np.mean(mode_scores)),
            "std_score": float(np.std(mode_scores)),
            "overall_score": ksim_score(mode_distances, args.gamma, scales[mode]),
            "mean_distance": float(np.mean(mode_distances)),
        }
    summary = {
        "method": args.method,
        "group": args.group,
        "ckpt": args.ckpt,
        "n_trajectories": len(results),
        "n_clusters": args.n_clusters,
        "nearest_set_size": args.nearest_set_size,
        "gamma": args.gamma,
        "state_metric": STATE_METRIC_NAME,
        "state_metric_dims": [STATE_METRIC_DIMS.start, STATE_METRIC_DIMS.stop],
        "standardization": f"dataset_zscore_over_{STATE_METRIC_NAME}_action",
        "distance_modes": list(DISTANCE_MODES),
        "distance_scales": scales,
        "metrics": metric_summaries,
        "distance_scale": scales["state_action"],
        "mean_score": metric_summaries["state_action"]["mean_score"],
        "std_score": metric_summaries["state_action"]["std_score"],
        "overall_score": metric_summaries["state_action"]["overall_score"],
        "mean_distance": metric_summaries["state_action"]["mean_distance"],
        "results": results,
    }
    print("\nSummary")
    for mode, metrics in metric_summaries.items():
        print(
            f"{mode}: mean_score={metrics['mean_score']:.6f} "
            f"std_score={metrics['std_score']:.6f} "
            f"overall_score={metrics['overall_score']:.6f} "
            f"mean_distance={metrics['mean_distance']:.6f}")

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
    parser.add_argument(
        "--group",
        choices=["medium", "large", "medium-play", "large-play", "medium-diverse", "large-diverse"],
        required=True,
        help="AntMaze group. medium/large are aliases for medium-play/large-play.")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--n", type=int, default=None, help="Number of collected trajectories to use. Default: all.")
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--nearest-set-size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--download-dataset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    calc_ksim(parse_args())
