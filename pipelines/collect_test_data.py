import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import minari
import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import CumRewClassifier, RNDClassifier
from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeDataset_minari
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d


class TrajectoryRND(nn.Module):
    def __init__(self, in_dim, horizon, out_dim=64, hidden_dim=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_dim * horizon, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))
        self.target = nn.Sequential(
            nn.Linear(in_dim * horizon, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        pred = self.predictor(x)
        target = self.target(x).detach()
        return ((pred - target) ** 2).mean(dim=-1, keepdim=True)


def parse_seed_list(value):
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_antmaze_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([
            np.asarray(obs["achieved_goal"], dtype=np.float32),
            np.asarray(obs["observation"], dtype=np.float32),
            np.asarray(obs["desired_goal"], dtype=np.float32),
        ], axis=-1)
    return np.asarray(obs, dtype=np.float32)


def squeeze_single_env(value):
    value = np.asarray(value)
    if value.shape[:1] == (1,):
        return value[0]
    return value


def resolve_ckpt_dir(pipeline_name, env_name):
    return Path("results") / pipeline_name / env_name


def antmaze_env_name(group):
    aliases = {
        "medium": "medium-play",
        "large": "large-play",
        "medium-play": "medium-play",
        "large-play": "large-play",
        "medium-diverse": "medium-diverse",
        "large-diverse": "large-diverse",
    }
    if group not in aliases:
        raise ValueError(f"Unsupported AntMaze group: {group}")
    return f"D4RL/antmaze/{aliases[group]}-v1"


def ckpt_path(ckpt_dir, prefix, ckpt):
    return ckpt_dir / f"{prefix}_ckpt_{ckpt}.pt"


def build_dataset_and_env(args):
    minari_dataset = minari.load_dataset(args.env_name, download=args.download_dataset)
    env = minari_dataset.recover_environment(eval_env=True)
    env_id = env.spec.id
    env.close()
    dataset = D4RLAntmazeDataset_minari(
        minari_dataset,
        horizon=args.horizon,
        max_path_length=args.max_path_length,
        discount=args.discount,
        noreaching_penalty=args.noreaching_penalty,
    )
    return dataset, env_id


def build_networks(args, obs_dim, act_dim, classifier_kind):
    traj_dim = obs_dim + act_dim
    nn_diffusion = JannerUNet1d(
        traj_dim,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=tuple(args.dim_mult),
        timestep_emb_type="positional",
        attention=False,
        kernel_size=5,
    )

    if classifier_kind == "reward":
        nn_classifier = HalfJannerUNet1d(
            args.horizon,
            traj_dim,
            out_dim=1,
            model_dim=args.model_dim,
            emb_dim=args.model_dim,
            dim_mult=tuple(args.dim_mult),
            timestep_emb_type="positional",
            kernel_size=3,
        )
        classifier = CumRewClassifier(nn_classifier, device=args.device)
    elif classifier_kind == "curiosity":
        nn_rnd_classifier = HalfJannerUNet1d(
            args.horizon,
            traj_dim,
            out_dim=64,
            model_dim=args.model_dim,
            emb_dim=args.model_dim,
            dim_mult=tuple(args.dim_mult),
            timestep_emb_type="positional",
            kernel_size=3,
        )
        nn_target = HalfJannerUNet1d(
            args.horizon,
            traj_dim,
            out_dim=64,
            model_dim=args.model_dim // 2,
            emb_dim=args.model_dim,
            dim_mult=(1, 2),
            timestep_emb_type="positional",
            kernel_size=3,
        )
        nn_reward = HalfJannerUNet1d(
            args.horizon,
            traj_dim,
            out_dim=1,
            model_dim=args.model_dim,
            emb_dim=args.model_dim,
            dim_mult=tuple(args.dim_mult),
            timestep_emb_type="positional",
            kernel_size=3,
        )
        classifier = RNDClassifier(
            nn_rnd_classifier,
            nn_target,
            nn_reward,
            device=args.device,
            curiosity_weight=args.curiosity_weight,
        )
    else:
        raise ValueError(f"Unknown classifier kind: {classifier_kind}")

    fix_mask = torch.zeros((args.horizon, traj_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.horizon, traj_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    agent = DiscreteDiffusionSDE(
        nn_diffusion,
        None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=classifier,
        ema_rate=args.ema_rate,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
        predict_noise=args.predict_noise,
    )
    return agent


def load_diffuser_agent(args, obs_dim, act_dim):
    ckpt_dir = Path(args.diffuser_ckpt_dir or resolve_ckpt_dir(args.diffuser_pipeline, args.env_name))
    agent = build_networks(args, obs_dim, act_dim, "reward")
    agent.load(str(ckpt_path(ckpt_dir, "diffusion", args.ckpt)))
    agent.classifier.load(str(ckpt_path(ckpt_dir, "classifier", args.ckpt)))
    agent.eval()
    return agent, ckpt_dir


def load_curiosity_diffuser_agent(args, obs_dim, act_dim):
    diffuser_dir = Path(args.diffuser_ckpt_dir or resolve_ckpt_dir(args.diffuser_pipeline, args.env_name))
    rnd_dir = Path(args.curiosity_ckpt_dir or resolve_ckpt_dir(args.curiosity_pipeline, args.env_name))
    agent = build_networks(args, obs_dim, act_dim, "curiosity")

    agent.load(str(ckpt_path(diffuser_dir, "diffusion", args.ckpt)))
    reward_ckpt = torch.load(ckpt_path(diffuser_dir, "classifier", args.ckpt), map_location=args.device)
    agent.classifier.reward_model.load_state_dict(reward_ckpt["model_ema"])

    agent.classifier.load(str(ckpt_path(rnd_dir, "classifier", args.ckpt)))
    target_ckpt = torch.load(rnd_dir / "rnd_classifier_target.pt", map_location=args.device)
    agent.classifier.target_model.load_state_dict(target_ckpt)

    agent.classifier.target_model.eval()
    agent.classifier.reward_model.eval()
    agent.eval()
    return agent, rnd_dir


def load_curiosity_rnd_evaluator(args, obs_dim, act_dim):
    rnd_dir = Path(args.curiosity_ckpt_dir or resolve_ckpt_dir(args.curiosity_pipeline, args.env_name))
    traj_dim = obs_dim + act_dim
    rnd_model = HalfJannerUNet1d(
        args.horizon,
        traj_dim,
        out_dim=64,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=tuple(args.dim_mult),
        timestep_emb_type="positional",
        kernel_size=3,
    ).to(args.device)
    target_model = HalfJannerUNet1d(
        args.horizon,
        traj_dim,
        out_dim=64,
        model_dim=args.model_dim // 2,
        emb_dim=args.model_dim,
        dim_mult=(1, 2),
        timestep_emb_type="positional",
        kernel_size=3,
    ).to(args.device)

    rnd_ckpt = torch.load(ckpt_path(rnd_dir, "classifier", args.ckpt), map_location=args.device)
    rnd_model.load_state_dict(rnd_ckpt["model_ema"])
    target_ckpt = torch.load(rnd_dir / "rnd_classifier_target.pt", map_location=args.device)
    target_model.load_state_dict(target_ckpt)
    rnd_model.eval()
    target_model.eval()
    return rnd_model, target_model


def load_rediffuser_agent(args, obs_dim, act_dim):
    ckpt_dir = Path(args.rediffuser_ckpt_dir or resolve_ckpt_dir(args.rediffuser_pipeline, args.env_name))
    agent = build_networks(args, obs_dim, act_dim, "reward")
    rnd_model = TrajectoryRND(
        obs_dim + act_dim,
        args.horizon,
        args.rnd_output_dim,
        args.rnd_hidden_dim,
    ).to(args.device)

    agent.load(str(ckpt_path(ckpt_dir, "diffusion", args.ckpt)))
    agent.classifier.load(str(ckpt_path(ckpt_dir, "classifier", args.ckpt)))
    rnd_model.load_state_dict(torch.load(ckpt_path(ckpt_dir, "rnd", args.ckpt), map_location=args.device))
    agent.eval()
    rnd_model.eval()
    return agent, rnd_model, ckpt_dir


def softmax_np(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def select_by_value(traj, logp, obs_dim, num_candidates, n_env):
    logp = logp.view(num_candidates, n_env, -1).sum(-1)
    idx = logp.argmax(0)
    traj = traj.view(num_candidates, n_env, traj.shape[1], traj.shape[2])
    selected_traj = traj[idx, torch.arange(n_env)]
    act = selected_traj[:, 0, obs_dim:]
    selected_logp = logp[idx, torch.arange(n_env)]
    return act, selected_traj, selected_logp


def curiosity_scores(classifier, selected_traj):
    t = torch.zeros((selected_traj.shape[0],), dtype=torch.long, device=selected_traj.device)
    with torch.no_grad():
        pred = classifier.model_ema(selected_traj, t)
        target = classifier.target_model(selected_traj, t, None)
        reward = classifier.reward_model(selected_traj, t)
        novelty = ((pred - target) ** 2).sum(dim=1, keepdim=True)
        guided_score = reward - classifier.curiosity_weight * novelty
    return reward.squeeze(-1), novelty.squeeze(-1), guided_score.squeeze(-1)


def curiosity_rnd_novelty(rnd_model, target_model, selected_traj):
    t = torch.zeros((selected_traj.shape[0],), dtype=torch.long, device=selected_traj.device)
    with torch.no_grad():
        pred = rnd_model(selected_traj, t)
        target = target_model(selected_traj, t, None)
        novelty = ((pred - target) ** 2).sum(dim=1)
    return novelty


def rediffuser_select_action(traj, value_score, rnd_model, obs_dim, top_value_size, scaling):
    n_env = value_score.shape[1]
    actions = []
    selected_trajs = []
    uncertainties = []
    selected_indices = []
    for env_idx in range(n_env):
        order = torch.argsort(value_score[:, env_idx], descending=True)
        selected = order[:top_value_size]
        selected_traj = traj[selected, env_idx]
        uncertainty = rnd_model(selected_traj).squeeze(-1).detach().cpu().numpy()
        probs = softmax_np(-scaling * uncertainty)
        pick = np.random.choice(len(selected), p=probs)
        actions.append(selected_traj[pick, 0, obs_dim:])
        selected_trajs.append(selected_traj[pick])
        uncertainties.append(float(uncertainty[pick]))
        selected_indices.append(int(selected[pick]))
    return torch.stack(actions, dim=0), torch.stack(selected_trajs, dim=0), uncertainties, selected_indices


def rollout_value_method(
        args, agent, dataset, env_id, output_dir, method_name,
        curiosity_evaluator, collect_curiosity_components=False):
    normalizer = dataset.get_normalizer()
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    traj_dim = obs_dim + act_dim
    env_eval = gym.make_vec(env_id, args.n_env)
    summary = []
    seed_values = [args.seeds + i for i in range(args.n_env)]

    set_all_seeds(args.sample_seed + args.seeds)
    obs, info = env_eval.reset(seed=args.seeds)
    obs = flatten_antmaze_obs(obs)

    obs_lists = [[] for _ in range(args.n_env)]
    act_lists = [[] for _ in range(args.n_env)]
    rew_lists = [[] for _ in range(args.n_env)]
    done_lists = [[] for _ in range(args.n_env)]
    logp_lists = [[] for _ in range(args.n_env)]
    reward_score_lists = [[] for _ in range(args.n_env)]
    rnd_novelty_lists = [[] for _ in range(args.n_env)]
    guided_score_lists = [[] for _ in range(args.n_env)]
    curiosity_rnd_novelty_lists = [[] for _ in range(args.n_env)]
    ep_reward = np.zeros(args.n_env)
    done = np.zeros(args.n_env, dtype=bool)
    steps = np.zeros(args.n_env, dtype=np.int64)
    t = 0

    prior = torch.zeros((args.n_env, args.horizon, traj_dim), device=args.device)
    while not np.all(done) and t < args.max_path_length:
        active = ~done
        for env_idx in np.where(active)[0]:
            obs_lists[env_idx].append(obs[env_idx])

        norm_obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
        prior[:, 0, :obs_dim] = norm_obs

        traj, log = agent.sample(
            prior.repeat(args.num_candidates, 1, 1),
            solver=args.solver,
            n_samples=args.num_candidates * args.n_env,
            sample_steps=args.sampling_steps,
            use_ema=args.use_ema,
            w_cg=args.w_cg,
            temperature=args.temperature,
        )
        act, selected_traj, selected_logp = select_by_value(
            traj, log["log_p"], obs_dim, args.num_candidates, args.n_env)
        act_np = act.clip(-1., 1.).cpu().numpy()
        act_np[done] = 0.

        unified_novelty = curiosity_rnd_novelty(
            curiosity_evaluator[0], curiosity_evaluator[1], selected_traj)
        if collect_curiosity_components:
            reward_score, novelty, guided_score = curiosity_scores(agent.classifier, selected_traj)

        for env_idx in np.where(active)[0]:
            act_lists[env_idx].append(act_np[env_idx])
            logp_lists[env_idx].append(float(selected_logp[env_idx].detach().cpu()))
            curiosity_rnd_novelty_lists[env_idx].append(float(unified_novelty[env_idx].detach().cpu()))
            if collect_curiosity_components:
                reward_score_lists[env_idx].append(float(reward_score[env_idx].detach().cpu()))
                rnd_novelty_lists[env_idx].append(float(novelty[env_idx].detach().cpu()))
                guided_score_lists[env_idx].append(float(guided_score[env_idx].detach().cpu()))

        obs, rew, terminations, truncations, info = env_eval.step(act_np)
        obs = flatten_antmaze_obs(obs)
        step_done = np.logical_or(terminations, truncations)
        for env_idx in np.where(active)[0]:
            rew_lists[env_idx].append(float(rew[env_idx]))
            done_lists[env_idx].append(bool(step_done[env_idx]))
            ep_reward[env_idx] += rew[env_idx]
            steps[env_idx] += 1
        done = np.logical_or(done, step_done)
        t += 1

    for env_idx, seed in enumerate(seed_values):
        saved_path = save_episode(
            output_dir,
            method_name,
            seed,
            obs_lists[env_idx],
            act_lists[env_idx],
            rew_lists[env_idx],
            done_lists[env_idx],
            np.asarray([ep_reward[env_idx]], dtype=np.float32),
            {
                "selected_logp": logp_lists[env_idx],
                "reward_score": reward_score_lists[env_idx],
                "rnd_novelty": rnd_novelty_lists[env_idx],
                "curiosity_rnd_novelty": curiosity_rnd_novelty_lists[env_idx],
                "guided_score": guided_score_lists[env_idx],
            },
        )
        summary.append({"seed": seed, "return": float(ep_reward[env_idx]), "steps": int(steps[env_idx]), "path": str(saved_path)})
        print(f"[{method_name}] seed={seed} return={ep_reward[env_idx]:.3f} steps={steps[env_idx]} saved={saved_path}")

    env_eval.close()
    save_summary(output_dir, method_name, args, summary)


def rollout_rediffuser(args, agent, rnd_model, dataset, env_id, output_dir, curiosity_evaluator):
    normalizer = dataset.get_normalizer()
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    traj_dim = obs_dim + act_dim
    env_eval = gym.make_vec(env_id, args.n_env)
    summary = []
    seed_values = [args.seeds + i for i in range(args.n_env)]

    set_all_seeds(args.sample_seed + args.seeds)
    obs, info = env_eval.reset(seed=args.seeds)
    obs = flatten_antmaze_obs(obs)

    obs_lists = [[] for _ in range(args.n_env)]
    act_lists = [[] for _ in range(args.n_env)]
    rew_lists = [[] for _ in range(args.n_env)]
    done_lists = [[] for _ in range(args.n_env)]
    value_lists = [[] for _ in range(args.n_env)]
    uncertainty_lists = [[] for _ in range(args.n_env)]
    selected_candidate_lists = [[] for _ in range(args.n_env)]
    curiosity_rnd_novelty_lists = [[] for _ in range(args.n_env)]
    ep_reward = np.zeros(args.n_env)
    done = np.zeros(args.n_env, dtype=bool)
    steps = np.zeros(args.n_env, dtype=np.int64)
    t = 0

    prior = torch.zeros((args.n_env, args.horizon, traj_dim), device=args.device)
    while not np.all(done) and t < args.max_path_length:
        active = ~done
        for env_idx in np.where(active)[0]:
            obs_lists[env_idx].append(obs[env_idx])

        norm_obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
        prior[:, 0, :obs_dim] = norm_obs

        flat_traj, log = agent.sample(
            prior.repeat(args.num_candidates, 1, 1),
            solver=args.solver,
            n_samples=args.num_candidates * args.n_env,
            sample_steps=args.sampling_steps,
            use_ema=args.use_ema,
            w_cg=args.w_cg,
            temperature=args.temperature,
        )
        traj = flat_traj.view(args.num_candidates, args.n_env, args.horizon, -1)
        value_score = log["log_p"].view(args.num_candidates, args.n_env, -1).sum(-1)
        act, selected_traj, uncertainty, selected_idx = rediffuser_select_action(
            traj,
            value_score,
            rnd_model,
            obs_dim,
            args.top_value_size,
            args.uncertainty_scaling,
        )
        act_np = act.clip(-1., 1.).cpu().numpy()
        act_np[done] = 0.

        unified_novelty = curiosity_rnd_novelty(
            curiosity_evaluator[0], curiosity_evaluator[1], selected_traj)

        for env_idx in np.where(active)[0]:
            act_lists[env_idx].append(act_np[env_idx])
            value_lists[env_idx].append(float(value_score[selected_idx[env_idx], env_idx].detach().cpu()))
            uncertainty_lists[env_idx].append(uncertainty[env_idx])
            selected_candidate_lists[env_idx].append(selected_idx[env_idx])
            curiosity_rnd_novelty_lists[env_idx].append(float(unified_novelty[env_idx].detach().cpu()))

        obs, rew, terminations, truncations, info = env_eval.step(act_np)
        obs = flatten_antmaze_obs(obs)
        step_done = np.logical_or(terminations, truncations)
        for env_idx in np.where(active)[0]:
            rew_lists[env_idx].append(float(rew[env_idx]))
            done_lists[env_idx].append(bool(step_done[env_idx]))
            ep_reward[env_idx] += rew[env_idx]
            steps[env_idx] += 1
        done = np.logical_or(done, step_done)
        t += 1

    for env_idx, seed in enumerate(seed_values):
        saved_path = save_episode(
            output_dir,
            "rediffuser",
            seed,
            obs_lists[env_idx],
            act_lists[env_idx],
            rew_lists[env_idx],
            done_lists[env_idx],
            np.asarray([ep_reward[env_idx]], dtype=np.float32),
            {
                "value_score": value_lists[env_idx],
                "rediffuser_uncertainty": uncertainty_lists[env_idx],
                "curiosity_rnd_novelty": curiosity_rnd_novelty_lists[env_idx],
                "selected_candidate": selected_candidate_lists[env_idx],
            },
        )
        summary.append({"seed": seed, "return": float(ep_reward[env_idx]), "steps": int(steps[env_idx]), "path": str(saved_path)})
        print(f"[rediffuser] seed={seed} return={ep_reward[env_idx]:.3f} steps={steps[env_idx]} saved={saved_path}")

    env_eval.close()
    save_summary(output_dir, "rediffuser", args, summary)


def save_episode(output_dir, method_name, seed, obs, act, rew, done, ep_reward, extras):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{method_name}_seed_{seed}.npz"
    payload = {
        "obs": np.asarray(obs, dtype=np.float32),
        "act": np.asarray(act, dtype=np.float32),
        "rew": np.asarray(rew, dtype=np.float32),
        "done": np.asarray(done, dtype=bool),
        "episode_return": np.asarray(ep_reward, dtype=np.float32),
        "seed": np.asarray(seed, dtype=np.int64),
    }
    for key, value in extras.items():
        if value:
            payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)
    return path


def save_summary(output_dir, method_name, args, summary):
    data = {
        "method": method_name,
        "env_name": args.env_name,
        "ckpt": args.ckpt,
        "seed_start": args.seeds,
        "n_env": args.n_env,
        "seeds": [args.seeds + i for i in range(args.n_env)],
        "summary": summary,
    }
    with open(output_dir / f"{method_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def collect_diffuser(args):
    dataset, env_id = build_dataset_and_env(args)
    agent, ckpt_dir = load_diffuser_agent(args, dataset.o_dim, dataset.a_dim)
    curiosity_evaluator = load_curiosity_rnd_evaluator(args, dataset.o_dim, dataset.a_dim)
    output_dir = ckpt_dir / "collected_test_data" / f"diffuser_ckpt_{args.ckpt}"
    rollout_value_method(
        args, agent, dataset, env_id, output_dir, "diffuser",
        curiosity_evaluator, collect_curiosity_components=False)


def collect_curiosity_diffuser(args):
    dataset, env_id = build_dataset_and_env(args)
    agent, ckpt_dir = load_curiosity_diffuser_agent(args, dataset.o_dim, dataset.a_dim)
    curiosity_evaluator = (agent.classifier.model_ema, agent.classifier.target_model)
    output_dir = ckpt_dir / "collected_test_data" / f"curiosity_diffuser_ckpt_{args.ckpt}"
    rollout_value_method(
        args, agent, dataset, env_id, output_dir, "curiosity_diffuser",
        curiosity_evaluator, collect_curiosity_components=True)


def collect_rediffuser(args):
    dataset, env_id = build_dataset_and_env(args)
    agent, rnd_model, ckpt_dir = load_rediffuser_agent(args, dataset.o_dim, dataset.a_dim)
    curiosity_evaluator = load_curiosity_rnd_evaluator(args, dataset.o_dim, dataset.a_dim)
    output_dir = ckpt_dir / "collected_test_data" / f"rediffuser_ckpt_{args.ckpt}"
    rollout_rediffuser(args, agent, rnd_model, dataset, env_id, output_dir, curiosity_evaluator)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect seeded AntMaze test trajectories for diffuser variants.")
    parser.add_argument("--method", choices=["diffuser", "curiosity_diffuser", "rediffuser"], required=True)
    parser.add_argument(
        "--group",
        choices=["medium", "large", "medium-play", "large-play", "medium-diverse", "large-diverse"],
        default="medium-play",
        help="AntMaze group. medium/large are aliases for medium-play/large-play.")
    parser.add_argument("--env-name", default=None, help="Explicit Minari env name. Overrides --group when set.")
    parser.add_argument("--seeds", type=int, default=0, help="First reset seed. Vector envs use seeds, seeds+1, ... by default.")
    parser.add_argument("--n-env", type=int, default=10, help="Number of parallel environments to collect in one run.")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--download-dataset", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--diffuser-pipeline", default="diffuser_d4rl_antmaze")
    parser.add_argument("--curiosity-pipeline", default="diffuser_d4rl_antmaze_rnd")
    parser.add_argument("--rediffuser-pipeline", default="rediffuser_minari_antmaze")
    parser.add_argument("--diffuser-ckpt-dir", default=None)
    parser.add_argument("--curiosity-ckpt-dir", default=None)
    parser.add_argument("--rediffuser-ckpt-dir", default=None)

    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-path-length", type=int, default=1000)
    parser.add_argument("--dim-mult", type=parse_seed_list, default=[1, 2, 2, 2])
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--predict-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--action-loss-weight", type=float, default=10.0)
    parser.add_argument("--ema-rate", type=float, default=0.9999)
    parser.add_argument("--solver", default="ddpm")
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w-cg", type=float, default=0.001)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--noreaching-penalty", type=float, default=-100.0)

    parser.add_argument("--curiosity-weight", type=float, default=1000000.0)
    parser.add_argument("--rnd-output-dim", type=int, default=64)
    parser.add_argument("--rnd-hidden-dim", type=int, default=256)
    parser.add_argument("--top-value-size", type=int, default=4)
    parser.add_argument("--uncertainty-scaling", type=float, default=0.2)
    args = parser.parse_args()
    if args.env_name is None:
        args.env_name = antmaze_env_name(args.group)
    return args


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")

    if args.method == "diffuser":
        collect_diffuser(args)
    elif args.method == "curiosity_diffuser":
        collect_curiosity_diffuser(args)
    elif args.method == "rediffuser":
        collect_rediffuser(args)
    else:
        raise ValueError(f"Unsupported method: {args.method}")


if __name__ == "__main__":
    main()
