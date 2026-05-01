import datetime
import os

import gymnasium as gym
import hydra
import minari
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeDataset_minari
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters
from utils import set_seed


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


def flatten_antmaze_obs(obs):
    if isinstance(obs, dict):
        parts = [obs["achieved_goal"], obs["observation"], obs["desired_goal"]]
        return np.concatenate([np.asarray(part, dtype=np.float32) for part in parts], axis=-1)
    return np.asarray(obs, dtype=np.float32)


def softmax_np(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def rediffuser_select_action(traj, value_score, rnd_model, obs_dim, top_value_size, scaling):
    n_envs = value_score.shape[1]
    actions = []
    uncertainties = []
    for env_idx in range(n_envs):
        order = torch.argsort(value_score[:, env_idx], descending=True)
        selected = order[:top_value_size]
        selected_traj = traj[selected, env_idx]
        uncertainty = rnd_model(selected_traj).squeeze(-1).detach().cpu().numpy()
        probs = softmax_np(-scaling * uncertainty)
        pick = np.random.choice(len(selected), p=probs)
        actions.append(selected_traj[pick, 0, obs_dim:])
        uncertainties.append(float(uncertainty[pick]))
    return torch.stack(actions, dim=0), np.asarray(uncertainties)


@hydra.main(config_path="../configs/rediffuser/antmaze", config_name="antmaze", version_base=None)
def pipeline(args):

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    log_file_path = os.path.join(save_path, 'log.txt')
    open(log_file_path, 'w').close()

    minari_dataset = minari.load_dataset(args.task.env_name, download=args.download_dataset)
    env = minari_dataset.recover_environment(eval_env=True)
    env_id = env.spec.id
    dataset = D4RLAntmazeDataset_minari(
        minari_dataset, horizon=args.task.horizon, max_path_length=args.max_path_length,
        discount=args.discount, noreaching_penalty=args.noreaching_penalty)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    traj_dim = obs_dim + act_dim
    print("dataset size:", len(dataset), ", batch size:", args.batch_size, ", obs_dim:", obs_dim, ", act_dim:", act_dim)

    nn_diffusion = JannerUNet1d(
        traj_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)
    nn_classifier = HalfJannerUNet1d(
        args.task.horizon, traj_dim, out_dim=1,
        model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", kernel_size=3)
    rnd_model = TrajectoryRND(traj_dim, args.task.horizon, args.rnd_output_dim, args.rnd_hidden_dim).to(args.device)
    rnd_optim = torch.optim.Adam(rnd_model.predictor.parameters(), lr=args.rnd_lr)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"======================= Parameter Report of Value Classifier =======================")
    report_parameters(nn_classifier)
    print(f"======================= Parameter Report of RND Model =======================")
    report_parameters(rnd_model)
    print(f"==============================================================================")

    classifier = CumRewClassifier(nn_classifier, device=args.device)

    fix_mask = torch.zeros((args.task.horizon, traj_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, traj_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    agent = DiscreteDiffusionSDE(
        nn_diffusion, None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)

    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        classifier_lr_scheduler = CosineAnnealingLR(agent.classifier.optim, args.classifier_gradient_steps)

        agent.train()
        rnd_model.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0., "avg_loss_rnd": 0.}

        pbar = tqdm(loop_dataloader(dataloader), total=args.diffusion_gradient_steps, desc="Training ReDiffuser")
        for batch in pbar:
            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device)
            x = torch.cat([obs, act], -1)

            if n_gradient_step < args.rnd_gradient_steps:
                rnd_loss = rnd_model(x).mean()
                rnd_optim.zero_grad()
                rnd_loss.backward()
                rnd_optim.step()
                log["avg_loss_rnd"] += rnd_loss.item()

            log["avg_loss_diffusion"] += agent.update(x)['loss']
            diffusion_lr_scheduler.step()

            if n_gradient_step <= args.classifier_gradient_steps:
                log["avg_loss_classifier"] += agent.update_classifier(x, val)['loss']
                classifier_lr_scheduler.step()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_classifier"] /= args.log_interval
                log["avg_loss_rnd"] /= min(args.log_interval, max(n_gradient_step + 1, 1))
                print(f'{datetime.datetime.now()}, {log}')
                with open(log_file_path, 'a') as f:
                    f.write(f'{datetime.datetime.now()}, {log}\n')
                log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0., "avg_loss_rnd": 0.}

            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.classifier.save(save_path + f"classifier_ckpt_{n_gradient_step + 1}.pt")
                torch.save(rnd_model.state_dict(), save_path + f"rnd_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")
                agent.classifier.save(save_path + f"classifier_ckpt_latest.pt")
                torch.save(rnd_model.state_dict(), save_path + f"rnd_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    elif args.mode == "inference":

        save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
        agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        agent.classifier.load(save_path + f"classifier_ckpt_{args.ckpt}.pt")
        rnd_model.load_state_dict(torch.load(save_path + f"rnd_ckpt_{args.ckpt}.pt", map_location=args.device))

        agent.eval()
        rnd_model.eval()

        env_eval = gym.make_vec(env_id, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((args.num_envs, args.task.horizon, traj_dim), device=args.device)
        for i in range(args.num_episodes):
            obs, info = env_eval.reset()
            obs = flatten_antmaze_obs(obs)
            ep_reward, cum_done, t = np.zeros(args.num_envs), np.zeros(args.num_envs, dtype=bool), 0

            while not np.all(cum_done) and t < args.max_path_length:
                norm_obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                prior[:, 0, :obs_dim] = norm_obs
                flat_traj, log = agent.sample(
                    prior.repeat(args.num_candidates, 1, 1),
                    solver=args.solver,
                    n_samples=args.num_candidates * args.num_envs,
                    sample_steps=args.sampling_steps,
                    use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature)

                traj = flat_traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)
                value_score = log["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
                act, uncertainty = rediffuser_select_action(
                    traj, value_score, rnd_model, obs_dim, args.top_value_size, args.uncertainty_scaling)
                act = act.clip(-1., 1.).cpu().numpy()

                obs, rew, terminations, truncations, info = env_eval.step(act)
                obs = flatten_antmaze_obs(obs)
                done = np.logical_or(terminations, truncations)
                t += 1
                cum_done = np.logical_or(cum_done, done)
                ep_reward += rew

                if t % 400 == 0:
                    print(f'[t={t}] cum_rew: {ep_reward}, '
                          f'value: {value_score.max(0).values}, uncertainty: {uncertainty}')

            # AntMaze is evaluated as binary success: reached -> 1, not reached -> 0.
            episode_rewards.append(np.clip(ep_reward, 0., 1.))

        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))
        env_eval.close()

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
