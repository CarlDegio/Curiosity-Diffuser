import os

import d4rl
import gym
import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import datetime
from tqdm import tqdm
from cleandiffuser.classifier import CumRewClassifier, RNDClassifier
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d, MLPNNClassifier
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters
from utils import set_seed


@hydra.main(config_path="../configs/diffuser/mujoco", config_name="mujoco_rnd", version_base=None)
def pipeline(args):

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    log_file_path = os.path.join(save_path, 'log.txt')
    open(log_file_path, 'w').close()

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoDataset(
        env.get_dataset(), horizon=args.task.horizon, terminal_penalty=args.terminal_penalty, discount=args.discount)
    # dataloader = DataLoader(
    #     dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    print("dataset size:", len(dataset), ", batch size:", args.batch_size, ", obs_dim:", obs_dim, ", act_dim:", act_dim)

    # --------------- Network Architecture -----------------
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)
    nn_reward_classifier = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=1,
        model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", kernel_size=3)
    nn_rnd_classifier = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=64,
        model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", kernel_size=3)
    nn_classifier_target = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=64,
        model_dim=args.model_dim//2, emb_dim=args.model_dim, dim_mult=(1,2),
        timestep_emb_type="positional", kernel_size=3)
    nn_classifier_target.eval()
    # nn_classifier = MLPNNClassifier(
    #     args.task.horizon * (obs_dim + act_dim), out_dim=1, emb_dim=args.model_dim, hidden_dims=[1024, 512, 256])

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"======================= Parameter Report of Classifier =======================")
    report_parameters(nn_rnd_classifier, topk=8)
    print(f"==============================================================================")

    # --------------- Classifier Guidance --------------------
    # classifier = CumRewClassifier(nn_classifier, device=args.device, optim_params = {"lr": 1e-3})
    classifier = RNDClassifier(nn_rnd_classifier, nn_classifier_target, nn_reward_classifier, device=args.device)

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    # --------------- Diffusion Model --------------------
    agent = DiscreteDiffusionSDE(
        nn_diffusion, None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)

    # ---------------------- Inference ----------------------

    ckpt_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    agent.load(ckpt_path + f"diffusion_ckpt_{args.ckpt}.pt")
    agent.classifier.load(ckpt_path + f"rnd_classifier/classifier_ckpt_{args.ckpt}.pt")
    target_net_ckpt=torch.load(ckpt_path + f"rnd_classifier/rnd_classifier_target.pt")
    nn_classifier_target.load_state_dict(target_net_ckpt)
    reward_net_ckpt=torch.load(ckpt_path + f"classifier_ckpt_{args.ckpt}.pt")
    nn_reward_classifier.load_state_dict(reward_net_ckpt["model_ema"])
    
    nn_classifier_target.eval()
    nn_reward_classifier.eval()
    agent.eval()

    env_eval = gym.vector.make(args.task.env_name, args.num_envs)
    normalizer = dataset.get_normalizer()
    episode_rewards = []

    prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
    for i in range(args.num_episodes):
        
        obs_list = []
        act_list = []
        
        obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

        while not np.all(cum_done) and t < 1000 + 1:
            obs_list.append(obs)
            # normalize obs
            obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

            # sample trajectories
            prior[:, 0, :obs_dim] = obs
            traj, log = agent.sample(
                prior.repeat(args.num_candidates, 1, 1),
                solver=args.solver,
                n_samples=args.num_candidates * args.num_envs,
                sample_steps=args.sampling_steps,
                use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature)

            # select the best plan
            logp = log["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
            idx = logp.argmax(0)
            act = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)[
                    idx, torch.arange(args.num_envs), 0, obs_dim:]
            act = act.clip(-1., 1.).cpu().numpy()

            act_list.append(act)
            # step
            obs, rew, done, info = env_eval.step(act)

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
            if t % 40 == 0:
                with open(log_file_path, 'a') as f:
                    f.write(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}, \nlogp: {logp[idx, torch.arange(args.num_envs)]}, \ntime: {datetime.datetime.now()}\n')
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}, '
                          f'logp: {logp[idx, torch.arange(args.num_envs)]}', f'time: {datetime.datetime.now()}')

        episode_rewards.append(ep_reward)
        obs_array = np.stack(obs_list).transpose(1, 0, 2)
        act_array = np.stack(act_list).transpose(1, 0, 2)
        np.save(save_path + f"episode_{i}_obs.npy", obs_array)
        np.save(save_path + f"episode_{i}_act.npy", act_array)

    episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
    episode_rewards = np.array(episode_rewards)
    with open(log_file_path, 'a') as f:
        f.write(f'mean: {np.mean(episode_rewards, -1)}, std: {np.std(episode_rewards, -1)}\n')
    print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))



if __name__ == "__main__":
    pipeline()
