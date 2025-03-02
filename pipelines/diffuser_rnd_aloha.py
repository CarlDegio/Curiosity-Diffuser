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
from cleandiffuser.dataset.aloha_stack_dataset import AlohaDataset, AlohaDatasetNPZ
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d, MLPNNClassifier
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters
from utils import set_seed


@hydra.main(config_path="../configs/diffuser/aloha", config_name="aloha_rnd", version_base=None)
def pipeline(args):

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    log_file_path = os.path.join(save_path, 'log.txt')
    open(log_file_path, 'w').close()

    # ---------------------- Create Dataset ----------------------
    dataset = AlohaDatasetNPZ(
        'dev/aloha', succ_only=True)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    print("dataset size:", len(dataset), ", batch size:", args.batch_size, ", obs_dim:", obs_dim, ", act_dim:", act_dim)

    # --------------- Network Architecture -----------------
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)
    nn_classifier = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=64,
        model_dim=args.model_dim//2, emb_dim=args.model_dim//4, dim_mult=(1,4),
        timestep_emb_type="positional", kernel_size=3)
    nn_classifier_target = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=64,
        model_dim=args.model_dim//4, emb_dim=args.model_dim//2, dim_mult=(1,2),
        timestep_emb_type="positional", kernel_size=3)
    torch.save(nn_classifier_target.state_dict(), save_path + f"rnd_classifier_target.pt")
    nn_classifier_target.eval()

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"======================= Parameter Report of Classifier =======================")
    report_parameters(nn_classifier, topk=8)
    print(f"==============================================================================")

    # --------------- Classifier Guidance --------------------
    classifier = RNDClassifier(nn_classifier, nn_classifier_target, device=args.device)

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

    # ---------------------- Training ----------------------
    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        classifier_lr_scheduler = CosineAnnealingLR(agent.classifier.optim, args.classifier_gradient_steps)

        agent.train()

        classifier_gradient_step = 0
        diffusion_gradient_step = 0
        log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}
        
        # Add progress bar for classifier training
        classifier_pbar = tqdm(loop_dataloader(dataloader), 
                             total=args.classifier_gradient_steps,
                             desc="Training Classifier")
        diffusion_pbar = tqdm(loop_dataloader(dataloader), 
                             total=args.diffusion_gradient_steps,
                             desc="Training Diffusion")
        
        
        for batch in classifier_pbar:
            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device)
            x = torch.cat([obs, act], -1)
            # ----------- Gradient Step ------------
            log["avg_loss_classifier"] += agent.update_classifier(x, val)['loss']
            classifier_lr_scheduler.step()
            # ----------- Logging ------------
            if (classifier_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = classifier_gradient_step + 1
                log["avg_loss_classifier"] /= args.log_interval
                print(f'{datetime.datetime.now()}, {log}')
                with open(log_file_path, 'a') as f:
                    f.write(f'{datetime.datetime.now()}, {log}\n')
                log = {"avg_loss_classifier": 0.}
            # ----------- Saving ------------
            if (classifier_gradient_step + 1) % args.save_interval_classifier == 0:
                agent.classifier.save(save_path + f"classifier_ckpt_{classifier_gradient_step + 1}.pt")
                agent.classifier.save(save_path + f"classifier_ckpt_latest.pt")
            
            classifier_gradient_step += 1
            if classifier_gradient_step >= args.classifier_gradient_steps:
                break
            
        log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}
            
        for batch in diffusion_pbar:

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device)

            x = torch.cat([obs, act], -1)

            # ----------- Gradient Step ------------
            log["avg_loss_diffusion"] += agent.update(x)['loss']
            diffusion_lr_scheduler.step()
            # ----------- Logging ------------
            if (diffusion_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = diffusion_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                print(f'{datetime.datetime.now()}, {log}')
                with open(log_file_path, 'a') as f:
                    f.write(f'{datetime.datetime.now()}, {log}\n')
                log = {"avg_loss_diffusion": 0.}

            # ----------- Saving ------------
            if (diffusion_gradient_step + 1) % args.save_interval_diffusion == 0:
                agent.save(save_path + f"diffusion_ckpt_{diffusion_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")

            diffusion_gradient_step += 1
            if diffusion_gradient_step >= args.diffusion_gradient_steps:
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":

        save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
        agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        agent.classifier.load(save_path + f"classifier_ckpt_{args.ckpt}.pt")

        agent.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:
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

                # step
                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += rew
                print(f'[t={t}] xy: {obs[:, :2]}')
                print(f'[t={t}] cum_rew: {ep_reward}, '
                      f'logp: {logp[idx, torch.arange(args.num_envs)]}', f'time: {datetime.datetime.now()}')

            # clip the reward to [0, 1] since the max cumulative reward is 1
            episode_rewards.append(np.clip(ep_reward, 0., 1.))

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
