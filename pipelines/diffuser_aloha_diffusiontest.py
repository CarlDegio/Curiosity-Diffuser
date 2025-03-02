import os

import gym
import hydra
import numpy as np
import torch
import datetime
from cleandiffuser.classifier import RNDClassifier
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters
from utils import set_seed
import torchvision
import time


@hydra.main(config_path="../configs/diffuser/aloha", config_name="aloha_rnd", version_base=None)
def pipeline(args):

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    log_file_path = os.path.join(save_path, 'log.txt')
    open(log_file_path, 'w').close()

    obs_dim = 7+512+512
    act_dim = 7

    # --------------- Network Architecture -----------------
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=True, kernel_size=5)
    nn_reward_classifier = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=1,
        model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", kernel_size=3)
    nn_rnd_classifier = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=64,
        model_dim=args.model_dim//2, emb_dim=args.model_dim//4, dim_mult=(1,4),
        timestep_emb_type="positional", kernel_size=3)
    nn_classifier_target = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=64,
        model_dim=args.model_dim//4, emb_dim=args.model_dim//2, dim_mult=(1,2),
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
    
    resnet = torchvision.models.resnet18(pretrained=True).to(args.device)
    resnet.eval()
    nn_classifier_target.eval()
    nn_reward_classifier.eval()
    agent.eval()

    env=gym.make("Aloha-v0")
    # normalizer = dataset.get_normalizer()

    prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
    for i in range(args.num_episodes):

        obs_list = []
        act_list = []
        
        obs = env.reset()

        action_run = 10
        t=0
        while t*action_run < 1000:
            start_time = time.time()
            with torch.no_grad():
                qpos, cam_high, cam_left_wrist = obs  # RGB格式的图像
                cam_high = cam_high.transpose(2, 0, 1) # 转化为(3, 480, 640)
                cam_left_wrist = cam_left_wrist.transpose(2, 0, 1)
                cam_high = cam_high[:,:,:480]
                cam_high = torch.from_numpy(cam_high.unsqueeze(0)).float().to(args.device) # 转化为(1, 3, 480, 640)
                cam_high = resnet(cam_high).squeeze() # (1, 512)
                cam_left_wrist = torch.from_numpy(cam_left_wrist.unsqueeze(0)).float().to(args.device)
                cam_left_wrist = resnet(cam_left_wrist).squeeze() # (1, 512)
                qpos = torch.from_numpy(qpos).float().to(args.device).unsqueeze(0)
                obs = torch.cat([qpos, cam_high, cam_left_wrist], dim=1) # (1, 7+1024)
            
            # obs_list.append(obs)

            # obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

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
            act_seq = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)[
                    idx, torch.arange(args.num_envs), :, obs_dim:]  # 尺寸不确定，可能得试试
            act_seq = act_seq.cpu().numpy()

            # act_list.append(act)
            # step
            for tick in range(action_run): # 执行10个动作，共0.1s，再推理
                obs = env.step(act_seq[tick])
            
            
            end_time = time.time()
            print(f"Hz: {1/(end_time - start_time)}")

            t += 1
            if t % 40 == 0:
                print('action', act_seq, 'obs', obs)

        # clip the reward to [0, 1] since the max cumulative reward is 1
        # obs_array = np.stack(obs_list).transpose(1, 0, 2)
        # act_array = np.stack(act_list).transpose(1, 0, 2)
        # np.save(save_path + f"episode_{i}_obs.npy", obs_array)
        # np.save(save_path + f"episode_{i}_act.npy", act_array)



if __name__ == "__main__":
    pipeline()
