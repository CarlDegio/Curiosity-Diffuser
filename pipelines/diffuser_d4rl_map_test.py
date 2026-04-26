import hydra
import d4rl
import gym
from tqdm import tqdm
import torch
from utils import set_seed
import numpy as np
import cv2  # 添加 OpenCV 用于图像显示
from cleandiffuser.classifier import CumRewClassifier, RNDClassifier
from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d, MLPNNClassifier
from cleandiffuser.nn_diffusion import JannerUNet1d

def load_policy(args, env):
    dataset = D4RLAntmazeDataset(
        env.get_dataset(), horizon=args.task.horizon, discount=args.discount,
        noreaching_penalty=args.noreaching_penalty,)
    # dataset = D4RLMuJoCoDataset(
    #     env.get_dataset(), horizon=args.task.horizon, terminal_penalty=args.terminal_penalty, discount=args.discount)
    
    obs_dim = dataset.o_dim
    act_dim = dataset.a_dim
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
    
    classifier = RNDClassifier(nn_rnd_classifier, nn_classifier_target, nn_reward_classifier, device=args.device)
    
    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight
    
    agent = DiscreteDiffusionSDE(
        nn_diffusion, None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)
    
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

    normalizer = dataset.get_normalizer()
    return agent, normalizer, obs_dim, act_dim

@hydra.main(config_path="../configs/diffuser/antmaze", config_name="antmaze_rnd", version_base=None)
def pipeline(args):
    args.num_envs = 1
    set_seed(2)
    RESET = R = 'r'  # Reset position.
    GOAL = G = 'g'  
    BIG_MAZE_TEST = [[1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]
    
    env = gym.make(args.task.env_name)
    
    agent, normalizer, obs_dim, act_dim = load_policy(args, env)
    
    
    obs_list = []
    act_list = []
    rew_list = []
    
    obs, ep_reward = env.reset(), 0

    # 获取第一帧渲染图像
    render_mode = 'rgb_array'  # 使用 rgb_array 模式获取 NumPy 数组
    frame = env.render(camera_name = "birdview",mode=render_mode)
    print(frame.shape)
    
    frame_list = []
    # 显示第一帧
    if frame is not None:
        # 将 RGB 转换为 BGR (OpenCV 使用 BGR 格式)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_list.append(frame_bgr)
    else:
        print("无法获取渲染帧")
        
    cv2.imwrite("first_frame.png", frame_bgr)
    
    prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
    
    for step in tqdm(range(1000), desc="testing"):  # 可以根据需要调整步数
        # 随机动作，仅用于演示
        obs_list.append(obs)
        obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
        
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
        act = act.clip(-1., 1.).cpu().squeeze().numpy()

        act_list.append(act)
        act_list.append(act)
        # step
        obs, rew, done, info = env.step(act)
        rew_list.append(rew)
        # 渲染环境并获取帧
        frame = env.render(mode=render_mode)
        
        # 显示当前帧
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_list.append(frame_bgr)
        
        if done:
            break
    
    
    # 关闭环境和窗口
    env.close()
    cv2.destroyAllWindows()

    # 保存帧列表为视频
    output_video_path = f'output_video_{args.task.env_name}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 60.0, (frame_list[0].shape[1], frame_list[0].shape[0]))

    for frame in frame_list:
        out.write(frame)
    out.release()
    print(f"video saved to {output_video_path}")
    
    np.save("obs_list.npy", obs_list)
    np.save("act_list.npy", act_list)
    print("obs_list and act_list saved")
    
    print("rew",sum(rew_list))

if __name__ == "__main__":
    pipeline()
