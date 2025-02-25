import hydra
import d4rl
import gym
import time  # 添加time模块用于控制渲染速度
from utils import set_seed
import numpy as np  # 添加numpy用于数组操作

@hydra.main(config_path="../configs/diffuser/antmaze", config_name="antmaze_rnd", version_base=None)
def pipeline(args):
    set_seed(1)
    env = gym.make(args.task.env_name)
    obs = env.reset()
    

    # 重置环境
    
    
    # 渲染一段时间的环境
    for step in range(1000):  # 可以根据需要调整步数
        # 随机动作，仅用于演示
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # 获取当前机器人位置
        robot_position = obs[:2]  # 假设前两个元素是xy位置
        print("机器人当前xy位置:", robot_position)

        # 渲染环境
        # env.render()
        
        # 添加短暂延迟使渲染更容易观察
        time.sleep(0.01)
        
        if done:
            obs = env.reset()
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    pipeline()
