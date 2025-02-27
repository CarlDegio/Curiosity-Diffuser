import os
import h5py
import cv2
import torchvision
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.utils.utils import dict_apply
import matplotlib.pyplot as plt

class AlohaDatasetNPZ(BaseDataset):
    def __init__(self, dataset_dir: str, succ_only: bool = False):
        super().__init__()
        if succ_only:
            all_data = np.load(os.path.join(dataset_dir, "aloha_stack_dataset_succ_only.npz"))
        else:
            all_data = np.load(os.path.join(dataset_dir, "aloha_stack_dataset.npz"))
            
        self.o_dim, self.a_dim = 7+512+512, 7
        self.horizon = 32
        self.seq_obs = all_data["seq_obs"]
        self.seq_act = all_data["seq_act"]
        self.seq_rew = all_data["seq_rew"]
        self.seq_val = all_data["seq_val"]
        self.indices = all_data["indices"]
        print("load aloha_stack_dataset.npz done")
        
        
    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]

        data = {
            'obs': {
                'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
            'rew': self.seq_rew[path_idx, start:end],
            'val': self.seq_val[path_idx, start],
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data
        

class AlohaDataset(BaseDataset):
    def __init__(
            self,
            dataset_dir: str,
            horizon: int = 32,
            max_path_length: int = 1000,
            noreaching_penalty: float = -100.,
            discount: float = 0.99,
            succ_only: bool = False,
    ):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = torchvision.models.resnet18(pretrained=True)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        resnet.eval()
        resnet.to(device)

        # self.normalizers = {
        #     "state": GaussianNormalizer(observations)}
        # normed_observations = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = 7+512+512, 7

        self.indices = []
        self.seq_obs, self.seq_act, self.seq_rew = [], [], []
        self.tml_and_not_timeout = []

        self.path_lengths, ptr = [], 0
        path_idx = 0
        
        image_keys = ['cam_high', 'cam_left_wrist']
        
        for i in tqdm(range(0,60), desc="Processing aloha episodes"):
            
            qpos = np.zeros((1000, 7), dtype=np.float32)
            actions = np.zeros((1000, 7), dtype=np.float32)
            images = {key: [] for key in image_keys}
            images_features_list = {key: [] for key in image_keys}
            images_features = {key: np.zeros((1000, 512), dtype=np.float32) for key in image_keys}
            length = 0
            
            with h5py.File(os.path.join(dataset_dir, f"episode_{i}.hdf5"), 'r') as f:
                compress_len = f['compress_len']
                action_np = np.array(f['action'],dtype=np.float32)
                length = action_np.shape[0]
                actions[:length] = action_np
                qpos_np = np.array(f['observations/qpos'],dtype=np.float32)
                qpos[:length] = qpos_np
                images_hdf5 = f['observations/images']
                compressed_images = {key: images_hdf5[key] for key in image_keys}
                compress_len = {image_keys[0]: compress_len[0], image_keys[1]: compress_len[1]}
                
                # process images
                for key in image_keys:
                    for j in range(compress_len[key].shape[0]):
                        image = compressed_images[key][j][:int(compress_len[key][j])]
                        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                        images[key].append(image)
                    images[key] = np.stack(images[key]).transpose(0,3,1,2)
                    
            images['cam_high'] = images['cam_high'][:,:,:,:480]
            
            with torch.no_grad():
                for key in image_keys:
                    chunk = [[0,250],[250,500],[500,750],[750,1000]]
                    for chunk_idx in chunk:
                        images_cuda = torch.from_numpy(images[key][chunk_idx[0]:chunk_idx[1]]).float().to(device)
                        images_features_list[key].append(resnet(images_cuda).squeeze().cpu())
            images_features['cam_high'][:length] = np.concatenate(images_features_list['cam_high'], axis=0)
            images_features['cam_left_wrist'][:length] = np.concatenate(images_features_list['cam_left_wrist'], axis=0)
            
            self.path_lengths.append(length)

            _seq_obs = np.concatenate([qpos, images_features['cam_high'], images_features['cam_left_wrist']], axis=1)
            _seq_act = actions
            _seq_rew = -np.ones((1000, 1), dtype=np.float32)
            if i < 60:
                _seq_rew[-1] = 100
            else:
                _seq_rew[-1] = -100

            self.seq_obs.append(_seq_obs)
            self.seq_act.append(_seq_act)
            self.seq_rew.append(_seq_rew)
            
            # max_start = min(self.path_lengths[-1] - 1, max_path_length - horizon)
            max_start = self.path_lengths[-1] - horizon
            self.indices += [(path_idx, start, start + horizon) for start in range(max_start + 1)]
            path_idx += 1

        

        self.seq_obs = np.stack(self.seq_obs)
        self.seq_act = np.stack(self.seq_act)
        self.seq_rew = np.stack(self.seq_rew)

        self.seq_val = np.copy(self.seq_rew)
        for i in range(max_path_length - 1):
            self.seq_val[:, - 2 - i] = self.seq_rew[:, -2 - i] + discount * self.seq_val[:, -1 - i]
        
        np.savez(os.path.join(dataset_dir, "aloha_stack_dataset_succ_only.npz"), seq_obs=self.seq_obs, seq_act=self.seq_act, seq_rew=self.seq_rew, seq_val=self.seq_val, indices=self.indices)

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]

        data = {
            'obs': {
                'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
            'rew': self.seq_rew[path_idx, start:end],
            'val': self.seq_val[path_idx, start],
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data


def read_hdf5_files(dataset_dir):
    for i in range(81):  # 假设最多有100个episode文件
        file_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as f:
                print(f"结构 of {file_path}:")
                # f.visit(print)  # 打印文件结构
                
                obs = f['observations']
                act = f['action']
                print(i,act.shape)
                compress_len = f['compress_len']
                images = f['observations']['images']
                cam_high = images['cam_high']
                cam_left_wrist = images['cam_left_wrist']
                
                test_compress_len = int(compress_len[0][500])
                test_cam_high = cam_high[500][:test_compress_len]
                test_cam_high = cv2.imdecode(test_cam_high, cv2.IMREAD_COLOR)
                test_cam_high = cv2.cvtColor(test_cam_high, cv2.COLOR_BGR2RGB)
                # print(test_cam_high.shape)
                # cv2.imwrite('cam_high.png', test_cam_high)
                # cv2.imshow('cam_high', test_cam_high)
                # cv2.waitKey(0)
                
                
                test_compress_len = int(compress_len[1][200])
                test_cam_left_wrist = cam_left_wrist[200][:test_compress_len]
                test_cam_left_wrist = cv2.imdecode(test_cam_left_wrist, cv2.IMREAD_COLOR)
                test_cam_left_wrist = cv2.cvtColor(test_cam_left_wrist, cv2.COLOR_BGR2RGB)
                # cv2.imwrite('cam_left_wrist.png', test_cam_left_wrist)
                # cv2.imshow('cam_left_wrist', test_cam_left_wrist)
                # cv2.waitKey(0)
                
                
        else:
            break  # 如果文件不存在，停止循环

if __name__ == "__main__":
    cv2.destroyAllWindows()
    plt.show()
    dataset_dir = "dev/aloha"
    # read_hdf5_files(dataset_dir)
    AlohaDataset(dataset_dir)
    # AlohaDatasetNPZ(dataset_dir)