import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Sign_Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_samples=50):
        # root_dir is expected to be e.g. /kaggle/input/.../keypoints_splited
        self.split = split
        self.data_dir = os.path.join(root_dir, split)
        self.num_samples = num_samples
        
        self.video_paths = []
        self.labels = []
        
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for file_name in os.listdir(cls_dir):
                if file_name.endswith('.npy'):
                    self.video_paths.append(os.path.join(cls_dir, file_name))
                    self.labels.append(self.class_to_idx[cls_name])
                    
        # Simulate label encoder for train script compatibility
        class DummyEncoder:
            pass
        self.label_encoder = DummyEncoder()
        self.label_encoder.classes_ = self.classes
                    
    def __len__(self):
        return len(self.video_paths)
        
    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label = self.labels[index]
        
        # Load .npy data of shape [T, 76, 3]
        data = np.load(video_path)
        T = data.shape[0] if len(data.shape) == 3 else 0
        
        if T == 0:
            data = np.zeros((self.num_samples, 76, 3), dtype=np.float32)
        elif T < self.num_samples:
            pad_len = self.num_samples - T
            pad = np.zeros((pad_len, 76, 3), dtype=data.dtype)
            data = np.concatenate((data, pad), axis=0)
        elif T > self.num_samples:
            indices = np.linspace(0, T - 1, self.num_samples, dtype=int)
            data = data[indices]
            
        # TGCN expects [76, num_samples * 3]
        data = data.transpose(1, 0, 2).reshape(76, -1)
        
        if self.split == 'test':
            # Nhân bản 4 lần để khớp với biến num_copies = 4 bị hardcode trong train_utils.py
            data = np.concatenate([data, data, data, data], axis=-1)
        
        video_id = os.path.basename(video_path)
        return torch.FloatTensor(data), label, video_id
