import os
import cv2
import math
import numpy as np
import torch
import torch.utils.data as data_utl

def load_rgb_frames_from_video(video_path, num, resize=(224, 224)):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames == 0:
        return np.zeros((num, resize[0], resize[1], 3), dtype=np.float32)
        
    step = max(1, total_frames / num)
    
    for i in range(num):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, min(int(i * step), int(total_frames-1)))
        success, img = vidcap.read()
        if not success:
            img = np.zeros((resize[0], resize[1], 3), dtype=np.uint8)
            
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))
            
        img = (img / 255.) * 2 - 1
        frames.append(img)
        
    return np.asarray(frames, dtype=np.float32)

class Dataset(data_utl.Dataset):
    def __init__(self, split_file, split, root, mode, transforms=None):
        # root is expected to be /kaggle/input/.../frame_splited
        self.data_dir = os.path.join(root, split)
        self.transforms = transforms
        self.mode = mode
        self.num_classes = 472
        
        self.video_paths = []
        self.labels = []
        
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for file_name in os.listdir(cls_dir):
                if file_name.endswith('.mp4'):
                    self.video_paths.append(os.path.join(cls_dir, file_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label = self.labels[index]
        
        # We assume 64 frames for I3D like original 
        imgs = load_rgb_frames_from_video(video_path, 64)

        if self.transforms is not None:
            imgs = self.transforms(imgs)
            
        # I3D expects [C, T, H, W]
        imgs = torch.from_numpy(imgs.transpose([3, 0, 1, 2]))
        
        # one-hot label
        label_onehot = np.zeros(self.num_classes, dtype=np.float32)
        label_onehot[label] = 1.0
        
        # return vid name as well for compatibility
        return imgs, torch.FloatTensor(label_onehot), os.path.basename(video_path)
