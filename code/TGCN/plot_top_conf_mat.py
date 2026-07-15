import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import glob

from configs import Config
from vsl_sign_dataset import Sign_Dataset
from vsl_tgcn_model import GCN_muti_att
from train_utils import validation

def plot_top_conf_mat(gts, preds, classes, top_k=15, save_to='output/top15-conf-mat.png'):
    # Tính ma trận nhầm lẫn đầy đủ
    cm = confusion_matrix(gts, preds)
    
    # Tìm các class bị nhầm lẫn nhiều nhất (số lượng sai khác dọc theo hàng)
    errors_per_class = cm.sum(axis=1) - np.diag(cm)
    
    # Lấy top_k class có số lỗi cao nhất
    top_k_indices = np.argsort(errors_per_class)[::-1][:top_k]
    
    top_k_classes = [classes[i] for i in top_k_indices]
    
    # Rút trích ma trận nhỏ top_k x top_k
    sub_cm = cm[np.ix_(top_k_indices, top_k_indices)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(sub_cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=top_k_classes, yticklabels=top_k_classes)
    plt.title(f'Top {top_k} Most Confused Classes (Requires Fine-tuning)', fontsize=14)
    plt.ylabel('True label (Thực tế)', fontsize=12)
    plt.xlabel('Predicted label (Mô hình đoán)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs('output', exist_ok=True)
    plt.savefig(save_to, dpi=300)
    print(f"Saved readable confusion matrix to {save_to}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, 'configs', 'vsl_472.ini')
    configs = Config(config_file)
    
    if len(sys.argv) > 1:
        best_model_path = sys.argv[1]
    else:
        # 1. Tìm file weights .pth mới nhất trong thư mục
        ckpt_dir = os.path.join('checkpoints', 'vsl472')
        pt_files = glob.glob(os.path.join(ckpt_dir, '*.pth'))
        if not pt_files:
            print("Không tìm thấy file .pth nào! Vui lòng truyền đường dẫn trực tiếp: python plot.py <đường_dẫn>")
            exit()
            
        best_model_path = max(pt_files, key=os.path.getctime)
        
    print(f"Loading weights from: {best_model_path}")
    
    # 2. Khởi tạo dataset
    pose_data_root = '/kaggle/input/datasets/nguyenanfms/vsl-vietnamese-sign-language-v2/processed_augmented/processed_augmented/keypoints_splited'
    val_dataset = Sign_Dataset(root_dir=pose_data_root, split='test', num_samples=configs.num_samples)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size, shuffle=False)
    
    # 3. Khởi tạo model
    model = GCN_muti_att(input_feature=configs.num_samples*3, hidden_feature=configs.num_samples*3,
                         num_class=472, p_dropout=configs.drop_p, num_stage=configs.num_stages).cuda()
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # 4. Chạy validation để lấy kết quả dự đoán
    print("Running evaluation to get predictions...")
    _, _, val_gts, val_preds, _ = validation(model, val_data_loader, epoch=0, save_to=None)
    
    # 5. Vẽ ma trận rút gọn top 15
    # Cần fake label encoder giống như vsl_sign_dataset
    class DummyEncoder: pass
    label_encoder = DummyEncoder()
    label_encoder.classes_ = val_dataset.classes
    
    plot_top_conf_mat(val_gts, val_preds, label_encoder.classes_, top_k=15)
