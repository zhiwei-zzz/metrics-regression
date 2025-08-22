import torch
import numpy as np
import os
from pathlib import Path
from torch.utils.data import DataLoader

data_root_address = '/scratch/hwlai/panoradar/new-PanoRadar-11K'



class PanoRadarDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, target_folder=[], transform=None):
        self.data_root = data_root

        # get the target folder name
        folder_names = os.listdir(data_root)
        if len (target_folder) > 0:
            self.folder_names = target_folder
        else:
            folder_names = [f for f in folder_names if 'static' in f]
            folder_names.sort()
            self.folder_names = folder_names

        self.transform = transform

        folder_paths = []

        for folder in self.folder_names:
            folder_path = os.path.join(self.data_root, folder)
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder {folder} does not exist in {self.data_root}")
            
            traj_folders = sorted(os.listdir(folder_path))
            for traj_folder in traj_folders:
                traj_folder_path = os.path.join(folder_path, traj_folder)
                
                # json_file_names = sorted((traj_folder_path / Path('obj_json')).iterdir())
                rf_npy_names = sorted((traj_folder_path / Path('rf_npy')).iterdir())
                # seg_npy_names = sorted((traj_folder_path / Path('seg_npy')).iterdir())
                lidar_npy_names = sorted((traj_folder_path / Path('lidar_npy')).iterdir())
                glass_npy_names = sorted((traj_folder_path / Path('glass_npy')).iterdir())

                for rf_npy_name, lidar_npy_name, glass_npy_name in zip(
                    rf_npy_names, lidar_npy_names, glass_npy_names
                ):
                    folder_paths.append({
                        'rf': rf_npy_name,
                        'lidar': lidar_npy_name,
                        'glass': glass_npy_name
                    })

        self.folder_paths = folder_paths

        
    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        
        rf_npy_name = self.folder_paths[idx]['rf']
        lidar_npy_name = self.folder_paths[idx]['lidar']
        glass_npy_name = self.folder_paths[idx]['glass']

        rf_data = np.load(rf_npy_name)      #  [1, 256, 64, 512]  1 x azimuth x elevation x range
        lidar_data = np.load(lidar_npy_name).squeeze().transpose(1, 0)     # [64, 512]
        glass_data = np.load(glass_npy_name).squeeze().transpose(1, 0)     # [64, 512]

        return rf_data, lidar_data, glass_data
    

dataset = PanoRadarDataset(data_root_address)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

rf_data, lidar_data, glass_data = next(iter(dataloader))     # [64, 256, 64, 512]; [64, 512, 64]; [64, 512, 64]