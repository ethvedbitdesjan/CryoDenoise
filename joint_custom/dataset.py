import torch
from PIL import Image
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import cv2
import random
from torch.utils.data import default_collate
import torch.nn.functional as F
import os

def load_image(path, transforms):
    img = Image.open(path)
    img = transforms(img)
    return img

def extract_segmentation_map(img, particles):
    segment_map = np.zeros(img.squeeze(0).shape, dtype=np.uint8)
    for x, y, r in particles:
        cv2.circle(segment_map, (x, y), r, 255, -1)
    
    segment_map = torch.FloatTensor(segment_map)
    #convert to grayscale
    segment_map = segment_map.unsqueeze(0)
    #TODO: convert to binary
    return segment_map

def extract_gaussian_map(img, particles):
    img_shape = img.shape[-2:]
    heatmap = np.zeros(img_shape)
    for x, y, radius in particles:
        #heatmap += create_gaussian_heatmap(img_shape, (y, x), radius)
        heatmap += create_binary_heatmap(img_shape, (y, x), radius)
    heatmap = np.clip(heatmap, 0, 1)  # Normalize overlapping regions
    return torch.FloatTensor(heatmap).unsqueeze(0) #add channel dimension

def create_binary_heatmap(shape, center, radius):
    """Create binary 0/1 targets instead of Gaussian"""
    heatmap = np.zeros(shape)
    y, x = np.ogrid[:shape[0], :shape[1]]
    center_y, center_x = center
    r2 = (x - center_x)**2 + (y - center_y)**2 + 1e-6
    heatmap[r2 <= radius**2] = 1.0
    return heatmap

def create_gaussian_heatmap(shape, center, radius, sigma=None):
    """
    """
    if sigma is None:
        sigma = radius / 3  # Rule of thumb for sigma for paper code

    y, x = np.ogrid[:shape[0], :shape[1]]
    center_y, center_x = center
    r2 = (x - center_x)**2 + (y - center_y)**2
    heatmap = np.exp(-r2 / (2 * sigma**2))
    return heatmap

class CryoEMDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, particle_coords_path=None, consistency_prob=0):
        self.image_paths = []
        for file in os.listdir(image_path):
            self.image_paths.append(os.path.join(image_path, file))
            assert file.endswith('.jpg') or file.endswith('.tif'), 'Only .jpg and .tif files are supported'
        
        self.particle_coords = {}
        for file in os.listdir(particle_coords_path):
            img_name = file.split('.')[0]
            particles = []
            with open(os.path.join(particle_coords_path, file), 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    x, y, diameter = map(int, line.strip().split(','))
                    particles.append((x, y, diameter//2)) #store radius
            self.particle_coords[img_name] = particles
        
        num_imgs_filtered = 0
        new_image_paths = []
        for img_path in self.image_paths:
            file_name = img_path.split('/')[-1].split('.')[0]
            if file_name in self.particle_coords:
                new_image_paths.append(img_path)
            else:
                num_imgs_filtered += 1
        self.image_paths = new_image_paths
        print(f'Filtered {num_imgs_filtered} images without particle coordinates')
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ])
        self.consistency_prob = consistency_prob
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load image
        image_filepath = self.image_paths[idx]
        img = load_image(image_filepath, self.transforms)
        
        
        file_name = image_filepath.split('/')[-1].split('.')[0]
        if self.particle_coords is not None:
            particles = self.particle_coords[file_name]
            seg_map = extract_gaussian_map(img, particles)
        else:
            raise ValueError('Particle coordinates must be provided')
        
        if np.random.rand() < self.consistency_prob:
            if random.random() < 0.5:
                img = torch.flip(img, [-1])
                seg_map = torch.flip(seg_map, [-1])
            else:
                k = random.randint(1, 3)  # 90, 180 or 270 degrees
                img = torch.rot90(img, k, [-2, -1])
                seg_map = torch.rot90(seg_map, k, [-2, -1])
        return {
            'image': img,
            'segmentation_map': seg_map,
            'coords': particles
        }

def collate_fn(batch):
    max_h = max([item['image'].shape[-2] for item in batch])
    max_w = max([item['image'].shape[-1] for item in batch])
    
    padded_batch = []
    for item in batch:
        img = item['image']
        seg_map = item['segmentation_map']
        
        # Calculate padding
        pad_h = max_h - img.shape[-2]
        pad_w = max_w - img.shape[-1]
        
        # Pad image and segmentation map
        img_padded = F.pad(img, (0, pad_w, 0, pad_h), value=-1)
        seg_map_padded = F.pad(seg_map, (0, pad_w, 0, pad_h), value=0)
        
        padded_batch.append({
            'image': img_padded,
            'segmentation_map': seg_map_padded,
            'coords': item['coords']
        })
    
    return default_collate(padded_batch)

def collate_with_padding(batch):
    max_h = max(x['image'].shape[-2] for x in batch)
    max_w = max(x['image'].shape[-1] for x in batch)
    
    # ensure 2^5 divisible dimensions for UNet where 5 down/up sampling occurs
    if max_h % 32:
        max_h = max_h + (32 - max_h % 32)
    if max_w % 32:
        max_w = max_w + (32 - max_w % 32)
    
    
    padded_batch = []
    padding_masks = []
    
    for item in batch:
        # Create padding mask
        mask = torch.ones_like(item['image'])
        
        # Calculate padding
        pad_h = max_h - item['image'].shape[-2]
        pad_w = max_w - item['image'].shape[-1]
        
        if pad_h > 0 or pad_w > 0:
            # Pad image and heatmap
            padded_img = F.pad(item['image'], (0, pad_w, 0, pad_h), value=-1)
            padded_heatmap = F.pad(item['segmentation_map'], (0, pad_w, 0, pad_h), value=0)
            # Pad mask
            mask = F.pad(mask, (0, pad_w, 0, pad_h), value=0)
        else:
            padded_img = item['image']
            padded_heatmap = item['segmentation_map']
            
        padded_batch.append({
            'image': padded_img,
            'heatmap': padded_heatmap,
            'coords': item['coords']
        })
        padding_masks.append(mask)

    all_imgs = [x['image'] for x in padded_batch]
    all_heatmaps = [x['heatmap'] for x in padded_batch]
    all_imgs = torch.stack(all_imgs)
    all_heatmaps = torch.stack(all_heatmaps)
    padding_masks = torch.stack(padding_masks)
    data_dict = {'image': all_imgs, 'heatmap': all_heatmaps, 'padding_mask': padding_masks, 'coords': [x['coords'] for x in padded_batch]}
    return data_dict