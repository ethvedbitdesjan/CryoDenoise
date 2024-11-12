import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb
from dataset import *
from loss import JointLoss
from joint import *
from collections import defaultdict

def log_images(images, predictions, targets, step, prefix="val"):
    """Helper function to log images to WandB"""
    # Take first few images from batch
    n_samples = min(4, images.shape[0])
    
    image_logs = {}
    for idx in range(n_samples):
        # Log original image
        image_logs[f"{prefix}_img_{idx}"] = wandb.Image(
            images[idx].cpu(),
            caption=f"Original Image {idx}"
        )
        
        # Log denoised image (mean prediction)
        image_logs[f"{prefix}_denoised_{idx}"] = wandb.Image(
            predictions['denoised'][idx].cpu(),
            caption=f"Denoised Image {idx}"
        )
        
        # Log detection heatmap with overlaid predictions
        fig = plt.figure(figsize=(10, 4))
        
        # Ground truth heatmap
        plt.subplot(121)
        plt.imshow(targets['heatmap'][idx].cpu(), cmap='viridis')
        plt.title("Ground Truth Heatmap")
        plt.colorbar()
        
        # Predicted heatmap
        plt.subplot(122)
        plt.imshow(predictions['detection'][idx].cpu(), cmap='viridis')
        plt.title("Predicted Heatmap")
        plt.colorbar()
        
        image_logs[f"{prefix}_heatmap_{idx}"] = wandb.Image(fig)
        plt.close(fig)
        
        # Log uncertainty map (variance prediction)
        image_logs[f"{prefix}_uncertainty_{idx}"] = wandb.Image(
            predictions['variance'][idx].cpu(),
            caption=f"Uncertainty Map {idx}"
        )
    
    wandb.log(image_logs, step=step)

def calculate_metrics(predictions, targets, padding_mask=None):
    """Calculate validation metrics"""
    metrics = {}
    
    # Denoising metrics
    denoised = predictions['denoised']
    orig = targets['clean'] if 'clean' in targets else targets['image']
    
    if padding_mask is not None:
        denoised = denoised * padding_mask
        orig = orig * padding_mask
    
    # PSNR, Not verified TODO
    mse = F.mse_loss(denoised, orig, reduction='mean')
    metrics['psnr'] = -10 * torch.log10(mse)
    
    # # SSIM
    # metrics['ssim'] = structural_similarity(
    #     denoised.cpu().numpy(), 
    #     orig.cpu().numpy(),
    #     data_range=1.0
    # )
    
    # Detection metrics
    pred_heatmap = predictions['detection']
    true_heatmap = targets['heatmap']
    
    if padding_mask is not None:
        pred_heatmap = pred_heatmap * padding_mask
        true_heatmap = true_heatmap * padding_mask
    
    # Convert heatmaps to binary predictions using threshold
    threshold = 0.5
    pred_binary = (pred_heatmap > threshold).float()
    true_binary = (true_heatmap > threshold).float()
    
    # Calculate precision, recall, F1
    tp = torch.sum(pred_binary * true_binary)
    fp = torch.sum(pred_binary * (1 - true_binary))
    fn = torch.sum((1 - pred_binary) * true_binary)
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    metrics.update({
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    })
    
    return metrics

@torch.no_grad()
def validate(model, val_loader, criterion, device, global_step):
    """Validation loop with metrics calculation and logging"""
    model.eval()
    val_stats = defaultdict(float)
    all_metrics = defaultdict(list)
    
    # For visualization
    sample_images = None
    sample_preds = None
    sample_targets = None
    
    for batch_idx, batch in enumerate(val_loader):
        images = batch['image'].to(device)
        heatmaps = batch['heatmap'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        
        denoise_stats, detect_out, noise_est = model(images)
        
        loss, loss_stats = criterion(
            denoise_stats, detect_out, images, heatmaps, 
            padding_mask=padding_mask
        )
        
        for k, v in loss_stats.items():
            val_stats[k] += v
        
        predictions = {
            'denoised': denoise_stats[:, 0],  # mean
            'variance': torch.exp(denoise_stats[:, 1]),  # variance
            'detection': detect_out
        }
        
        targets = {
            'image': images,
            'heatmap': heatmaps
        }
        
        batch_metrics = calculate_metrics(predictions, targets, padding_mask)
        for k, v in batch_metrics.items():
            all_metrics[k].append(v)
        
        # Store first batch for visualization
        if batch_idx == 0:
            sample_images = images
            sample_preds = predictions
            sample_targets = targets
    
    # Average metrics
    avg_metrics = {k: sum(v)/len(v) for k, v in all_metrics.items()}
    avg_loss = {k: v/len(val_loader) for k, v in val_stats.items()}
    
    wandb.log({
        "val_loss": avg_loss['total_loss'],
        "val_denoising_loss": avg_loss['denoising_loss'],
        "val_detection_loss": avg_loss['detection_loss'],
        "val_psnr": avg_metrics['psnr'],
        "val_ssim": avg_metrics['ssim'],
        "val_precision": avg_metrics['precision'],
        "val_recall": avg_metrics['recall'],
        "val_f1": avg_metrics['f1']
    }, step=global_step)
    
    log_images(
        sample_images, 
        sample_preds,
        sample_targets, 
        step=global_step
    )
    
    return avg_metrics

def train(model, train_loader, val_loader, num_epochs, device):
    wandb.init(project="cryo-em-joint", config={
        "learning_rate": 1e-4,
        "denoising_weight": 0.75,
        "consistency_weight": 0.1,
        "batch_size": train_loader.batch_size,
        "epochs": num_epochs
    })
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    criterion = JointLoss()
    
    best_val_f1 = 0
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_stats = defaultdict(float)
        
        for batch_idx, batch in enumerate(train_loader):
            imgs = batch['image'].to(device)
            heatmaps = batch['heatmap'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            denoise_stats, detect_out, noise_est = model(imgs)
            
            loss, loss_stats = criterion(
                denoise_stats, noise_est, detect_out, imgs, heatmaps,
                padding_mask=padding_mask
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update stats
            for k, v in loss_stats.items():
                epoch_stats[k] += v
            
            global_step += 1
            
            if global_step % 100 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    **loss_stats
                }, step=global_step)
        val_metrics = validate(model, val_loader, criterion, device, global_step)
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_val_f1,
            }, 'best_model.pth')
            
            wandb.log({
                "best_val_f1": best_val_f1
            }, step=global_step)
        scheduler.step()
        
        avg_stats = {k: v/len(train_loader) for k, v in epoch_stats.items()}
        wandb.log({
            "epoch": epoch,
            **avg_stats
        })
        
    wandb.finish()

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Cryo-EM Joint Training")
    parser.add_argument("--data_dir_maps", type=str, default="data for maps")
    parser.add_argument("--data_dir_particles", type=str, default="data for particles")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    # Load datasets
    train_dataset = CryoEMDataset(os.path.join(args.data_dir_maps, "train"), args.data_dir_particles)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_padding
    )
    
    val_dataset = CryoEMDataset(os.path.join(args.data_dir_maps, "val"), args.data_dir_particles)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_padding
    )
    
    # Initialize model
    model = JointNetwork()
    model.to(args.device)
    
    # Start training
    train(model, train_loader, val_loader, args.num_epochs, args.device)
    