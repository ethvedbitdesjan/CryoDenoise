import torch
import torch.nn as nn
import torch.nn.functional as F

class JointLoss:
    def __init__(self, denoising_weight=0.75, consistency_weight=0.1):
        self.denoising_weight = denoising_weight
        self.consistency_weight = consistency_weight
        
    def gaussian_nll_loss(self, pred_stats, std_noise, noisy_input, padding_mask=None):
        """
        Gaussian negative log likelihood loss for denoising
        """
        mu = pred_stats[:, 0:1] #mu_x=mu_y
        std_x = torch.exp(pred_stats[:, 1:2]) #predict log std for numerical stability
        std_y = std_x + torch.exp(std_noise)
        
        #print(noisy_input.shape, mu.shape, std_y.shape, pred_stats.shape, 'loss')
        
        loss = torch.log(std_y) + 0.5 * ((noisy_input - mu)**2 / std_y**2)
        if padding_mask is not None:
            loss = loss * padding_mask
            
        return loss.mean()

    def detection_loss(self, pred_heatmap, true_heatmap, padding_mask=None):
        """
        Binary cross entropy for particle detection
        """
        loss = F.binary_cross_entropy(pred_heatmap, true_heatmap, reduction='none')
        if padding_mask is not None:
            loss = loss * padding_mask
            
        return loss.mean()
        
    def consistency_loss(self, pred1, pred2):
        """
        MSE loss between predictions of transformed pairs
        """
        return F.mse_loss(pred1, pred2)
        
    def __call__(self, pred_stats, noise_est, pred_heatmap, noisy_input, true_heatmap, 
                 padding_mask=None, consistency_pairs=None):
        denoising_loss = self.gaussian_nll_loss(pred_stats, noise_est, noisy_input, padding_mask)
        detection_loss = self.detection_loss(pred_heatmap, true_heatmap, padding_mask)
        
        total_loss = self.denoising_weight * denoising_loss + \
                     (1 - self.denoising_weight) * detection_loss
                     
        if consistency_pairs is not None:
            pred1, pred2 = consistency_pairs
            total_loss += self.consistency_weight * self.consistency_loss(pred1, pred2)
            
        return total_loss, {
            'denoising_loss': denoising_loss.item(),
            'detection_loss': detection_loss.item()
        }