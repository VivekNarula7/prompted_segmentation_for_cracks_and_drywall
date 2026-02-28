import torch

def calculate_metrics(pred_mask, true_mask):
    """Calculates IoU and Dice Score for two binary masks."""
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    intersection = torch.sum(pred_flat * true_flat)
    union = torch.sum(pred_flat) + torch.sum(true_flat) - intersection
    
    iou = intersection / (union + 1e-6)
    dice = (2. * intersection) / (torch.sum(pred_flat) + torch.sum(true_flat) + 1e-6)
    
    return iou.item(), dice.item()

