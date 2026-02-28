from data.dataloader import test_dataset,val_dataset
import torch
from models.model import processor, model
from utils.metrics import calculate_metrics
import torch.nn.functional as F
import numpy as np

total_iou = 0.0
total_dice = 0.0
num_images = len(test_dataset)

print(f"\nEvaluating on {num_images} test images...")

for i in range(num_images):
    img, true_mask, prompt = test_dataset[i]
    
    inputs = processor(text=[prompt], images=[img], padding="max_length", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    raw_pred = outputs.logits.view(1, 1, outputs.logits.shape[-2], outputs.logits.shape[-1])
    pred_resized = F.interpolate(raw_pred, size=true_mask.shape, mode='bilinear', align_corners=False)
    pred_resized = pred_resized.squeeze() # Remove the dummy dimensions back to (Height, Width)
    binary_pred = (torch.sigmoid(pred_resized) > 0.5).float()
    true_mask_tensor = torch.tensor(true_mask/255.0, dtype = torch.float32)
    iou, dice = calculate_metrics(binary_pred, true_mask_tensor)
    total_iou += iou
    total_dice += dice
    
    print(f"Image {i+1}/{num_images} | Prompt: '{prompt}' | IoU: {iou:.4f} | Dice: {dice:.4f}")

mIoU = total_iou / num_images
mean_dice = total_dice / num_images

print("\n--- ZERO-SHOT BASELINE RESULTS ---")
print(f"mIoU: {mIoU:.4f}")
print(f"Mean Dice: {mean_dice:.4f}")