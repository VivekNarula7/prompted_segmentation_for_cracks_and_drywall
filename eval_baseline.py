from data.dataloader import test_dataset,val_dataset
import torch
from models.model import processor, model
from utils.metrics import calculate_metrics
import torch.nn.functional as F
import numpy as np

# --- THE EVALUATION LOOP ---
total_iou = 0.0
total_dice = 0.0
num_images = len(test_dataset)

print(f"\nEvaluating on {num_images} test images...")

for i in range(num_images):
    img, true_mask, prompt = test_dataset[i]
    
    # 1. Prepare inputs
    inputs = processor(text=[prompt], images=[img], padding="max_length", return_tensors="pt")
    
    # 2. Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 3. Get raw prediction (shape: 1, 352, 352)
    raw_pred = outputs.logits.view(1, 1, outputs.logits.shape[-2], outputs.logits.shape[-1])
    # 4. RESIZE prediction to match ground truth's exact shape
    # F.interpolate expects shape (Batch, Channels, Height, Width), so we add dummy dimensions
    pred_resized = F.interpolate(raw_pred, size=true_mask.shape, mode='bilinear', align_corners=False)
    pred_resized = pred_resized.squeeze() # Remove the dummy dimensions back to (Height, Width)
    
    # ---------------------------------------------------------
    # YOUR TURN: Fill in the code below!
    # ---------------------------------------------------------
    
    # TODO 5: Apply sigmoid to 'pred_resized', check if > 0.5, and convert to .float()
    binary_pred = (torch.sigmoid(pred_resized) > 0.5).float()
    
    # TODO 6: Convert 'true_mask' (NumPy array of 0, 255) to a PyTorch tensor of 0.0 and 1.0
    # Hint: Divide by 255.0 and wrap it in torch.tensor()
    true_mask_tensor = torch.tensor(true_mask/255.0, dtype = torch.float32)
    
    # ---------------------------------------------------------
    
    # 7. Calculate metrics for this specific image
    iou, dice = calculate_metrics(binary_pred, true_mask_tensor)
    total_iou += iou
    total_dice += dice
    
    print(f"Image {i+1}/{num_images} | Prompt: '{prompt}' | IoU: {iou:.4f} | Dice: {dice:.4f}")

# Calculate final averages
mIoU = total_iou / num_images
mean_dice = total_dice / num_images

print("\n--- ZERO-SHOT BASELINE RESULTS ---")
print(f"mIoU: {mIoU:.4f}")
print(f"Mean Dice: {mean_dice:.4f}")