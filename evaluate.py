from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from data.dataloader import test_dataset, val_dataset, taping_train, taping_val
import torch
import torch.nn.functional as F
from peft import PeftModel
import numpy as np
import cv2
from tqdm import tqdm

# --- METRIC UTILS ---
def calculate_metrics(pred, target, smooth=1e-6):
    """Calculate IoU and Dice for binary tensors."""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return iou.item(), dice.item()

def clean_binary_mask(mask_tensor, kernel_size=3):
    """Applies morphological Opening and Closing to refine the mask."""
    mask_np = mask_tensor.numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Opening removes noise, Closing fills holes
    refined = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
    return torch.tensor(refined, dtype=torch.float32)

# --- MODEL LOADING ---
saved_lora_path = "models/joint_lora_clipseg"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model and LoRA adapters...")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
base_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model = PeftModel.from_pretrained(base_model, saved_lora_path)
model.to(device).eval()

dataset = taping_val
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

best_overall_iou = 0.0
best_results = {}

print(f"\nEvaluating on {len(dataset)} images across {len(thresholds)} thresholds...")

for thresh in thresholds:
    print(f"\n--- Testing Threshold: {thresh} ---")
    prompt_metrics = {} 
    all_ious = []
    all_dices = []

    for i in tqdm(range(len(dataset)), desc=f"Thresh {thresh}"):
        img, true_mask, prompt = dataset[i]
        
        inputs = processor(text=[prompt], images=[img], padding="max_length", return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        raw_pred = outputs.logits.view(1, 1, outputs.logits.shape[-2], outputs.logits.shape[-1])
        pred_resized = F.interpolate(raw_pred, size=true_mask.shape, mode='bilinear', align_corners=False).squeeze()
        
        binary_pred = (torch.sigmoid(pred_resized) > thresh).float().cpu()
        clean_pred = clean_binary_mask(binary_pred)
        
        true_mask_tensor = torch.tensor(true_mask / 255.0, dtype=torch.float32)
        
        iou, dice = calculate_metrics(clean_pred, true_mask_tensor)
        
        all_ious.append(iou)
        all_dices.append(dice)
        
        if prompt not in prompt_metrics:
            prompt_metrics[prompt] = {"ious": [], "dices": []}
        prompt_metrics[prompt]["ious"].append(iou)
        prompt_metrics[prompt]["dices"].append(dice)

    mean_iou = np.mean(all_ious)
    mean_dice = np.mean(all_dices)
    
    print(f"Overall Results -> mIoU: {mean_iou:.4f} | Mean Dice: {mean_dice:.4f}")
    
    if mean_iou > best_overall_iou:
        best_overall_iou = mean_iou
        best_results = {
            "threshold": thresh,
            "overall_iou": mean_iou,
            "overall_dice": mean_dice,
            "prompt_wise": {p: {"iou": np.mean(m["ious"]), "dice": np.mean(m["dices"])} for p, m in prompt_metrics.items()}
        }

# --- FINAL SUMMARY FOR YOUR REPORT ---
print("\n" + "="*30)
print("FINAL EVALUATION SUMMARY")
print("="*30)
print(f"Optimal Threshold: {best_results['threshold']}")
print(f"Overall mIoU:      {best_results['overall_iou']:.4f}")
print(f"Overall Mean Dice: {best_results['overall_dice']:.4f}")

print("\nPrompt-wise Breakdown:")
for prompt, metrics in best_results['prompt_wise'].items():
    print(f"  > '{prompt}':")
    print(f"    mIoU: {metrics['iou']:.4f} | Dice: {metrics['dice']:.4f}")