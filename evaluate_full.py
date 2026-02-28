from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from data.dataloader import test_dataset # Focused on the Cracks test split
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# --- METRIC UTILS ---
def calculate_metrics(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return iou.item(), dice.item()

# --- MODEL LOADING ---
saved_model_path = "models/fine_tuned_clipseg_cracks" 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading FULLY FINE-TUNED model from {saved_model_path}...")
processor = CLIPSegProcessor.from_pretrained(saved_model_path)
model = CLIPSegForImageSegmentation.from_pretrained(saved_model_path)
model.to(device).eval()

dataset = test_dataset
threshold = 0.3

all_ious = []
all_dices = []
prompt_metrics = {}

print(f"\nEvaluating Full Fine-Tuning on {len(dataset)} images...")

for i in tqdm(range(len(dataset))):
    img, true_mask, prompt = dataset[i]
    
    inputs = processor(text=[prompt], images=[img], padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    raw_pred = outputs.logits.view(1, 1, outputs.logits.shape[-2], outputs.logits.shape[-1])
    pred_resized = F.interpolate(raw_pred, size=true_mask.shape, mode='bilinear', align_corners=False).squeeze()
    
    binary_pred = (torch.sigmoid(pred_resized) > threshold).float().cpu()
    true_mask_tensor = torch.tensor(true_mask / 255.0, dtype=torch.float32)
    
    iou, dice = calculate_metrics(binary_pred, true_mask_tensor)
    
    all_ious.append(iou)
    all_dices.append(dice)
    
    if prompt not in prompt_metrics:
        prompt_metrics[prompt] = {"ious": [], "dices": []}
    prompt_metrics[prompt]["ious"].append(iou)
    prompt_metrics[prompt]["dices"].append(dice)

# --- FINAL RESULTS ---
print("\n" + "="*30)
print("FULL FINE-TUNING RESULTS (CRACKS)")
print("="*30)
print(f"Overall mIoU: {np.mean(all_ious):.4f}")
print(f"Overall Dice: {np.mean(all_dices):.4f}")

print("\nPrompt-wise Breakdown:")
for prompt, metrics in prompt_metrics.items():
    print(f"  > '{prompt}': mIoU: {np.mean(metrics['ious']):.4f} | Dice: {np.mean(metrics['dices']):.4f}")