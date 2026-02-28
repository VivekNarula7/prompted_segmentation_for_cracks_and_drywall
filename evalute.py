import argparse
import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from peft import PeftModel
import numpy as np
import cv2
from tqdm import tqdm
from data.dataloader import test_dataset, val_dataset, taping_train, taping_val
from utils.metrics import calculate_metrics

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device being used : {DEVICE}")
BASE_MODEL_ID = "CIDAS/clipseg-rd64-refined"
LORA_ADAPTER_PATH = "data/joint_lora_clipseg"
FULL_FT_MODEL_PATH = "data/fine_tuned_clipseg_cracks"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

def clean_binary_mask(mask_tensor, kernel_size=3):
    mask_np = mask_tensor.numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    refined = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
    return torch.tensor(refined, dtype=torch.float32)

def run_evaluation_sweep(model, processor, dataset, model_name):
    print(f"\n>>> Starting Threshold Sweep: {model_name}")
    model.eval()
    
    best_overall_iou = 0.0
    best_summary = {}

    all_logits, all_targets, all_prompts = [], [], []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Inference ({model_name})"):
            img, true_mask, prompt = dataset[i]
            inputs = processor(text=[prompt], images=[img], padding="max_length", return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            
            raw_pred = outputs.logits.view(1, 1, outputs.logits.shape[-2], outputs.logits.shape[-1])
            pred_resized = F.interpolate(raw_pred, size=true_mask.shape, mode='bilinear', align_corners=False).squeeze()
            
            all_logits.append(torch.sigmoid(pred_resized).cpu())
            all_targets.append(torch.tensor(true_mask / 255.0, dtype=torch.float32))
            all_prompts.append(prompt)

    for thresh in THRESHOLDS:
        current_ious, current_dices = [], []
        prompt_metrics = {}

        for i in range(len(all_logits)):
            binary_pred = (all_logits[i] > thresh).float()
            clean_pred = clean_binary_mask(binary_pred)
            iou, dice = calculate_metrics(clean_pred, all_targets[i])
            
            current_ious.append(iou)
            current_dices.append(dice)
            
            p = all_prompts[i]
            if p not in prompt_metrics:
                prompt_metrics[p] = {"iou": [], "dice": []}
            prompt_metrics[p]["iou"].append(iou)
            prompt_metrics[p]["dice"].append(dice)

        mIoU = np.mean(current_ious)
        mDice = np.mean(current_dices)
        print(f"  Thresh {thresh} -> mIoU: {mIoU:.4f} | Dice: {mDice:.4f}")

        if mIoU > best_overall_iou:
            best_overall_iou = mIoU
            best_summary = {
                "threshold": thresh, "iou": mIoU, "dice": mDice,
                "prompt_wise": {p: {"iou": np.mean(m["iou"]), "dice": np.mean(m["dice"])} for p, m in prompt_metrics.items()}
            }

    print(f"\n--- BEST {model_name} RESULTS (Thresh: {best_summary['threshold']}) ---")
    print(f"Overall mIoU: {best_summary['iou']:.4f} | Overall Dice: {best_summary['dice']:.4f}")
    for p, m in best_summary['prompt_wise'].items():
        print(f"  > '{p}': mIoU: {m['iou']:.4f} | Dice: {m['dice']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLIPSeg variants on Drywall/Cracks datasets.")
    parser.add_argument("--model", type=str, choices=["base", "lora", "full", "all"], default="all", 
                        help="Which model version to evaluate.")
    parser.add_argument("--dataset", type=str, choices=["cracks_test", "cracks_val", "taping_val", "taping_train"], 
                        default="cracks_test", help="Which dataset split to evaluate on.")
    
    args = parser.parse_args()

    dataset_map = {
        "cracks_test": test_dataset,
        "cracks_val": val_dataset,
        "taping_val": taping_val,
        "taping_train": taping_train
    }
    selected_dataset = dataset_map[args.dataset]
    
    processor = CLIPSegProcessor.from_pretrained(BASE_MODEL_ID)

    if args.model in ["base", "all"]:
        model = CLIPSegForImageSegmentation.from_pretrained(BASE_MODEL_ID).to(DEVICE)
        run_evaluation_sweep(model, processor, selected_dataset, "ZERO-SHOT BASELINE")
        del model

    if args.model in ["lora", "all"]:
        base = CLIPSegForImageSegmentation.from_pretrained(BASE_MODEL_ID)
        model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH).to(DEVICE)
        run_evaluation_sweep(model, processor, selected_dataset, "JOINT LoRA (PEFT)")
        del model, base

    if args.model in ["full", "all"]:
        try:
            model = CLIPSegForImageSegmentation.from_pretrained(FULL_FT_MODEL_PATH).to(DEVICE)
            run_evaluation_sweep(model, processor, selected_dataset, "FULL FINE-TUNING")
            del model
        except Exception as e:
            print(f"Could not load Full FT model: {e}")