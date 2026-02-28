import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from peft import LoraConfig, get_peft_model
import csv
import os
from data.dataloader import DrywallCrackDataset, DrywallTapingDataset

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, smooth=1e-6):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        dice_score = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_score
        return bce_loss + dice_loss
    
print("Loading datasets...")
cracks_train = DrywallCrackDataset(image_dir="cracks/train", mask_dir="cracks_mask/train")
drywall_train = DrywallTapingDataset(image_dir="Drywall-Join-Detect/train", mask_dir="drywall_join_detect_mask/train")

joint_train_dataset = ConcatDataset([cracks_train, drywall_train])
train_loader = DataLoader(dataset=joint_train_dataset, batch_size=4, shuffle=True)

print(f"Total Joint Training Images: {len(joint_train_dataset)}")

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

lora_config = LoraConfig(
    r=16,                  
    lora_alpha=32,         
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device being used is : {device}")
model.to(device)

epochs = 40
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
criterion = BCEDiceLoss()
print(f"Ready to train on {device}!")

log_filename = "logfiles/joint_lora_training_log.csv"
with open(log_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Average_Loss","Learning_Rate"])

print(f"Ready to train on {device}! Logging to {log_filename}")

### Training Loop ###
model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

    for images, true_masks, prompts in loop:
        
        inputs = processor(text=list(prompts), images=images, padding="max_length", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        true_masks = true_masks.unsqueeze(1).to(device, dtype=torch.float32) / 255.0
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        
        raw_logits = outputs.logits.unsqueeze(1)
        resized_logits = F.interpolate(raw_logits, size=(true_masks.shape[2], true_masks.shape[3]), mode='bilinear', align_corners=False)
        
        loss = criterion(resized_logits, true_masks)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_epoch_loss = epoch_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Average Loss for Epoch {epoch+1}: {avg_epoch_loss:.4f}\n")
    with open(log_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.writerow([epoch + 1, avg_epoch_loss, current_lr])

    scheduler.step()

save_path = "models/joint_lora_clipseg"
print(f"Saving fine-tuned model to {save_path}...")
model.save_pretrained(save_path)
processor.save_pretrained(save_path)
print("Training complete and model saved!")