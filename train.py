import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from data.dataloader import train_dataset, val_dataset
from models.model import processor, model
import torch.nn.functional as F
from tqdm import tqdm

# Move the model to the GPU if you have one, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device being used is : {device}")
model.to(device)

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# We use BCEWithLogitsLoss because CLIPSeg outputs raw, un-sigmoided scores (logits)
criterion = nn.BCEWithLogitsLoss()

print(f"Ready to train on {device}!")

### Training Loop###
model.train()

epochs = 5

for epoch in range(epochs):
    print(f"--- Epoch {epoch+1}/{epochs} ---")
    epoch_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

    for images, true_masks, prompts in loop:
        
        # 0. Prep the Data
        # 'prompts' comes out of the dataloader as a tuple of strings. The processor needs a list.
        # 'images' is already a batch of images.
        inputs = processor(text=list(prompts), images=images, padding="max_length", return_tensors="pt")
        
        # Move the processed inputs to the GPU (if available)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Convert true_masks to float32, add a channel dimension, divide by 255, and move to GPU
        # Shape goes from (Batch, Height, Width) -> (Batch, 1, Height, Width)
        true_masks = true_masks.unsqueeze(1).to(device, dtype=torch.float32) / 255.0
        
        # ---------------------------------------------------------
        # YOUR TURN: The 5-Step Deep Learning Recipe
        # ---------------------------------------------------------
        
        # TODO 1: Clear the old gradients from the previous batch
        # Hint: call .zero_grad() on your optimizer
        optimizer.zero_grad()
    
        
        # TODO 2: The Forward Pass
        # Hint: pass **inputs into your model and store the result in 'outputs'
        outputs = model(**inputs)
        
        # --- Resizing Magic (I did this part for you!) ---
        # Get the raw logits and force them into 4D shape: (Batch, 1, 352, 352)
        raw_logits = outputs.logits.unsqueeze(1)
        # Resize logits UP to match your true_mask shape (Batch, 1, 640, 640)
        resized_logits = F.interpolate(raw_logits, size=(true_masks.shape[2], true_masks.shape[3]), mode='bilinear', align_corners=False)
        # -------------------------------------------------
        
        # TODO 3: Calculate the Loss
        # Hint: Pass your 'resized_logits' and 'true_masks' into your 'criterion'
        loss = criterion(resized_logits, true_masks)
        
        
        # TODO 4: The Backward Pass (Do the calculus!)
        # Hint: call .backward() on your 'loss' variable
        loss.backward()
        
        # TODO 5: Take a Step (Update the weights!)
        # Hint: call .step() on your optimizer
        optimizer.step()
        
        # ---------------------------------------------------------
        
        # Keep track of the loss so we can print it
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
            
    # Print the average loss at the end of the epoch
    print(f"Average Loss for Epoch {epoch+1}: {epoch_loss/len(train_loader):.4f}\n")

# Saving Logic
save_path = "models/fine_tuned_clipseg_cracks"
print(f"Saving fine-tuned model to {save_path}...")

model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print("Training complete and model saved!")