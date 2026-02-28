from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# "CIDAS/clipseg-rd64-refined" is the standard pre-trained version of CLIPSeg
print("Loading CLIPSeg Processor and Model...")
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
print("Successfully loaded!")

