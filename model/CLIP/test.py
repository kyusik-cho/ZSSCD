import torch
import clip
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device="cpu")
model_image = model.visual
model_image = model_image.to(device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
#text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    #text_features = model.encode_text(text)
    
    #logits_per_image, logits_per_text = model(image, text)
    #probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    