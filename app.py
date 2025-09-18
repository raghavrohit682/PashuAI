import gradio as gr
import torch, torch.nn.functional as F
import timm, json
from PIL import Image
import torchvision.transforms as transforms
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (224, 224)

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

if os.path.exists("breed_info.json"):
    with open("breed_info.json","r") as f:
        breed_info = json.load(f)
else:
    breed_info = {}

species_classes = ["buffaloes","cows"]
species_model = timm.create_model("resnet34", pretrained=False, num_classes=len(species_classes))
species_model.load_state_dict(torch.load("species_model.pth", map_location=DEVICE))
species_model = species_model.to(DEVICE).eval()

cow_classes = ["Gir","Sahiwal","Kankrej","Tharparkar","Hariana",
               "Red_Sindhi","Ongole","Banni","Jersey_cattle",
               "Holstein_Friesian_cattle"]
cow_model = timm.create_model("tf_efficientnetv2_m", pretrained=False, num_classes=len(cow_classes))
cow_model.load_state_dict(torch.load("cow_breed_model.pth", map_location=DEVICE))
cow_model = cow_model.to(DEVICE).eval()

buff_classes = ["Murrah","Mehsana","Jaffarabadi","Surti","Nili_Ravi","Banni"]
buff_model = timm.create_model("tf_efficientnetv2_m", pretrained=False, num_classes=len(buff_classes))
buff_model.load_state_dict(torch.load("buffalo_breed_model.pth", map_location=DEVICE))
buff_model = buff_model.to(DEVICE).eval()

def predict(image):
    img = Image.fromarray(image).convert("RGB")
    x = val_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = species_model(x); prob = F.softmax(out, dim=1)
        sp_idx = prob.argmax(1).item()
    species = species_classes[sp_idx]
    if species == "cows":
        model, classes = cow_model, cow_classes
    else:
        model, classes = buff_model, buff_classes
    with torch.no_grad():
        out = model(x); prob = F.softmax(out, dim=1)
        idx = prob.argmax(1).item(); conf = prob[0][idx].item()
    breed = classes[idx]
    info = breed_info.get(breed, {
        "lifespan": "N/A",
        "milk_yield": "N/A",
        "suitable_food": "N/A",
        "diseases": [],
        "region": "N/A",
        "traits": "N/A"
    })
    result = f"""
    🐂 **Species**: {species}  
    🔖 **Breed**: {breed} ({conf*100:.2f}%)  

    ⏳ **Lifespan**: {info.get('lifespan','N/A')}  
    🥛 **Milk Yield**: {info.get('milk_yield','N/A')}  
    🌾 **Food**: {info.get('suitable_food','N/A')}  
    🦠 **Diseases**: {", ".join(info.get('diseases',[]))}  
    📍 **Region**: {info.get('region','N/A')}  
    ✨ **Traits**: {info.get('traits','N/A')}  
    """
    return img, result

with gr.Blocks() as demo:
    gr.Markdown("# 🐄 PashuAI – Cow & Buffalo Breed Recognition")
    gr.Markdown("Upload an image to identify the species and breed, along with details like milk yield, lifespan, and traits.")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="numpy", label="Upload Animal Image")
            btn = gr.Button("Predict")
        with gr.Column():
            img_out = gr.Image(label="Uploaded Image")
            result_out = gr.Markdown()
    btn.click(fn=predict, inputs=img_in, outputs=[img_out, result_out])

if __name__ == "__main__":
    demo.launch()
