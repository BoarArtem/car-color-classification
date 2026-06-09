import streamlit as st
import torch
from model.model import device, model
from torchvision import transforms
from PIL import Image

classes = ["Black", "Blue", "Green", "Grey", "Orange", "Red", "Silver", "White", "Yellow"]

@st.cache_resource
def load_model():
    model.load_state_dict(
        torch.load("model/car_color_model.pth", map_location=device)
    )
    model.eval()
    return model

inference_model = load_model()
inference_model = inference_model.to(device)

def image_preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

st.title("Car color classification")
st.write("Upload an image to predict your car color")

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner('Analyzing...'):
            tensor = image_preprocess(image)
            tensor = tensor.to(device)
            with torch.no_grad():
                output = inference_model(tensor)
                pred = output.argmax(dim=1).item()

            st.success(f"Prediction: **{classes[pred]}**")