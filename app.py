import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import os

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================
# Custom CSS
# ============================================
st.markdown("""
    <style>
    .big-font {
        font-size:48px !important;
        font-weight:bold;
    }
    .result-cat {
        background-color: #ffe6e6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .result-dog {
        background-color: #e6f2ff;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .accuracy {
        font-size:24px;
        font-weight:bold;
        color: #2ecc71;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# Title
# ============================================
st.markdown('<p class="big-font">🐱 Cat vs Dog Classifier 🐶</p>', unsafe_allow_html=True)
st.markdown("**Upload an image to predict if it's a cat or dog!**")
st.markdown("---")

# ============================================
# Load Model (cached)
# ============================================
@st.cache_resource
def load_model():
    """Load trained ResNet50 model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load ResNet50
        model = resnet50(pretrained=True)
        
        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace last layer (2 classes: cat, dog)
        model.fc = nn.Linear(2048, 2)
        
        # Load trained weights
        if os.path.exists('cat_dog_model.pth'):
            model.load_state_dict(torch.load('cat_dog_model.pth', map_location=device))
            st.success("✅ Loaded trained model (94% accuracy)")
        else:
            st.warning("⚠️ Trained model not found. Using pre-trained weights.")
            st.info("To save your model: torch.save(model.state_dict(), 'cat_dog_model.pth')")
        
        model = model.to(device)
        model.eval()
        
        return model, device
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# ============================================
# Transform (Same as training: 128x128, ToTensor, NO normalization)
# ============================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ============================================
# Prediction Function
# ============================================
def predict_image(image, model, device):
    """Make prediction on single image"""
    try:
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        class_names = ['🐱 Cat', '🐶 Dog']
        return class_names[predicted_class], confidence
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# ============================================
# Main App
# ============================================

# Load model
model, device = load_model()

if model is None:
    st.error("❌ Failed to load model. Please check the setup.")
    st.stop()

# Create upload section
st.subheader("📸 Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif", "bmp"])

# Process uploaded image
if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Your Image", use_column_width=True)
    
    with col2:
        st.write("")
        st.write("")
        
        # Make prediction
        predicted_class, confidence = predict_image(image, model, device)
        
        if predicted_class and confidence:
            # Show prediction
            if "Cat" in predicted_class:
                st.markdown(f'<div class="result-cat"><p class="accuracy">{predicted_class}</p></div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-dog"><p class="accuracy">{predicted_class}</p></div>', 
                           unsafe_allow_html=True)
            
            # Show confidence
            st.metric("Confidence", f"{confidence:.2%}")
            
            # Progress bar
            st.progress(confidence)
            
            # Detailed confidence
            st.write("")
            st.write("**Confidence Breakdown:**")
            col_cat, col_dog = st.columns(2)
            with col_cat:
                st.write(f"🐱 Cat: {(1-confidence if 'Dog' in predicted_class else confidence):.2%}")
            with col_dog:
                st.write(f"🐶 Dog: {(confidence if 'Dog' in predicted_class else 1-confidence):.2%}")

else:
    st.info("👆 Upload an image to get started!")
    
    # Show example
    with st.expander("ℹ️ How to use"):
        st.write("""
        1. **Click the upload button** above
        2. **Select a cat or dog image** from your computer
        3. **Wait** for the prediction (takes ~1-2 seconds)
        4. **View results** showing the predicted class and confidence
        
        **Supported formats:** JPG, JPEG, PNG, GIF, BMP
        
        **Image size:** Any size works (automatically resized to 128×128)
        """)

# ============================================
# Test with Sample Images
# ============================================
st.markdown("---")
st.subheader("📁 Test with Sample Images from Your Dataset")

col1, col2 = st.columns(2)

with col1:
    if st.button("🐱 Test Random Cat Image"):
        try:
            cat_files = os.listdir("val/cat") if os.path.exists("val/cat") else os.listdir("train")
            cat_files = [f for f in cat_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if cat_files:
                img_path = os.path.join("val/cat", cat_files[0]) if os.path.exists("val/cat") else "train/cat/" + cat_files[0]
                image = Image.open(img_path)
                
                st.image(image, caption="Sample Cat Image", use_column_width=True)
                
                predicted_class, confidence = predict_image(image, model, device)
                if predicted_class and confidence:
                    st.success(f"**Predicted:** {predicted_class}")
                    st.metric("Confidence", f"{confidence:.2%}")
            else:
                st.error("No cat images found in val/cat folder")
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    if st.button("🐶 Test Random Dog Image"):
        try:
            dog_files = os.listdir("val/dog") if os.path.exists("val/dog") else os.listdir("train")
            dog_files = [f for f in dog_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if dog_files:
                img_path = os.path.join("val/dog", dog_files[0]) if os.path.exists("val/dog") else "train/dog/" + dog_files[0]
                image = Image.open(img_path)
                
                st.image(image, caption="Sample Dog Image", use_column_width=True)
                
                predicted_class, confidence = predict_image(image, model, device)
                if predicted_class and confidence:
                    st.success(f"**Predicted:** {predicted_class}")
                    st.metric("Confidence", f"{confidence:.2%}")
            else:
                st.error("No dog images found in val/dog folder")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================
# Model Info
# ============================================
st.markdown("---")
with st.expander("ℹ️ Model Information"):
    st.write("""
    **Architecture:** ResNet50 with Transfer Learning
    
    **Pre-trained on:** ImageNet (1.2M images, 1000 classes)
    
    **Fine-tuned for:** Cat vs Dog Classification
    
    **Input Size:** 128 × 128 pixels
    
    **Training Dataset:** 275 images (138 cats, 137 dogs)
    
    **Validation Accuracy:** 94% ⭐
    
    **Training Time:** ~5-10 minutes
    
    **Device:** CPU/GPU (auto-detected)
    
    **Model Parameters:**
    - Total: 23.5M
    - Trainable: 4,098 (last layer only)
    - Frozen: 23.5M (ImageNet features)
    """)



# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🤖 Built with PyTorch & Streamlit</p>
    <p><small>Model: ResNet50 | Accuracy: 94% | Training Data: 275 images</small></p>
</div>
""", unsafe_allow_html=True)
