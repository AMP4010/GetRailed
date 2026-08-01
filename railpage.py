#railpage.py
import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# ==========================================
# PAGE CONFIGURATION (Must be first)
# ==========================================
st.set_page_config(
    page_title="Railway Component Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ==========================================
# MODEL SETUP & CACHING
# ==========================================
class YOLOClsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            return out[1]
        return out


@st.cache_resource
def load_model():
    """Loads the model once and caches it in memory"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO('bestrailer.pt')
    
    wrapped_model = YOLOClsWrapper(yolo.model).to(device)
    wrapped_model.train()  # Keep in train mode for gradients
    
    # Target the second to last layer as per your original code
    target_layers = [yolo.model.model[-2]]
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    
    return yolo, cam, device


yolo_model, cam, device = load_model()

# ==========================================
# UI & LOGIC
# ==========================================
st.title("🛤️ Railway Component Analyzer")
st.markdown("Upload an image of a railway component to classify it and see exactly what the AI is looking at.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Read and preprocess the image
    image = Image.open(uploaded_file).convert('RGB')
    rgb_img = np.array(image)
    rgb_img_resized = cv2.resize(rgb_img, (224, 224))
    rgb_img_float = np.float32(rgb_img_resized) / 255.0
    
    # UI: Analyze Button
    if st.button("🔍 Analyze Image", use_container_width=True):
        with st.spinner("Analyzing and generating heatmap..."):
            # --- YOLO Classification ---
            results = yolo_model(rgb_img_resized, verbose=False)[0]
            top_class_idx = results.probs.top1
            confidence = results.probs.top1conf.item()
            
            # Grab the exact raw lowercase class name from the model
            raw_class_name = results.names[top_class_idx]
            
            # --- SUPERVISOR'S "RISK LEVEL" & EXPLANATION LOGIC ---
            if raw_class_name == "allgood":
                if confidence > 0.50:
                    risk_level = "None"
                    explanation = (
                        f"The model detected no structural or fastener anomalies with high confidence ({confidence:.1%}). "
                        "The heatmap shows low activation across the rail, indicating a healthy track section."
                    )
                else:
                    risk_level = "Very Low"
                    explanation = (
                        f"The model classified this as non-defective, but confidence is moderate ({confidence:.1%}). "
                        "The heatmap displays slight surface noise or minor visual variations; routine monitoring recommended."
                    )

            elif raw_class_name == "rail":
                if confidence > 0.90:
                    risk_level = "Very High"
                    explanation = (
                        f"Critical rail track defect detected with very high confidence ({confidence:.1%}). "
                        "The heatmap highlights severe structural anomalies (e.g., major cracks, severe wear, or severe alignment faults)."
                    )
                elif confidence > 0.75:
                    risk_level = "High"
                    explanation = (
                        f"Significant rail track defect detected with high confidence ({confidence:.1%}). "
                        "The heatmap intensely focuses on visual surface irregularities or structural degradation on the rail head/web."
                    )
                elif confidence > 0.50:
                    risk_level = "Medium"
                    explanation = (
                        f"Possible track defect identified with moderate confidence ({confidence:.1%}). "
                        "The heatmap marks potential surface damage or material fatigue that warrants physical inspection."
                    )
                elif confidence > 0.30:
                    risk_level = "Low"
                    explanation = (
                        f"Minor anomaly detected with low confidence ({confidence:.1%}). "
                        "The heatmap shows weak activation, likely caused by rust, dirt, or lighting shadows rather than a critical fault."
                    )
                else:
                    risk_level = "Very Low"
                    explanation = (
                        f"Uncertain detection of track defect (confidence: {confidence:.1%}). "
                        "The heatmap displays scattered, weak activation; unlikely to be a structural defect, but logged for review."
                    )

            elif raw_class_name == "fastener":
                if confidence > 0.90:
                    risk_level = "Very High"
                    explanation = (
                        f"Severe fastener anomaly detected with very high confidence ({confidence:.1%}). "
                        "The heatmap concentrates sharply on missing, broken, or completely detached clip/bolt components."
                    )
                elif confidence > 0.75:
                    risk_level = "High"
                    explanation = (
                        f"Fastener defect detected with high confidence ({confidence:.1%}). "
                        "The heatmap highlights loose, misplaced, or damaged mounting clips along the tie/sleeper line."
                    )
                elif confidence > 0.50:
                    risk_level = "Medium"
                    explanation = (
                        f"Potential fastener defect identified with moderate confidence ({confidence:.1%}). "
                        "The heatmap indicates localized irregularities around the clip assembly; manual verification suggested."
                    )
                elif confidence > 0.30:
                    risk_level = "Low"
                    explanation = (
                        f"Minor fastener anomaly flagged with low confidence ({confidence:.1%}). "
                        "The heatmap shows weak focal points, possibly due to ballast occlusion, rust, or debris near the fastener."
                    )
                else:
                    risk_level = "Very Low"
                    explanation = (
                        f"Uncertain fastener defect signal (confidence: {confidence:.1%}). "
                        "The heatmap displays minimal focal clustering around tie points; likely a false positive due to visual noise."
                    )

            else:
                # Fallback for unexpected class names
                risk_level = "Unknown"
                explanation = f"Unrecognized class '{raw_class_name}' with confidence score of {confidence:.1%}."
            
            # --- Grad-CAM Generation ---
            input_tensor = preprocess_image(rgb_img_float, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]).to(device)
            input_tensor.requires_grad_(True)
            
            with torch.set_grad_enabled(True):
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            
            heatmap_vis = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
            
            # --- Display Results ---
            st.divider()
            
            # Classification Result
            st.success(f"**Prediction Analysis:** {explanation}")
            
            # Display the Risk Level
            st.warning(f"**Analysed Risk Level:** {risk_level}")
            
            # Image Columns
            col1, col2 = st.columns(2)
            with col1:
                st.image(rgb_img_resized, caption="Original Image", use_container_width=True)
            with col2:
                st.image(heatmap_vis, caption="AI Attention Heatmap", use_container_width=True)
            
            # --- Heatmap Legend ---
            st.markdown("""
            ### 🧠 How to read the heatmap
            This map shows exactly where the AI was looking to make its decision.
            * 🔴 **Red / Warm colors:** The most critical areas. The AI relied heavily on these pixels.
            * 🟢 **Green / Yellow:** Areas of moderate interest.
            * 🔵 **Blue / Cool colors:** Areas the AI mostly ignored (background, irrelevant details).
            """)
