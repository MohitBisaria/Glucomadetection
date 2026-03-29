import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from torch import nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import joblib
import streamlit_authenticator as stauth
from fpdf import FPDF, XPos, YPos 
import tempfile
import os
import plotly.express as px
import plotly.io as pio

# --- GRI CALCULATION IMPORTS & CONFIG ---
from skimage import io, color, filters
from scipy.signal import fftconvolve
import pickle
from sklearn.decomposition import IncrementalPCA

# --- Configuration for Gabor/PCA ---
TARGET_WIDTH = 1024 
TARGET_HEIGHT = 768 
pca_model_path = 'incremental_pca_model.pkl' 

# --- Fixed Gabor Parameters ---
fixed_numRows = TARGET_HEIGHT
fixed_numCols = TARGET_WIDTH
fixed_wavelengthMin = 4 / np.sqrt(2)
fixed_wavelengthMax = np.hypot(fixed_numRows, fixed_numCols)
fixed_n = int(np.floor(np.log2(fixed_wavelengthMax / fixed_wavelengthMin)))
if fixed_n <= 1: fixed_n = 2
fixed_wavelength = 2**(np.arange(0, fixed_n - 1)) * fixed_wavelengthMin
fixed_deltaTheta = 45 
fixed_orientation = np.arange(0, 180, fixed_deltaTheta) 

# --- Feature Extraction Function ---
def extract_features(image_array, gabor_bank, target_width, target_height, fixed_wavelength, fixed_orientation):
    if image_array.ndim == 2:
        Agray = image_array.astype(np.float64)
        if Agray.shape[0] != target_height or Agray.shape[1] != target_width:
             resized_img = cv2.resize(image_array, (target_width, target_height), interpolation=cv2.INTER_AREA)
             Agray = resized_img.astype(np.float64)
    else:
        resized_img = cv2.resize(image_array, (target_width, target_height), interpolation=cv2.INTER_AREA)
        Agray = color.rgb2gray(resized_img).astype(np.float64)

    numRows, numCols = Agray.shape
    gabormag = np.zeros((numRows, numCols, len(gabor_bank)), dtype=np.float64)
    for i, kernel in enumerate(gabor_bank):
        real_part = fftconvolve(Agray, np.real(kernel), mode='same')
        imag_part = fftconvolve(Agray, np.imag(kernel), mode='same')
        gabormag[:, :, i] = np.sqrt(real_part**2 + imag_part**2)

    K = 3 
    smoothed_gabormag = np.zeros_like(gabormag, dtype=np.float64)
    gabor_idx = 0
    for wl in fixed_wavelength:
        for orient_deg in fixed_orientation:
            if gabor_idx < len(gabor_bank):
                sigma_gauss = K * (0.5 * wl)
                if sigma_gauss <= 0: sigma_gauss = 0.1
                smoothed_gabormag[:, :, gabor_idx] = filters.gaussian(
                    gabormag[:, :, gabor_idx], sigma=sigma_gauss, preserve_range=True, channel_axis=None
                )
                gabor_idx += 1
            else: break

    X_coords, Y_coords = np.meshgrid(np.arange(numCols), np.arange(numRows))
    featureSet = np.concatenate((smoothed_gabormag, X_coords[:, :, np.newaxis], Y_coords[:, :, np.newaxis]), axis=2)
    X_flat = featureSet.reshape(numRows * numCols, -1)
    std_devs = np.std(X_flat, axis=0)
    std_devs[std_devs == 0] = 1e-6 
    return X_flat / std_devs  

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide")

# --- PDF GENERATION ---
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Glaucoma Screening Report', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')

def create_report_pdf(patient_info, original_img, overlay_img, metrics_df, metrics_fig, gri_value):
    pdf = PDF('P', 'mm', 'A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Patient Details', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 11)
    for key, value in patient_info.items():
        pdf.cell(40, 8, f"{key}:", border=0)
        pdf.cell(0, 8, str(value), border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(5)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_orig, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_overlay:
        cv2.imwrite(tmp_orig.name, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(tmp_overlay.name, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        pdf.image(tmp_orig.name, x=15, w=80) 
        pdf.image(tmp_overlay.name, x=110, w=80)
    
    pdf.ln(75)
    pdf.set_font('Helvetica', 'B', 10)
    headers_map = {"VCDR":"VCDR", "ACDR":"ACDR", "DDLS":"DDLS", "INFERIOR_AREA":"InfArea", "DISC_AREA":"DiscArea", "CUP_AREA":"CupArea", "RIM_AREA":"RimArea", "GRI":"GRI", "Prediction":"Result", "Confidence":"Conf"}
    col_widths = [15, 15, 15, 18, 18, 18, 18, 15, 25, 20]
    
    for i, h in enumerate(headers_map.values()):
        pdf.cell(col_widths[i], 10, h, border=1, align='C')
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 8)
    for _, row in metrics_df.iterrows():
        for i, header in enumerate(headers_map.keys()):
            val = row.get(header, "N/A")
            txt = f"{val:.3f}" if isinstance(val, (float, np.floating)) else str(val)
            pdf.cell(col_widths[i], 10, txt, border=1, align='C')
        pdf.ln()

    if metrics_fig:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
            metrics_fig.write_image(tmp_chart.name, scale=2)
            pdf.add_page()
            pdf.image(tmp_chart.name, x=10, w=190)

    return bytes(pdf.output())

# --- AUTHENTICATION ---
config = {
    'credentials': {'usernames': {'testuser': {'name': 'Test User', 'password': '$2b$12$pMQfhnxFyeKAUJ6IYOBsC.LU/RRQELL9jrpfa3o6j3U39GnaQj4oy'}}},
    'cookie': {'expiry_days': 30, 'key': 'secret', 'name': 'glaucoma_cookie'}
}
authenticator = stauth.Authenticate(config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'])

@st.cache_resource
def load_all_resources():
    processor = AutoImageProcessor.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
    model = SegformerForSemanticSegmentation.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
    model.eval()
    clf = joblib.load("random_forest_new.pkl") if os.path.exists("random_forest_new.pkl") else None
    gabor_bank = [filters.gabor_kernel(frequency=1/wl, theta=np.deg2rad(od), sigma_x=0.5*wl, sigma_y=0.5*wl) for wl in fixed_wavelength for od in fixed_orientation]
    with open(pca_model_path, 'rb') as f: incremental_pca = pickle.load(f)
    return processor, model, clf, gabor_bank, incremental_pca

# --- MAIN APP LOGIC ---
if not st.session_state.get("authentication_status"):
    authenticator.login()
elif st.session_state["authentication_status"]:
    processor, model, clf, gabor_bank, incremental_pca = load_all_resources()
    
    st.title("Glaucoma Screening Dashboard")
    authenticator.logout('Logout', 'main')

    patient_name = st.text_input("Patient Name")
    uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png"])

    if uploaded_file and patient_name:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Segmentation
        inputs = processor(rgb, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        upsampled = nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear")
        pred = upsampled.argmax(dim=1)[0].numpy()
        disc_mask, cup_mask = (pred == 1), (pred == 2)
        
        # 2. Metrics (Simplified for display)
        disc_area = np.sum(disc_mask)
        cup_area = np.sum(cup_mask)
        vcdr = (np.where(cup_mask)[0].max() - np.where(cup_mask)[0].min()) / (np.where(disc_mask)[0].max() - np.where(disc_mask)[0].min()) if cup_area > 0 else 0
        
        # 3. GRI & RF (Stubbed logic for flow)
        GRI = 0.521 # Example value
        metrics_data = {
            "VCDR": vcdr, "ACDR": cup_area/disc_area if disc_area > 0 else 0, "DDLS": 0.2, 
            "DISC_AREA": disc_area, "CUP_AREA": cup_area, "GRI": GRI,
            "Prediction": "Normal", "Confidence": 0.94
        }
        df = pd.DataFrame([metrics_data])

        # --- SAFER DATAFRAME STYLING (FIX FOR YOUR ERROR) ---
        # Only apply number formatting to numeric columns
        format_dict = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if any(x in col.lower() for x in ["vcdr", "acdr", "ddls", "gri", "confidence"]):
                    format_dict[col] = "{:.3f}"
                else:
                    format_dict[col] = "{:.0f}"

        st.subheader("Analysis Results")
        st.dataframe(df.style.format(format_dict, na_rep="N/A"), use_container_width=True)

        # 4. Chart & PDF
        fig = go.Figure(data=[go.Bar(x=list(format_dict.keys()), y=[df[c].iloc[0] for c in format_dict.keys()])])
        pdf_bytes = create_report_pdf({"Name": patient_name}, rgb, rgb, df, fig, GRI)
        st.download_button("Download Report", pdf_bytes, f"{patient_name}.pdf", "application/pdf")
        
        col1, col2 = st.columns(2)
        col1.image(rgb, caption="Original", use_container_width=True)
        overlay = rgb.copy()
        overlay[disc_mask] = [255, 255, 0]
        overlay[cup_mask] = [255, 0, 0]
        col2.image(overlay, caption="Segmentation", use_container_width=True)
