import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import Generator, VAE
import math

# FORCE CPU (important for deployment)
device = torch.device("cpu")

st.set_page_config(page_title="AI Generator", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>🚀 AI Image Generator</h1>
<p style='text-align:center;'>GAN vs VAE Comparison</p>
""", unsafe_allow_html=True)

# ---------- SAFE MODEL LOAD ----------
@st.cache_resource
def load_models():
    try:
        G = Generator().to(device)
        G.load_state_dict(torch.load("gan_model.pth", map_location=device))
        G.eval()

        vae = VAE().to(device)
        vae.load_state_dict(torch.load("vae_model.pth", map_location=device))
        vae.eval()

        return G, vae, True
    except:
        return None, None, False

G, vae, model_loaded = load_models()

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Controls")
num_images = st.sidebar.slider("Number of Images", 4, 16, 8)
mode = st.sidebar.selectbox("Mode", ["GAN", "VAE", "Compare"])

# ---------- MAIN ----------
if not model_loaded:
    st.error("⚠️ Model files not found or too large for cloud")
    st.info("👉 Please ensure .pth files are uploaded properly")
else:
    if st.button("✨ Generate Images"):

        with st.spinner("Generating..."):

            cols = int(math.sqrt(num_images))
            rows = math.ceil(num_images / cols)

            if mode == "GAN":
                z = torch.randn(num_images, 100).to(device)
                images = G(z).detach()

                fig, axes = plt.subplots(rows, cols, figsize=(5,5))
                for i, ax in enumerate(axes.flatten()):
                    if i < num_images:
                        ax.imshow(images[i][0], cmap='gray')
                    ax.axis('off')

                st.subheader("🧠 GAN Output")
                st.pyplot(fig)

            elif mode == "VAE":
                z = torch.randn(num_images, 20).to(device)
                images = vae.decode(z).detach().view(-1,1,28,28)

                fig, axes = plt.subplots(rows, cols, figsize=(5,5))
                for i, ax in enumerate(axes.flatten()):
                    if i < num_images:
                        ax.imshow(images[i][0], cmap='gray')
                    ax.axis('off')

                st.subheader("🔵 VAE Output")
                st.pyplot(fig)

            else:
                col1, col2 = st.columns(2)

                z1 = torch.randn(num_images, 100).to(device)
                gan_images = G(z1).detach()

                z2 = torch.randn(num_images, 20).to(device)
                vae_images = vae.decode(z2).detach().view(-1,1,28,28)

                with col1:
                    st.subheader("🧠 GAN")
                    fig1, axes1 = plt.subplots(rows, cols, figsize=(4,4))
                    for i, ax in enumerate(axes1.flatten()):
                        if i < num_images:
                            ax.imshow(gan_images[i][0], cmap='gray')
                        ax.axis('off')
                    st.pyplot(fig1)

                with col2:
                    st.subheader("🔵 VAE")
                    fig2, axes2 = plt.subplots(rows, cols, figsize=(4,4))
                    for i, ax in enumerate(axes2.flatten()):
                        if i < num_images:
                            ax.imshow(vae_images[i][0], cmap='gray')
                        ax.axis('off')
                    st.pyplot(fig2)

            st.success("✅ Done!")