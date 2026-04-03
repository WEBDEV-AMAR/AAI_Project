import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import Generator, VAE
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="AI Generator", layout="wide")

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center; color:#ff4b4b;'>🚀 Advanced AI Image Generator</h1>
<p style='text-align:center;'>GAN vs VAE | Generative AI Project</p>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Controls")

num_images = st.sidebar.slider("Number of Images", 4, 25, 16)
model_choice = st.sidebar.selectbox("Select Model", ["GAN", "VAE", "Compare"])

st.sidebar.markdown("---")
st.sidebar.info("Built using GAN & VAE models trained on MNIST dataset")

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    G = Generator().to(device)
    G.load_state_dict(torch.load("gan_model.pth", map_location=device))
    G.eval()

    vae = VAE().to(device)
    vae.load_state_dict(torch.load("vae_model.pth", map_location=device))
    vae.eval()

    return G, vae

G, vae = load_models()

# ---------- GENERATE BUTTON ----------
if st.button("✨ Generate Images"):

    with st.spinner("Generating images... please wait"):
        
        cols = int(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(6,6))

        if model_choice == "GAN":
            z = torch.randn(num_images, 100).to(device)
            images = G(z).cpu().detach()

            for i, ax in enumerate(axes.flatten()):
                if i < num_images:
                    ax.imshow(images[i][0], cmap='gray')
                ax.axis('off')

            st.subheader("🧠 GAN Generated Images")
            st.pyplot(fig)

        elif model_choice == "VAE":
            z = torch.randn(num_images, 20).to(device)
            images = vae.decode(z).cpu().detach().view(-1,1,28,28)

            for i, ax in enumerate(axes.flatten()):
                if i < num_images:
                    ax.imshow(images[i][0], cmap='gray')
                ax.axis('off')

            st.subheader("🔵 VAE Generated Images")
            st.pyplot(fig)

        else:
            col1, col2 = st.columns(2)

            z1 = torch.randn(num_images, 100).to(device)
            gan_images = G(z1).cpu().detach()

            z2 = torch.randn(num_images, 20).to(device)
            vae_images = vae.decode(z2).cpu().detach().view(-1,1,28,28)

            with col1:
                st.subheader("🧠 GAN Output")
                fig1, axes1 = plt.subplots(rows, cols, figsize=(5,5))
                for i, ax in enumerate(axes1.flatten()):
                    if i < num_images:
                        ax.imshow(gan_images[i][0], cmap='gray')
                    ax.axis('off')
                st.pyplot(fig1)

            with col2:
                st.subheader("🔵 VAE Output")
                fig2, axes2 = plt.subplots(rows, cols, figsize=(5,5))
                for i, ax in enumerate(axes2.flatten()):
                    if i < num_images:
                        ax.imshow(vae_images[i][0], cmap='gray')
                    ax.axis('off')
                st.pyplot(fig2)

        st.success("✅ Done!")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
### 📌 Key Insights:
- **GAN** produces sharper and realistic images  
- **VAE** produces smoother and structured images  
- Both models learn from latent space representation
""")