import streamlit as st
import numpy as np
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.inference import InferenceHelper

# --- Page Configuration ---
st.set_page_config(
    page_title="Diffusion Model - FashionMNIST",
    page_icon="🎨",
    layout="wide"
)

# --- Initialize Model (cached) ---
@st.cache_resource
def load_model():
    """Load the inference helper once and cache it."""
    return InferenceHelper()

# --- Main App ---
st.title("🎨 Modèle de Diffusion - FashionMNIST")
st.markdown("---")

# Load model
with st.spinner("Chargement du modèle..."):
    helper = load_model()

# --- Sidebar for Tool Selection ---
st.sidebar.title("🛠️ Outils")
tool = st.sidebar.radio(
    "Sélectionnez un outil:",
    ["Génération d'images", "Reconstruction d'images"],
    index=0
)

# ============================================
# TOOL 1: Image Generation
# ============================================
if tool == "Génération d'images":
    st.header("🖼️ Génération d'images")
    st.markdown("Générez des images de vêtements FashionMNIST en utilisant le modèle de diffusion conditionnel.")
    
    # --- Parameters ---
    col1, col2 = st.columns(2)
    
    with col1:
        # Class selection
        class_name = st.selectbox(
            "Classe à générer:",
            helper.class_list,
            index=7,  # Default to "Sneaker"
            help="Sélectionnez le type de vêtement à générer"
        )
        
        # Number of images (grid size)
        num_images = st.select_slider(
            "Nombre d'images:",
            options=[1, 4, 9, 16, 25],
            value=9,
            help="Nombre total d'images à générer"
        )
    
    with col2:
        # Number of steps
        n_steps = st.slider(
            "Nombre d'étapes (DDIM):",
            min_value=10,
            max_value=300,
            value=50,
            step=10,
            help="Plus d'étapes = meilleure qualité mais plus lent"
        )
        
        # Guidance scale (w parameter)
        w = st.slider(
            "Échelle de guidage (w):",
            min_value=0.0,
            max_value=50.0,
            value=3.0,
            step=0.5,
            help="Contrôle l'intensité du conditionnement. w=0: pas de guidage, w élevé: guidage fort"
        )
    
    # Calculate grid dimensions
    grid_size = int(np.sqrt(num_images))
    
    # --- Generate Button ---
    st.markdown("---")
    
    if st.button("🚀 Générer", type="primary", use_container_width=True):
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress = int((current / total) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Génération en cours... Étape {current}/{total}")
        
        with st.spinner(f"Génération de {num_images} images de '{class_name}'..."):
            # Run inference
            images = helper.run_inference_ddim(
                class_name=class_name,
                s=w,
                n_steps=n_steps,
                num_row=grid_size,
                num_col=grid_size,
                return_images=True,
                progress_callback=update_progress
            )
            
            progress_bar.progress(100)
            status_text.text("Génération terminée!")
        
        # Display results
        st.success(f"✅ {num_images} images générées avec succès!")
        
        # Display images in a grid (smaller size)
        st.subheader(f"Résultats: {class_name}")
        
        # Use more columns with padding to make images smaller
        num_display_cols = min(grid_size, 5)
        rows = (num_images + num_display_cols - 1) // num_display_cols
        
        # Add padding columns on sides to center and reduce image size
        for row in range(rows):
            _, *img_cols, _ = st.columns([1] + [1] * num_display_cols + [1])
            for col_idx, col in enumerate(img_cols):
                img_idx = row * num_display_cols + col_idx
                if img_idx < num_images:
                    with col:
                        st.image(
                            images[img_idx],
                            caption=f"Image {img_idx+1}",
                            width=200,
                            clamp=True
                        )
        
        # Parameters summary
        with st.expander("📊 Paramètres utilisés"):
            st.json({
                "Classe": class_name,
                "Nombre d'images": num_images,
                "Étapes DDIM": n_steps,
                "Échelle de guidage (w)": w
            })

# ============================================
# TOOL 2: Image Reconstruction (Inpainting)
# ============================================
elif tool == "Reconstruction d'images":
    st.header("🔧 Reconstruction d'images (Inpainting)")
    st.markdown("""
    Utilisez **Diffusion Posterior Sampling (DPS)** pour reconstruire des zones masquées d'une image.
    Le modèle va compléter les parties manquantes en se basant sur le contexte visible et la classe de l'image.
    """)
    
    # Load dataset (cached)
    @st.cache_resource
    def load_dataset():
        return helper.load_dataset()
    
    dataset = load_dataset()
    
    # --- Image Selection ---
    st.subheader("1️⃣ Sélection de l'image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Selection mode
        selection_mode = st.radio(
            "Mode de sélection:",
            ["Par classe", "Par index"],
            horizontal=True
        )
    
    if selection_mode == "Par classe":
        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox(
                "Classe:",
                helper.class_list,
                index=7,  # Default to "Sneaker"
                key="inpaint_class"
            )
        
        # Get sample indices for the selected class
        @st.cache_data
        def get_class_indices(class_name):
            label_idx = list(helper.label_to_name_map.values()).index(class_name)
            indices = []
            for i in range(len(dataset)):
                _, label = dataset[i]
                if label == label_idx:
                    indices.append(i)
                if len(indices) >= 100:  # Limit to 100 samples for performance
                    break
            return indices
        
        class_indices = get_class_indices(selected_class)
        
        with col2:
            sample_idx_in_class = st.slider(
                f"Échantillon ({len(class_indices)} disponibles):",
                0, min(len(class_indices)-1, 99), 0,
                key="sample_slider"
            )
            image_index = class_indices[sample_idx_in_class]
    else:
        image_index = st.number_input(
            "Index de l'image (0-59999):",
            min_value=0,
            max_value=len(dataset)-1,
            value=0,
            step=1
        )
    
    # Get the selected image
    image_np, class_name, image_tensor = helper.get_sample_image(dataset, image_index)
    
    # --- Mask Configuration ---
    st.subheader("2️⃣ Configuration du masque")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mask_type = st.selectbox(
            "Type de masque:",
            ["center", "top", "bottom", "left", "right", "custom", "random"],
            format_func=lambda x: {
                "center": "🎯 Centre",
                "top": "⬆️ Haut",
                "bottom": "⬇️ Bas", 
                "left": "⬅️ Gauche",
                "right": "➡️ Droite",
                "custom": "🕹️ Carré déplaçable",
                "random": "🎲 Aléatoire"
            }[x]
        )
    
    with col2:
        if mask_type != "random":
            mask_size = st.slider(
                "Taille du masque:",
                min_value=4,
                max_value=20,
                value=10,
                help="Taille de la zone masquée en pixels"
            )
        else:
            mask_size = 8  # Not used for random
    
    # Custom mask position controls
    pos_x, pos_y = None, None
    if mask_type == "custom":
        st.markdown("**Position du masque:**")
        col_x, col_y = st.columns(2)
        with col_x:
            pos_x = st.slider(
                "Position X (gauche → droite):",
                min_value=0,
                max_value=32 - mask_size,
                value=(32 - mask_size) // 2,
                key="pos_x"
            )
        with col_y:
            pos_y = st.slider(
                "Position Y (haut → bas):",
                min_value=0,
                max_value=32 - mask_size,
                value=(32 - mask_size) // 2,
                key="pos_y"
            )
    
    # Create mask
    import torch
    mask = helper.create_mask(mask_type, mask_size, pos_x, pos_y)
    mask_np = mask.squeeze().numpy()
    
    # Create masked image for visualization
    masked_image_np = image_np * mask_np
    
    # Display original, mask, and masked image
    st.subheader("3️⃣ Aperçu")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Image originale**")
        st.image(image_np, width=150, clamp=True)
        st.caption(f"Classe: {class_name}")
    
    with col2:
        st.markdown("**Masque**")
        st.image(mask_np, width=150, clamp=True)
        st.caption("Blanc=visible, Noir=masqué")
    
    with col3:
        st.markdown("**Image masquée**")
        st.image(masked_image_np, width=150, clamp=True)
        st.caption("Entrée du modèle")
    
    # --- DPS Parameters ---
    st.subheader("4️⃣ Paramètres DPS")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        w_dps = st.slider(
            "Échelle de guidage (w):",
            min_value=0.0,
            max_value=50.0,
            value=3.0,
            step=0.5,
            key="w_dps",
            help="Contrôle l'intensité du conditionnement"
        )
    
    with col2:
        zeta = st.slider(
            "Zeta (force DPS):",
            min_value=0.1,
            max_value=2.0,
            value=0.4,
            step=0.1,
            help="Contrôle la force du guidage vers la mesure"
        )
    
    with col3:
        steps_dps = st.slider(
            "Nombre d'étapes:",
            min_value=20,
            max_value=200,
            value=100,
            step=10,
            key="steps_dps",
            help="Plus d'étapes = meilleure qualité mais plus lent"
        )
    
    # --- Reconstruct Button ---
    st.markdown("---")
    
    if st.button("🔄 Reconstruire", type="primary", use_container_width=True):
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress = int((current / total) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Reconstruction en cours... Étape {current}/{total}")
        
        # Prepare tensors
        measurement = image_tensor.unsqueeze(0) * mask  # (1, 1, 32, 32)
        
        with st.spinner(f"Reconstruction de l'image ({class_name})..."):
            reconstruction = helper.run_inference_dps(
                class_name=class_name,
                s=w_dps,
                measurement=measurement,
                mask=mask,
                zeta=zeta,
                steps=steps_dps,
                return_images=True,
                progress_callback=update_progress
            )
            
            progress_bar.progress(100)
            status_text.text("Reconstruction terminée!")
        
        # Display results
        st.success("✅ Reconstruction terminée!")
        
        st.subheader("5️⃣ Résultats")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Original**")
            st.image(image_np, width=150, clamp=True)
        
        with col2:
            st.markdown("**Masquée**")
            st.image(masked_image_np, width=150, clamp=True)
        
        with col3:
            st.markdown("**Reconstruction**")
            st.image(reconstruction, width=150, clamp=True)
        
        with col4:
            st.markdown("**Comparaison**")
            # Show difference
            diff = np.abs(image_np - reconstruction)
            st.image(diff, width=150, clamp=True)
            st.caption("Différence")
        
        # Parameters summary
        with st.expander("📊 Paramètres utilisés"):
            st.json({
                "Classe": class_name,
                "Index image": int(image_index),
                "Type de masque": mask_type,
                "Taille du masque": mask_size,
                "Échelle de guidage (w)": w_dps,
                "Zeta": zeta,
                "Étapes DPS": steps_dps
            })

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Modèle de diffusion conditionnel entraîné sur FashionMNIST"
    "</div>",
    unsafe_allow_html=True
)
