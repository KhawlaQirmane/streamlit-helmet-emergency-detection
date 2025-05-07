import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import glob
from datetime import datetime

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = r"C:\Users\user\Desktop\yolo_yolo\custom_yolov8_20250505_125324_yahya_final_model_final.pt"  # Ton modèle personnalisé
DATA_YAML = r"C:\Users\user\Desktop\yolo_yolo\data.yaml"         # Facultatif pour metrics

# Charger le modèle
model = YOLO(MODEL_PATH)

st.set_page_config(
    page_title="Détection Routière YOLOv8",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🚗"
)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🧠 Infos sur le modèle")
st.sidebar.markdown(f"**Modèle utilisé :** `{MODEL_PATH}`")
st.sidebar.markdown("**Version :** YOLOv8 - Ultralytics")

show_metrics = st.sidebar.checkbox("📊 Afficher les métriques (mAP)", value=False)

if show_metrics:
    with st.spinner("Évaluation du modèle..."):
        try:
            metrics = model.val(data=DATA_YAML, split="test", conf=0.5)

            mAP50 = metrics.box.map50
            precision = metrics.box.p
            recall = metrics.box.r

            st.sidebar.markdown("---")
            st.sidebar.write(f"**mAP@0.5 :** `{mAP50:.3f}`")
            st.sidebar.write(f"**Précision moyenne :** `{sum(precision)/len(precision):.3f}`")
            st.sidebar.write(f"**Recall moyenne :** `{sum(recall)/len(recall):.3f}`")
        except Exception as e:
            st.sidebar.error(f"Erreur lors de l'affichage des métriques : {e}")

# -------------------------------
# PAGE PRINCIPALE
# -------------------------------
st.title("🚦 Détection d’objets routiers (YOLOv8)")
st.write("Uploadez une **image** ou une **vidéo** et observez les objets détectés automatiquement.")

file_type = st.radio("📁 Type de fichier :", ["Image", "Vidéo"])

# -------------------------------
# IMAGE
# -------------------------------
if file_type == "Image":
    uploaded_file = st.file_uploader("Uploader une image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Image originale", use_container_width=True)

        with st.spinner("🕵️ Détection en cours..."):
            results = model.predict(image)
            results[0].save(filename="result.jpg")

        st.image("result.jpg", caption="✅ Résultat de la détection", use_container_width=True)

        with open("result.jpg", "rb") as f:
            st.download_button("⬇ Télécharger l’image détectée", f, file_name="resultat_detection.jpg")

# -------------------------------
# VIDÉO
# -------------------------------

elif file_type == "Vidéo":
    uploaded_video = st.file_uploader("Uploader une vidéo", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video:
        # Enregistrer temporairement la vidéo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(uploaded_video.read())
            video_path = temp_video_file.name

        st.video(video_path)

        # Créer un dossier unique de sortie
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predict_name = f"streamlit_{timestamp}"
        output_dir = os.path.join("runs", "predict", predict_name)

        with st.spinner("🕵️ Détection en cours sur la vidéo..."):
            results = model.predict(
                source=video_path,
                save=True,
                conf=0.5,
                project="runs/predict",
                name=predict_name,
                exist_ok=True,
                save_txt=False,
                save_conf=False,
                stream=False
            )

        # Afficher l'arborescence du répertoire de sortie
        st.write(f"Arborescence de sortie : {os.listdir(output_dir)}")

        # Chercher la vidéo générée au format .avi ou .mp4
        result_video_path = glob.glob(os.path.join(output_dir, "*.avi")) or glob.glob(os.path.join(output_dir, "*.mp4"))

        if result_video_path:
            detected_video_path = result_video_path[0]
            st.video(detected_video_path)  # Affichage de la vidéo annotée
            with open(detected_video_path, "rb") as f:
                st.download_button("⬇ Télécharger la vidéo détectée", f, file_name="video_detectee.mp4")
        else:
            st.warning("❌ La vidéo détectée n’a pas été trouvée.")

        st.success("✅ Détection terminée !")
