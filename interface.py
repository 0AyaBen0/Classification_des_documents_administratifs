"""
Interface Web Streamlit
Interface utilisateur pour la classification de documents
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import plotly.express as px

# Ajouter le dossier src au path
project_root = Path(__file__).parent
src_path = project_root / "src"
# Ensure project root is first so imports like `from main import ...` resolve
sys.path.insert(0, str(project_root))
# Also add `src` so internal modules can be imported as `src.xxx`
sys.path.insert(1, str(src_path))

from main import DocumentClassifier, CLASSES
from src.utils.config_loader import load_config

# Configuration
st.set_page_config(
    page_title="Classification de Documents Administratifs",
    page_icon="📄",
    layout="wide"
)

# Titre
st.title("📄 Classification de Documents Administratifs")
st.markdown("Système intelligent de classification automatique de documents PDF")

# Sidebar
st.sidebar.header("Configuration")
device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
use_light = st.sidebar.checkbox("Modèle léger", value=False)

# Initialiser le classificateur
@st.cache_resource
def load_classifier(device, use_light):
    return DocumentClassifier(device=device, use_light_model=use_light)

classifier = load_classifier(device, use_light)

# Onglets
tab1, tab3 = st.tabs(["📤 Upload", "ℹ️ À propos"])

with tab1:
    st.header("Upload et Classification")
    
    uploaded_file = st.file_uploader(
        "Choisir un fichier PDF ou une image",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        help="Téléchargez un fichier PDF ou une image (PNG/JPG) à classifier"
    )
    
    if uploaded_file is not None:
        # Sauvegarder temporairement
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Classifier
        with st.spinner("Classification en cours..."):
            try:
                # Détecter le type de fichier
                file_ext = temp_path.suffix.lower()
                
                if file_ext == '.pdf':
                    results = classifier.classify_pdf(str(temp_path))
                    is_pdf = True
                elif file_ext in ['.png', '.jpg', '.jpeg']:
                    # Image unique
                    result = classifier.classify_image(str(temp_path))
                    results = [result]  # Convertir en liste pour compatibilité
                    is_pdf = False
                else:
                    st.error(f"Format de fichier non supporté: {file_ext}")
                    results = []
                    is_pdf = False
                
                if results:
                    num_pages = len(results)
                    st.success(f"✅ PDF classifié avec succès ({num_pages} page{'s' if num_pages > 1 else ''})")
                    
                    # Résumé global si plusieurs pages
                    if num_pages > 1:
                        st.subheader("📊 Résumé du PDF")
                        summary_data = []
                        for i, result in enumerate(results, 1):
                            if "error" not in result:
                                summary_data.append({
                                    "Page": i,
                                    "Classe": result.get("predicted_class", "N/A"),
                                    "Confiance": result.get("confidence", 0.0),
                                    "Stratégie": result.get("strategy", "N/A")
                                })
                        
                        if summary_data:
                            df_summary = pd.DataFrame(summary_data)
                            st.dataframe(df_summary, use_container_width=True)
                            
                            # Graphique des classes par page
                            fig_summary = px.bar(
                                df_summary, 
                                x="Page", 
                                y="Confiance",
                                color="Classe",
                                title="Classification par Page",
                                labels={"Confiance": "Confiance (%)", "Page": "Numéro de Page"}
                            )
                            st.plotly_chart(fig_summary, use_container_width=True)
                    
                    # Afficher les détails pour chaque page
                    st.subheader("📄 Détails par Page")
                    
                    # Créer des onglets pour chaque page
                    if num_pages > 1:
                        page_tabs = st.tabs([f"Page {i+1}" for i in range(num_pages)])
                    else:
                        page_tabs = [st.container()]
                    
                    for page_idx, (result, tab) in enumerate(zip(results, page_tabs)):
                        with tab:
                            if "error" in result:
                                st.error(f"❌ Erreur sur la page {page_idx + 1}: {result.get('error', 'Erreur inconnue')}")
                            else:
                                st.markdown(f"### Page {page_idx + 1}")
                                
                                # Métriques principales
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    predicted_class = result.get("predicted_class", "N/A")
                                    st.metric("Classe Prédite", predicted_class)
                                
                                with col2:
                                    confidence = result.get("confidence", 0.0)
                                    st.metric("Confiance", f"{confidence:.2%}")
                                
                                with col3:
                                    rejection = result.get("rejection_score", 0.0)
                                    st.metric("Score de Rejet", f"{rejection:.2%}")
                                
                                # Détails
                                st.markdown("#### Détails de la Classification")
                                
                                # Scores par méthode
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Computer Vision**")
                                    cv_info = result.get("cv", {})
                                    cv_class = cv_info.get('class', 'N/A')
                                    cv_conf = cv_info.get('confidence', 0.0)
                                    st.write(f"- Classe: {cv_class}")
                                    st.write(f"- Confiance: {cv_conf:.2%}")
                                
                                with col2:
                                    st.write("**NLP**")
                                    nlp_info = result.get("nlp", {})
                                    nlp_class = nlp_info.get('class', 'N/A')
                                    nlp_conf = nlp_info.get('confidence', 0.0)
                                    st.write(f"- Classe: {nlp_class}")
                                    st.write(f"- Confiance: {nlp_conf:.2%}")
                                
                                # Scores Gabarits
                                st.write("**Scores Gabarits**")
                                gabarits_scores = result.get("gabarits_scores", {})
                                if gabarits_scores:
                                    df_gabarits = pd.DataFrame([
                                        {"Famille": k, "Score": v}
                                        for k, v in gabarits_scores.items()
                                    ])
                                    fig = px.bar(
                                        df_gabarits, 
                                        x="Famille", 
                                        y="Score", 
                                        title=f"Scores par Famille - Page {page_idx + 1}"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Stratégie utilisée
                                strategy = result.get("strategy", "N/A")
                                st.info(f"**Stratégie de fusion:** {strategy}")
                    
                    # Télécharger le résultat JSON complet (toutes les pages)
                    result_json = json.dumps(results, indent=2, ensure_ascii=False)
                    st.download_button(
                        label=f"📥 Télécharger le résultat JSON ({num_pages} page{'s' if num_pages > 1 else ''})",
                        data=result_json,
                        file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.error("Aucun résultat obtenu")
                    
            except Exception as e:
                st.error(f"Erreur lors de la classification: {e}")
                st.exception(e)
            finally:
                # Nettoyer
                if temp_path.exists():
                    temp_path.unlink()
with tab3:
    st.header("À propos")
    st.markdown("""
    ### Système de Classification de Documents Administratifs
    
    Ce système permet de classifier automatiquement des documents PDF en 5 catégories :
    
    1. **Pièce d'identité** (CNIE – Recto et Verso)
    2. **Relevé bancaire** (Différentes Banques)
    3. **Facture d'électricité** (Différentes Régies)
    4. **Facture d'eau** (Différentes Régies)
    5. **Document employeur** (Bulletins de paie + Attestations de Travail)
    
    ### Architecture
    
    Le système utilise une approche multimodale combinant :
    - **Computer Vision** : Modèles CNN (ResNet50/EfficientNet) avec features de gabarits
    - **NLP** : OCR + Classification textuelle (motifs sémantiques, CamemBERT)
    - **Fusion Intelligente** : Combinaison des prédictions avec validation métier
    
    ### Technologies
    
    - PyTorch / TensorFlow
    - Transformers (CamemBERT)
    - Tesseract OCR
    - OpenCV
    - Streamlit
    
    ### Auteurs
    
    Équipe INDIA-S5 - assia-aya-khaoula
    """)

