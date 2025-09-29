import pandas as pd
import numpy as np
import streamlit as st

# Importation de la base
df = pd.read_csv("data/The_Cancer_data_1500_V2.csv",  sep=",")

df["Sexe_label"] = df["Gender"].map({0: "Homme", 1: "Femme"}) 
df["Tabac_label"] = df["Smoking"].map({0: "Non", 1: "Oui"}) 
df["Genetic_label"] = df["GeneticRisk"].map({0: "Faible", 1: "Moyen", 2: "Élevé"})
df["Diagnosis_label"] = df["Diagnosis"].map({0: "Pas de cancer", 1: "Cancer"})

# configuration de la page
st.set_page_config(page_title="Cancer Prediction Dataset", page_icon="🩺", layout="wide")


# Configuration Logo
#st.logo(image="images/image_1.jpg", icon_image="images/image_1.jpg")
st.sidebar.image("images/image_1.jpg", caption="Cancer Prediction Dataset")
with st.sidebar:
    st.title("Cancer Prediction Dataset")  
    st.header("⚙️ Filtres")
    age_range = st.slider("Âge", int(df.Age.min()), int(df.Age.max()), (20, 80))
    sexe_filter = st.multiselect("Sexe", options=["Homme", "Femme"], default=["Homme", "Femme"]) 
    tabac_filter = st.multiselect("Tabagisme", options=["Non", "Oui"], default=["Non", "Oui"]) 
    genetic_filter = st.multiselect("Risque génétique", options=["Faible", "Moyen", "Élevé"], default=["Faible", "Moyen", "Élevé"])


# Application des filtres
df_filtered = df[ 
    (df["Age"].between(age_range[0], age_range[1])) & 
    (df["Sexe_label"].isin(sexe_filter)) & 
    (df["Tabac_label"].isin(tabac_filter)) & 
    (df["Genetic_label"].isin(genetic_filter)) ]
 
def Home():
    with st.expander("Table des patients"):
        apercu_data = st.multiselect('Choisissez les variables : ', df_filtered.columns, default=[])
        st.write(df_filtered[apercu_data])
    
    col1, col2, col3, col4 = st.columns(4, gap='large') 
    with col1:
        st.info("EFFECTIF")
        st.metric(label="Nombres de patients suivis" ,value=len(df_filtered))
    
    with col2:
        st.info("DIAGNOSTIC")
        st.metric(label="% Cancer" ,value=f"{100*df_filtered.Diagnosis.mean():.1f}%", help="pourcentage de patients diagnostiqués d'un cancer")
    
    with col3:
        st.info("IMC MOYEN")
        st.metric(label="indice de masse Corporel" ,value=f"{df_filtered.BMI.mean():.1f}")

    with col4:
        st.info("AGE")
        st.metric(label="âge moyen des patients" ,value=f"{df_filtered.Age.mean():.1f} ans")


Home()

st.subheader("Profil démographique")