import pandas as pd
import numpy as np
import streamlit as st
#import matplotlib.pyplot as plt
#import seaborn as sns
#from plotnine import ggplot, aes, geom_tile, geom_text, scale_fill_gradient, theme_minimal, theme, element_text, geom_col, element_blank
import plotly.express as px

# Importation de la base
df = pd.read_csv("data/The_Cancer_data_1500_V2.csv",  sep=",")

df["Sexe_label"] = df["Gender"].map({0: "Homme", 1: "Femme"}) 
df["Tabac_label"] = df["Smoking"].map({0: "Non", 1: "Oui"}) 
df["Genetic_label"] = df["GeneticRisk"].map({0: "Faible", 1: "Moyen", 2: "Élevé"})
df["Diagnosis_label"] = df["Diagnosis"].map({0: "Pas de cancer", 1: "Cancer"})
conditions = [(df["Age"]<30),
              (df["Age"]>=30)&(df["Age"]<40),
              (df["Age"]>=40)&(df["Age"]<50),
              (df["Age"]>=50)&(df["Age"]<60),
              (df["Age"]>=60)&(df["Age"]<70),
              (df["Age"]>=70)&(df["Age"]<=80)]

choix = ["moins de 30 ans", "30 ans - 39 ans", "40 ans - 49 ans","50 ans - 59 ans","60 ans - 69 ans","70 ans - 80 ans"]
df["classe_age"] = np.select(conditions, choix, default="plus de 80 ans")

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

# graphs
st.subheader("Profil démographique")
#########################################################

ct_cancer = pd.crosstab(df_filtered["classe_age"], df_filtered["Diagnosis"])
ct_cancer = ct_cancer[[1]].reset_index()
ct_cancer.columns = ["classe_age", "Cancer"]

heatmap_fig = px.imshow(ct_cancer[["Cancer"]].T, # valeurs
labels=dict(x="Classe d'âge", y="Diagnostic", color="Nombre de cas"),
x=ct_cancer["classe_age"],y=["Cancer"],color_continuous_scale="Reds",text_auto=True)

heatmap_fig.update_traces(
textfont=dict(size=18) # chiffres plus gros
)

heatmap_fig.update_layout(
title="Répartition des cas de cancer par classe d'âge",
xaxis_title="Classe d'âge",
yaxis_title="",
font=dict(color="white"),
plot_bgcolor="rgba(0,0,0,0)",
paper_bgcolor="rgba(0,0,0,0)"
)

###############################################################

summary = df_filtered.groupby(["Sexe_label", "Diagnosis_label"]).size().reset_index(name="Count")
summary["Diagnostic"] = summary["Diagnosis_label"]

barplot_groupe_fig = px.bar(summary,x="Sexe_label",y="Count",
color="Diagnostic", # barre par couleur selon le diagnostic
barmode="group", # barres côte à côte (groupées)
text="Count", # chiffres sur les barres
color_discrete_map={"Pas de cancer": "#4B56EB", "Cancer": "#E63718"} # optionnel : couleurs custom
)

barplot_groupe_fig.update_traces(
textfont=dict(size=18) # chiffres plus gros
)

barplot_groupe_fig.update_layout(
title="Nombre de cas par Sexe",
yaxis=dict(showgrid=True, gridcolor="gray", gridwidth=1),
xaxis_title="Sexe",
yaxis_title="Nombre de cas",
plot_bgcolor="rgba(0,0,0,0)",
paper_bgcolor="rgba(0,0,0,0)",
font=dict(color="white")
)


col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(heatmap_fig, use_container_width=True)

with col2:
    st.plotly_chart(barplot_groupe_fig, use_container_width=True)






st.subheader("Facteurs du Mode de Vie")
#######################################
moyennes = df_filtered.groupby("Diagnosis")[["PhysicalActivity", "AlcoholIntake"]].mean().reset_index()
stds = df_filtered.groupby("Diagnosis")[["PhysicalActivity", "AlcoholIntake"]].std().reset_index()

moyennes_long = moyennes.melt(id_vars="Diagnosis", var_name="Variable", value_name="Moyenne")
stds_long = stds.melt(id_vars="Diagnosis", var_name="Variable", value_name="SD")

moyennes_long["SD"] = stds_long["SD"]

moyennes_long["Diagnosis"] = moyennes_long["Diagnosis"].astype(str)

barplot_groupe_fig_2 = px.bar(moyennes_long,x="Diagnosis",y="Moyenne",color="Variable",barmode="group",text="Moyenne",error_y="SD",
color_discrete_map={"PhysicalActivity": "#8A91F5", "AlcoholIntake": "#EF553B"},
labels={"Diagnosis": "Diagnostic", "Moyenne": "Moyenne"})

barplot_groupe_fig_2.update_traces(texttemplate="%{text:.2f}",textposition="outside",textfont=dict(size=14, color="white"))

barplot_groupe_fig_2.update_layout(
title="Moyenne activité physique et alcool selon diagnostic",
yaxis=dict(showgrid=True, gridcolor="gray", gridwidth=1),
xaxis=dict(tickvals=["0", "1"], ticktext=["Pas de cancer (0)", "Cancer (1)"]),
yaxis_title="Moyenne",
plot_bgcolor="rgba(0,0,0,0)",
paper_bgcolor="rgba(0,0,0,0)",
font=dict(color="white")
)

####################################################
smokers = df_filtered[df_filtered["Smoking"] == 1] # 1 = fumeur
smokers_summary = smokers["Diagnosis"].value_counts().reset_index()
smokers_summary.columns = ["Diagnosis", "Count"]
smokers_summary["Diagnosis"] = smokers_summary["Diagnosis"].map({0: "Pas de Cancer", 1: "Cancer"})

fig_smokers = px.pie(smokers_summary,values="Count",names="Diagnosis",color="Diagnosis",
color_discrete_map={"Cancer": "#9C2C28", "Pas de Cancer": "#E2E2EB"},
hole=0.5 # pour un donut si tu veux
)
fig_smokers.update_traces(textinfo="percent+label", textfont_size=16)

fig_smokers.update_layout(
title="Fumeurs",
plot_bgcolor="rgba(0,0,0,0)",
paper_bgcolor="rgba(0,0,0,0)",
font=dict(color="white")
)

######################################################
nonsmokers = df_filtered[df_filtered["Smoking"] == 0] # 0 = non-fumeur
nonsmokers_summary = nonsmokers["Diagnosis"].value_counts().reset_index()
nonsmokers_summary.columns = ["Diagnosis", "Count"]
nonsmokers_summary["Diagnosis"] = nonsmokers_summary["Diagnosis"].map({0: "Pas de Cancer", 1: "Cancer"})

fig_nonsmokers = px.pie(nonsmokers_summary,values="Count",names="Diagnosis",color="Diagnosis",
color_discrete_map={"Cancer": "#9C2C28", "Pas de Cancer": "#E2E2EB"},
hole=0.5
)
fig_nonsmokers.update_traces(textinfo="percent+label", textfont_size=16)
fig_nonsmokers.update_layout(
title="Non-fumeurs",
plot_bgcolor="rgba(0,0,0,0)",
paper_bgcolor="rgba(0,0,0,0)",
font=dict(color="white")
)

###################################################
col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(barplot_groupe_fig_2, use_container_width=True)

with col2:
    st.plotly_chart(fig_smokers, use_container_width=True)

with col3:
    st.plotly_chart(fig_nonsmokers, use_container_width=True)




st.subheader("Facteurs Médicaux")
