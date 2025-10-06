import pandas as pd
import numpy as np
import streamlit as st
#import matplotlib.pyplot as plt
#import seaborn as sns
#from plotnine import ggplot, aes, geom_tile, geom_text, scale_fill_gradient, theme_minimal, theme, element_text, geom_col, element_blank
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

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









st.subheader("Prédiction : prédire la probabilité d'avoir un cancer")

# je retire la cible
X = df[["Age", "BMI", "Smoking", "GeneticRisk", "PhysicalActivity", "AlcoholIntake"]]
y = df["Diagnosis"]

#split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.markdown(f"Précision du modèle (test) : {accuracy:.3f}")

# --- SECTION INTERACTIVE : PREDICTION PERSONNALISÉE ---
st.markdown("### Simulation")

# Curseurs
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Âge", 18, 80, 40)
    bmi = st.slider("IMC (BMI)", 10.0, 40.0, 22.0)
with col2:
    smoking = st.selectbox("Fumeur", ["Non", "Oui"])
    genetic = st.selectbox("Risque génétique", ["Faible", "Moyen", "Élevé"])
with col3:
    physical = st.slider("Activité physique (0 à 10)", 0, 10, 5)
    alcohol = st.slider("Consommation d'alcool (0 à 10)", 0, 10, 3)


smoking_val = 1 if smoking == "Oui" else 0
genetic_map = {"Faible": 0, "Moyen": 1, "Élevé": 2}
genetic_val = genetic_map[genetic]

# Création du DataFrame d’un seul individu
X_new = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "Smoking": [smoking_val],
    "GeneticRisk": [genetic_val],
    "PhysicalActivity": [physical],
    "AlcoholIntake": [alcohol]
})

# Prédiction de la probabilité de cancer
proba_cancer = model.predict_proba(X_new)[0][1]

# Affichage de la probabilité
st.markdown("---")
st.markdown("### Résultat de la prédiction :")

col1, col2 = st.columns([2, 3])
with col1:
    st.metric(
        label="Probabilité estimée d'avoir un cancer",
        value=f"{100 * proba_cancer:.1f} %",
        help="Calculée via la régression logistique"
    )

# Graphique visuel de la probabilité
fig_proba = px.bar(
    x=["Pas de cancer", "Cancer"],
    y=[1 - proba_cancer, proba_cancer],
    color=["Pas de cancer", "Cancer"],
    color_discrete_map={"Pas de cancer": "#4B56EB", "Cancer": "#E63718"},
    text=[f"{(1 - proba_cancer) * 100:.1f} %", f"{proba_cancer * 100:.1f} %"]
)
fig_proba.update_traces(textposition="outside", textfont=dict(size=16, color="white"))
fig_proba.update_layout(
    title="Probabilité estimée par le modèle",
    yaxis_title="Probabilité",
    xaxis_title="Diagnostic",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    showlegend=False
)
with col2:
    st.plotly_chart(fig_proba, use_container_width=True)

# PERF MODÈLE (ROC, GINI, CM) ---
with st.expander("Performances du modèle (ROC, Gini, matrice de confusion)"):

    y_proba_test = model.predict_proba(X_test)[:, 1]

    #  AUC et Gini
    auc = roc_auc_score(y_test, y_proba_test)
    gini = 2 * auc - 1

    # courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    roc_fig = px.line(
        roc_df, x="FPR", y="TPR",
        title=f"Courbe ROC (AUC = {auc:.3f}, Gini = {gini:.3f})"
    )
    roc_fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aléatoire", line=dict(dash="dash"))
    roc_fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_title="Taux de faux positifs (FPR)",
        yaxis_title="Taux de vrais positifs (TPR)"
    )

    # matrice confusion
    cm = confusion_matrix(y_test, model.predict(X_test))
    cm_df = pd.DataFrame(cm, index=["Vrai : 0 (Pas de cancer)", "Vrai : 1 (Cancer)"],
                         columns=["Prédit : 0", "Prédit : 1"])

    cm_fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Reds",
        labels=dict(x="Prédiction", y="Valeur réelle", color="Effectif"),
    )
    cm_fig.update_layout(
        title="Matrice de confusion",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    # Affichage
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(roc_fig, use_container_width=True)
    with col2:
        st.plotly_chart(cm_fig, use_container_width=True)

    st.markdown(f"*AUC :* {auc:.3f}  *Gini :* {gini:.3f}")

    # =====================================================
    # 🔎 CONTRIBUTION DES FACTEURS (importance des variables)
    # =====================================================
    with st.expander("Contribution des facteurs (importance des variables)"):
        # Récupération des coefficients du modèle
        coef = model.coef_[0]
        variables = X.columns

        coef_df = pd.DataFrame({
            "Variable": variables,
            "Coefficient": coef,
            "Contribution": np.abs(coef)  # pour classer par importance absolue
        }).sort_values(by="Contribution", ascending=True)

        # Interprétation :
        # - Coefficient positif → augmente la probabilité d'avoir un cancer
        # - Coefficient négatif → diminue cette probabilité

        st.markdown("""
        Les coefficients de la régression logistique indiquent l'influence de chaque variable :
        - *Valeur positive* → augmente le risque estimé de cancer  
        - *Valeur négative* → réduit le risque estimé  
        """)

        # Graphique Plotly barres horizontales
        fig_coef = px.bar(
            coef_df,
            x="Coefficient",
            y="Variable",
            orientation="h",
            color="Coefficient",
            color_continuous_scale=["#4B56EB", "#E63718"],  # bleu -> négatif, rouge -> positif
            text="Coefficient",
            title="Contribution des facteurs au risque estimé de cancer",
        )

        fig_coef.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                               textfont=dict(size=14, color="white"))
        fig_coef.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            coloraxis_showscale=False,
            xaxis_title="Coefficient (influence sur la probabilité)",
            yaxis_title="Variable"
        )

        st.plotly_chart(fig_coef, use_container_width=True)

        # On affiche aussi les valeurs exactes
        st.dataframe(coef_df[["Variable", "Coefficient"]].sort_values(by="Coefficient", ascending=False))
