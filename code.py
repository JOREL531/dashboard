import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, aes, geom_histogram, labs, theme_minimal, scale_fill_brewer

# sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

# statsmodels
import statsmodels.api as sm



#st.subheader("🌸 Hello Kitty 💖")

#st.image(
#    "/Users/benallal/Desktop/hello_kitty.jpg",  # chemin vers ton image
#    caption="🌸",
#    use_column_width=False,
#    width=200
#)


st.set_page_config(page_title="Cancer par âge", page_icon="🩺")

# couleur du fond
st.markdown(
    """
    <style>
        .stApp {background-color: #ffe6f2;}
        .block-container {max-width: 900px; padding-top: 1rem; padding-bottom: 2rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
#   CHARGEMENT DONNÉES
# =========================
df = pd.read_csv(
    "data/The_Cancer_data_1500_V2.csv",
    sep=","
)

# ------------------------------ # Filtres # ------------------------------
st.sidebar.header("Filtres")
age_range = st.sidebar.slider("Âge", int(df.Age.min()), int(df.Age.max()), (20, 80)) 
sexe_filter = st.sidebar.multiselect("Sexe", options=["Homme", "Femme"], default=["Homme", "Femme"]) 
tabac_filter = st.sidebar.multiselect("Tabagisme", options=["Non", "Oui"], default=["Non", "Oui"]) 
genetic_filter = st.sidebar.multiselect("Risque génétique", options=["Faible", "Moyen", "Élevé"], default=["Faible", "Moyen", "Élevé"])

# Transformation des labels 
df["Sexe_label"] = df["Gender"].map({0: "Homme", 1: "Femme"}) 
df["Tabac_label"] = df["Smoking"].map({0: "Non", 1: "Oui"}) 
df["Genetic_label"] = df["GeneticRisk"].map({0: "Faible", 1: "Moyen", 2: "Élevé"})

# Application des filtres
df_filtered = df[ 
    (df["Age"].between(age_range[0], age_range[1])) & 
    (df["Sexe_label"].isin(sexe_filter)) & 
    (df["Tabac_label"].isin(tabac_filter)) & 
    (df["Genetic_label"].isin(genetic_filter)) ]


# ------------------------------ # KPIs # ------------------------------
st.title("📊 Dashboard Cancer Patients")
col1, col2, col3, col4 = st.columns(4) 
col1.metric("Patients", len(df_filtered)) 
col2.metric("% Cancer", f"{100*df_filtered.Diagnosis.mean():.1f}%") 
col3.metric("Âge moyen", f"{df_filtered.Age.mean():.1f} ans") 
col4.metric("IMC moyen", f"{df_filtered.BMI.mean():.1f}")


# ------------------------------ # Graphiques démographiques # ------------------------------
st.header("📌 Profil démographique")

plot = (
    ggplot(df_filtered, aes(x="Age", fill="Diagnosis"))
    + geom_histogram(bins=20, position="stack", color="black")  # barres de même taille, bord noir
    + scale_fill_brewer(type='qual', palette='Set2')            # palette de couleurs plus jolie
    + theme_minimal()                                           # style épuré à la ggplot
    + labs(title="Distribution des âges par Diagnostic", 
           x="Âge", 
           y="Nombre de patients",
           fill="Diagnosis")
)


fig, ax = plt.subplots() 
df_filtered["Sexe_label"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax) 
ax.set_ylabel("") 

# Créer deux colonnes
col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot.draw())

with col2:
    st.pyplot(fig.draw())



# Fonction utilitaire pour des figures plus compactes
def small_fig(w=5, h=3, dpi=110):
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    return fig, ax



# Force le typage numérique
for col in ["Age", "Gender", "BMI", "Smoking", "GeneticRisk",
            "PhysicalActivity", "AlcoholIntake", "CancerHistory", "Diagnosis"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# =========================
#   CAS DE CANCER PAR ÂGE
# =========================


cancer_counts = (
    df[df["Diagnosis"] == 1]
    .groupby("Age")
    .size()
    .reset_index(name="Nombre de cancers")
    .sort_values("Age")
)

st.write("Tableau résumé :")
st.dataframe(cancer_counts, height=250)

fig, ax = small_fig()
ax.bar(cancer_counts["Age"], cancer_counts["Nombre de cancers"], color="tomato")
ax.set_xlabel("Âge")
ax.set_ylabel("Nombre de cas")
ax.set_title("Cas de cancer par âge")
st.pyplot(fig)


# =========================
#   TABAC & ALCOOL
# =========================
# Proportion tabac
prop_smoking = (
    df.groupby("Smoking")["Diagnosis"]
      .mean()
      .reset_index(name="Proportion de cancers")
)

# Proportion alcool
df["AlcoholIntake"] = pd.to_numeric(df["AlcoholIntake"], errors="coerce")
df["AlcoholIntake_round"] = (
    df["AlcoholIntake"].clip(0, 15).round().astype("Int64")
)
prop_alcohol = (
    df.dropna(subset=["AlcoholIntake_round"])
      .groupby("AlcoholIntake_round")["Diagnosis"]
      .mean()
      .reset_index(name="Proportion de cancers")
      .sort_values("AlcoholIntake_round")
)

# Affichage côte à côte
col1, col2 = st.columns(2)

with col1:
    st.write("📊 Proportion de cancers selon le tabagisme")
    st.dataframe(prop_smoking, height=180)

    fig, ax = small_fig()
    ax.bar(prop_smoking["Smoking"], prop_smoking["Proportion de cancers"], color="darkred")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non fumeur", "Fumeur"])
    ax.set_ylabel("Proportion")
    ax.set_title("Impact du tabagisme")
    st.pyplot(fig)

with col2:
    st.write("🍷 Proportion de cancers selon l’alcool")
    st.dataframe(
        prop_alcohol.assign(**{"Proportion (%)": (prop_alcohol["Proportion de cancers"]*100).round(1)}),
        height=180
    )

    fig, ax = small_fig()
    ax.bar(prop_alcohol["AlcoholIntake_round"].astype(str),
           prop_alcohol["Proportion de cancers"], color="steelblue")
    ax.set_xlabel("Unités/sem (arr.)")
    ax.set_ylim(0, 1)
    ax.set_title("Consommation d’alcool (0–15)")
    st.pyplot(fig)


# =========================
#   VARIABLES CROISÉES
# =========================
df["Age_x_Smoking"] = df["Age"] * df["Smoking"]
df["BMI_x_Physical"] = df["BMI"] * df["PhysicalActivity"]
df["Age_x_Alcohol"] = df["Age"] * df["AlcoholIntake"]
df["Smoking_x_Alcohol"] = df["Smoking"] * df["AlcoholIntake"]
df["Gender_x_Genetic"] = df["Gender"] * df["GeneticRisk"]


# =========================
#   RÉGRESSION LOGIT
# =========================
st.subheader("📈 Régression logistique (statsmodels) avec interactions")

base_features = ["Age", "Gender", "BMI", "Smoking", "GeneticRisk",
                 "PhysicalActivity", "AlcoholIntake", "CancerHistory"]
interactions = ["Age_x_Smoking", "BMI_x_Physical", "Age_x_Alcohol",
                "Smoking_x_Alcohol", "Gender_x_Genetic"]
features_sm = base_features + interactions

df_sm = df.dropna(subset=features_sm + ["Diagnosis"]).copy()
X_sm = sm.add_constant(df_sm[features_sm])
y_sm = df_sm["Diagnosis"].astype(int)

result = None
try:
    logit_model = sm.Logit(y_sm, X_sm)
    result = logit_model.fit(disp=False)
    st.text(result.summary())
except Exception as e:
    st.warning(f"Le modèle statsmodels n'a pas convergé : {e}")

if result is not None:
    odds_ratios = pd.DataFrame({
        "Variable": result.params.index,
        "Coef": result.params.values,
        "OR": np.exp(result.params.values),
        "p-value": result.pvalues
    })

    st.write("📊 Odds Ratios")
    st.dataframe(odds_ratios, height=250)

    fig, ax = small_fig()
    ax.barh(odds_ratios["Variable"], odds_ratios["OR"], color="skyblue")
    ax.axvline(1, color="red", linestyle="--")
    ax.set_xlabel("Odds Ratio (exp(coef))")
    ax.set_title("Impact des variables")
    st.pyplot(fig)


# =========================
#   SIMULATEUR
# =========================
st.subheader("🧑‍⚕️ Simulateur de risque")

age = st.slider("Âge", 20, 80, 50)
gender = st.radio("Genre", [0,1], format_func=lambda x: "Homme" if x==0 else "Femme")
bmi = st.slider("BMI", 15.0, 40.0, 25.0)
smoking = st.radio("Fumeur", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
genetic = st.radio("Risque génétique", [0,1,2], format_func=lambda x: ["Faible","Moyen","Élevé"][x])
activity = st.slider("Activité physique (h/sem)", 0.0, 30.0, 3.0)
alcohol = st.slider("Alcool (unités/sem)", 0.0, 15.0, 2.0)
history = st.radio("Antécédent cancer", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")

obs = pd.DataFrame([{
    "const": 1.0,
    "Age": age, "Gender": gender, "BMI": bmi, "Smoking": smoking,
    "GeneticRisk": genetic, "PhysicalActivity": activity,
    "AlcoholIntake": alcohol, "CancerHistory": history,
    "Age_x_Smoking": age*smoking,
    "BMI_x_Physical": bmi*activity,
    "Age_x_Alcohol": age*alcohol,
    "Smoking_x_Alcohol": smoking*alcohol,
    "Gender_x_Genetic": gender*genetic
}])

if result is not None:
    pred_prob = float(result.predict(obs[result.model.exog_names])[0])
    st.metric("Probabilité estimée de cancer", f"{pred_prob*100:.1f} %")


# =========================
#   FEATURE IMPORTANCE
# =========================
st.subheader("🔎 Feature importance (permutation)")

features_skl = features_sm
df_skl = df.dropna(subset=features_skl + ["Diagnosis"]).copy()
X = df_skl[features_skl]
y = df_skl["Diagnosis"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000, solver="liblinear"))
])

grid = GridSearchCV(pipe,
    {"model__C":[0.01,0.1,1,10], "model__penalty":["l1","l2"]},
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc", n_jobs=-1, refit=True
)
grid.fit(X_train, y_train)

best_pipe = grid.best_estimator_
y_proba = best_pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
st.write(f"AUC sur test : **{auc:.3f}**")

perm = permutation_importance(
    best_pipe, X_test, y_test,
    n_repeats=20, random_state=42, scoring="roc_auc"
)
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm.importances_mean,
    "Std": perm.importances_std
}).sort_values("Importance", ascending=True)

st.dataframe(imp_df.sort_values("Importance", ascending=False), height=250)

fig, ax = small_fig()
ax.barh(imp_df["Feature"], imp_df["Importance"])
ax.set_xlabel("Baisse d'AUC moyenne")
st.pyplot(fig)


# =========================
#   CONCLUSION
# =========================
st.subheader("🎯 Conclusion : Population à cibler")

st.markdown(
    """
    <div style="
        background-color: #ffe6f2;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #ff80bf;
        text-align: center;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    ">
        <h3 style="color:#cc0066;">Les personnes les plus à risque</h3>
        <p style="font-size:18px; color:#333;">
            ✅ <b>Femmes</b><br>
            ✅ <b>À partir de 50 ans</b><br><br>
            <i>Population à cibler pour des campagnes de sensibilisation</i>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
