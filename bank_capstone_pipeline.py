"""
Week 6 Capstone: Bank Customer Subscription Prediction and Customer Segmentation
Data Science with Python — YuvaIntern Internship

End-to-end pipeline:
  1. Data acquisition
  2. Data quality assessment
  3. Missing value / categorical handling
  4. EDA + statistical summaries + visualizations
  5. Feature engineering
  6. Supervised learning (subscription prediction)
  7. Model comparison + cross-validation + evaluation
  8. Unsupervised learning (customer segmentation)
  9. Cluster evaluation (Elbow + Silhouette)
  10. Business interpretation is written up separately in the report,
      driven by the numeric results this script prints/saves.

All numeric results are saved to outputs/results.json so the report
generation step uses only genuine, computed values.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42

FIG_DIR = "figures"
OUT_DIR = "outputs"
results = {}

# ---------------------------------------------------------------
# 1. DATA ACQUISITION
# ---------------------------------------------------------------
df_raw = pd.read_csv("data/bank.csv")
results["dataset"] = {
    "source": "UCI Machine Learning Repository - Bank Marketing Dataset "
              "(Moro, Cortez & Rita, 2014), mirrored as a public CSV",
    "n_rows": int(df_raw.shape[0]),
    "n_columns": int(df_raw.shape[1]),
    "columns": list(df_raw.columns),
}

# ---------------------------------------------------------------
# 2. DATA QUALITY ASSESSMENT
# ---------------------------------------------------------------
dq = {}
dq["duplicate_rows"] = int(df_raw.duplicated().sum())
dq["null_counts"] = {c: int(v) for c, v in df_raw.isnull().sum().items()}
unknown_cols = ["job", "education", "contact", "poutcome"]
dq["unknown_value_counts"] = {c: int((df_raw[c] == "unknown").sum()) for c in unknown_cols}
dq["target_distribution"] = df_raw["deposit"].value_counts().to_dict()
dq["numeric_summary"] = df_raw.describe().round(2).to_dict()
results["data_quality"] = dq

df = df_raw.drop_duplicates().copy()

# ---------------------------------------------------------------
# 3. MISSING VALUE / CATEGORICAL HANDLING
# ---------------------------------------------------------------
# "unknown" is this dataset's explicit missing-value marker for categoricals.
# poutcome is >70% unknown (no prior campaign) -> keep as its own informative
# category rather than imputing. job/education/contact -> impute with mode
# since they are a small share of rows.
for col in ["job", "education", "contact"]:
    mode_val = df.loc[df[col] != "unknown", col].mode()[0]
    n_imputed = int((df[col] == "unknown").sum())
    df[col] = df[col].replace("unknown", mode_val)
    dq.setdefault("imputation_log", {})[col] = {"imputed_with": mode_val, "rows_imputed": n_imputed}
# poutcome: keep "unknown" as a valid category (means "no previous contact")
dq.setdefault("imputation_log", {})["poutcome"] = "kept as own category (no prior contact)"

# pdays == -1 means "never previously contacted" -> convert to explicit flag
df["was_previously_contacted"] = (df["pdays"] != -1).astype(int)

# ---------------------------------------------------------------
# 4. EDA + STATISTICAL SUMMARIES + VISUALIZATIONS
# ---------------------------------------------------------------
df["deposit_flag"] = (df["deposit"] == "yes").astype(int)

# 4a. Target balance
plt.figure(figsize=(5, 4))
sns.countplot(x="deposit", data=df, palette="viridis")
plt.title("Target Class Distribution: Term Deposit Subscription")
plt.xlabel("Subscribed to term deposit")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_target_distribution.png", dpi=150)
plt.close()

# 4b. Age distribution by subscription
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="age", hue="deposit", bins=30, kde=True, multiple="stack")
plt.title("Age Distribution by Subscription Outcome")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_age_distribution.png", dpi=150)
plt.close()

# 4c. Subscription rate by job
job_rate = df.groupby("job")["deposit_flag"].mean().sort_values(ascending=False)
plt.figure(figsize=(7, 5))
job_rate.plot(kind="barh", color="teal")
plt.title("Subscription Rate by Job Category")
plt.xlabel("Subscription rate")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_subscription_by_job.png", dpi=150)
plt.close()

# 4d. Correlation heatmap (numeric features)
num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
plt.figure(figsize=(7, 6))
sns.heatmap(df[num_cols + ["deposit_flag"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_correlation_heatmap.png", dpi=150)
plt.close()

# 4e. Balance by subscription (boxplot, log scale for readability)
plt.figure(figsize=(5, 4))
sns.boxplot(x="deposit", y="balance", data=df, showfliers=False)
plt.title("Account Balance by Subscription Outcome")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_balance_by_deposit.png", dpi=150)
plt.close()

# 4f. Subscription rate by month
month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month_rate = df.groupby("month")["deposit_flag"].mean().reindex(month_order)
plt.figure(figsize=(7, 4))
month_rate.plot(kind="bar", color="slateblue")
plt.title("Subscription Rate by Contact Month")
plt.ylabel("Subscription rate")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_subscription_by_month.png", dpi=150)
plt.close()

eda = {
    "job_subscription_rate": job_rate.round(3).to_dict(),
    "month_subscription_rate": month_rate.round(3).dropna().to_dict(),
    "overall_subscription_rate": round(float(df["deposit_flag"].mean()), 3),
    "mean_duration_subscribed": round(float(df.loc[df.deposit_flag==1, "duration"].mean()), 1),
    "mean_duration_not_subscribed": round(float(df.loc[df.deposit_flag==0, "duration"].mean()), 1),
    "correlation_with_target": df[num_cols].corrwith(df["deposit_flag"]).round(3).to_dict(),
}
results["eda"] = eda

# ---------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ---------------------------------------------------------------
df["age_group"] = pd.cut(df["age"], bins=[17,30,40,50,60,100],
                          labels=["18-30","31-40","41-50","51-60","60+"])
df["has_any_loan"] = ((df["housing"]=="yes") | (df["loan"]=="yes")).astype(int)
df["contact_recency"] = np.where(df["pdays"] == -1, 0, 1)  # duplicate-safe flag
df["campaign_intensity"] = pd.cut(df["campaign"], bins=[0,1,3,6,100],
                                   labels=["1","2-3","4-6","7+"])

# IMPORTANT MODELING DECISION (documented for the report):
# 'duration' (last call length in seconds) is only known AFTER a call is made
# and is a well-known leakage feature in this dataset (very high correlation
# with the target because calls that end in "yes" tend to run longer).
# We build two model sets: one WITH duration (for reference, matches most
# published benchmarks) and one WITHOUT duration (deployable pre-call model).
feature_cols_full = ["age","job","marital","education","default","balance",
                     "housing","loan","contact","day","month","duration",
                     "campaign","pdays","previous","poutcome",
                     "was_previously_contacted","has_any_loan"]
feature_cols_deployable = [c for c in feature_cols_full if c != "duration"]

target = "deposit_flag"

categorical_features = ["job","marital","education","default","housing",
                         "loan","contact","month","poutcome"]
numeric_features_full = ["age","balance","day","duration","campaign","pdays",
                          "previous","was_previously_contacted","has_any_loan"]
numeric_features_deploy = [c for c in numeric_features_full if c != "duration"]

def build_preprocessor(numeric_features):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

# ---------------------------------------------------------------
# 6/7. SUPERVISED LEARNING + MODEL COMPARISON + CROSS-VALIDATION
# ---------------------------------------------------------------
def run_model_suite(feature_cols, numeric_features, label):
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    preprocessor = build_preprocessor(numeric_features)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    suite_results = {}
    best_name, best_auc, best_pipe = None, -1, None

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "cv_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_auc_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "test_precision": round(float(precision_score(y_test, y_pred)), 4),
            "test_recall": round(float(recall_score(y_test, y_pred)), 4),
            "test_f1": round(float(f1_score(y_test, y_pred)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        }
        suite_results[name] = metrics
        if metrics["test_roc_auc"] > best_auc:
            best_auc = metrics["test_roc_auc"]
            best_name = name
            best_pipe = pipe

    # Confusion matrix + ROC curve for the best model
    y_pred_best = best_pipe.predict(X_test)
    y_proba_best = best_pipe.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No","Yes"], yticklabels=["No","Yes"])
    plt.title(f"Confusion Matrix — {best_name} ({label})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = f"{FIG_DIR}/07_confusion_matrix_{label}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba_best)
    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, label=f"{best_name} (AUC={best_auc:.3f})", color="darkorange")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — Best Model ({label})")
    plt.legend()
    plt.tight_layout()
    fname2 = f"{FIG_DIR}/08_roc_curve_{label}.png"
    plt.savefig(fname2, dpi=150)
    plt.close()

    # Feature importance (if tree-based best model, else use Random Forest for insight)
    importance_source = best_pipe if hasattr(best_pipe.named_steps["clf"], "feature_importances_") \
        else Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1))]).fit(X_train, y_train)
    ohe_names = importance_source.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_features) + list(ohe_names)
    importances = importance_source.named_steps["clf"].feature_importances_
    fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(15)

    plt.figure(figsize=(7, 6))
    fi.iloc[::-1].plot(kind="barh", color="seagreen")
    plt.title(f"Top 15 Feature Importances ({label} model)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/09_feature_importance_{label}.png", dpi=150)
    plt.close()

    return {
        "model_comparison": suite_results,
        "best_model": best_name,
        "best_test_roc_auc": round(best_auc, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred_best, output_dict=True),
        "top_features": fi.round(4).to_dict(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

results["supervised_with_duration"] = run_model_suite(feature_cols_full, numeric_features_full, "with_duration")
results["supervised_deployable"] = run_model_suite(feature_cols_deployable, numeric_features_deploy, "deployable")

# ---------------------------------------------------------------
# 8/9. UNSUPERVISED LEARNING: CUSTOMER SEGMENTATION
# ---------------------------------------------------------------
cluster_features = ["age", "balance", "campaign", "previous"]
X_cluster = df[cluster_features].copy()
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Elbow method
inertias = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_cluster_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster_scaled, labels))

plt.figure(figsize=(6, 4))
plt.plot(list(K_range), inertias, marker="o", color="crimson")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for Optimal k")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/10_elbow_method.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(list(K_range), sil_scores, marker="o", color="navy")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by k")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/11_silhouette_scores.png", dpi=150)
plt.close()

# Silhouette technically peaks at k=8 in the tested range, but the gain over
# k=4-6 is marginal (~0.02) while interpretability for a business audience
# drops sharply with more segments. We choose k within a business-practical
# range (3-6) that maximizes silhouette, and report the full curve so the
# trade-off is transparent rather than hidden.
practical_k_range = [k for k in K_range if 3 <= k <= 6]
practical_sil = [sil_scores[list(K_range).index(k)] for k in practical_k_range]
best_k = practical_k_range[int(np.argmax(practical_sil))]
km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
df["cluster"] = km_final.fit_predict(X_cluster_scaled)
final_silhouette = silhouette_score(X_cluster_scaled, df["cluster"])

cluster_profile = df.groupby("cluster")[cluster_features + ["deposit_flag"]].mean().round(2)
cluster_sizes = df["cluster"].value_counts().sort_index()

plt.figure(figsize=(6.5, 5))
sns.scatterplot(x="age", y="balance", hue="cluster", data=df, palette="Set2", alpha=0.6, s=25)
plt.title(f"Customer Segments (k={best_k}) — Age vs Balance")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/12_cluster_scatter.png", dpi=150)
plt.close()

plt.figure(figsize=(6.5, 4.5))
(cluster_profile["deposit_flag"]).plot(kind="bar", color="darkcyan")
plt.title("Subscription Rate by Customer Segment")
plt.ylabel("Subscription rate")
plt.xlabel("Cluster")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/13_cluster_subscription_rate.png", dpi=150)
plt.close()

results["clustering"] = {
    "features_used": cluster_features,
    "k_range_tested": list(K_range),
    "inertias": [round(x, 1) for x in inertias],
    "silhouette_scores": [round(x, 4) for x in sil_scores],
    "chosen_k": int(best_k),
    "chosen_k_rationale": "Max silhouette score within a business-interpretable "
        "range of 3-6 segments; global max silhouette in tested range 2-10 was "
        f"at k={list(K_range)[int(np.argmax(sil_scores))]} (silhouette="
        f"{round(max(sil_scores),4)}) but was judged too granular for actionable segmentation.",
    "final_silhouette_score": round(float(final_silhouette), 4),
    "cluster_sizes": cluster_sizes.to_dict(),
    "cluster_profile": cluster_profile.to_dict(orient="index"),
}

# ---------------------------------------------------------------
# SAVE ALL RESULTS
# ---------------------------------------------------------------
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("PIPELINE COMPLETE")
print(json.dumps({
    "best_model_with_duration": results["supervised_with_duration"]["best_model"],
    "auc_with_duration": results["supervised_with_duration"]["best_test_roc_auc"],
    "best_model_deployable": results["supervised_deployable"]["best_model"],
    "auc_deployable": results["supervised_deployable"]["best_test_roc_auc"],
    "chosen_k": best_k,
    "final_silhouette": round(float(final_silhouette), 4),
}, indent=2))
