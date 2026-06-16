import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except Exception:
    pass
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ARCHIVE = "/Users/hansjoseph/Downloads/archive"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11})

# ===== 1. LOAD & INSPECT =====
print("="*45); print("DATA INSPECTION"); print("="*45)
df = pd.read_csv(f"{ARCHIVE}/order_products__prior.csv", nrows=1_000_000)
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# ===== 2. FEATURE ENGINEERING =====
prod = df.groupby("product_id").agg(
    product_orders=("reordered", "size"),
    product_reorder_rate=("reordered", "mean"),
).reset_index()
df = df.merge(prod, on="product_id", how="left")
X = df[["add_to_cart_order", "product_orders", "product_reorder_rate"]]
y = df["reordered"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== 3. TRAIN =====
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=80, max_depth=14, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# ===== 4. EVALUATE =====
print("\n"+"="*45); print("MODEL EVALUATION"); print("="*45)
print("Accuracy:", round(accuracy_score(y_test, preds), 4))
print("\nClassification Report:\n", classification_report(y_test, preds))

# ===== 5. CHARTS =====
# Fig 13 - reorder distribution
counts = df["reordered"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7,5))
bars = ax.bar(["Not Reordered (0)","Reordered (1)"], counts.values, color=["#E07A5F","#3D9970"])
for b,v in zip(bars, counts.values):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
ax.set_title("Distribution of Reordered vs Non-Reordered Products", fontweight="bold")
ax.set_ylabel("Number of records")
plt.tight_layout(); plt.savefig("reorder_distribution.png"); plt.close()

# Fig 14 - reorder rate by position
rate = df[df["add_to_cart_order"]<=30].groupby("add_to_cart_order")["reordered"].mean()
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(rate.index, rate.values, marker="o", color="#3D5A80", linewidth=2)
ax.set_title("Reorder Rate by Add-to-Cart Position", fontweight="bold")
ax.set_xlabel("Add-to-cart position"); ax.set_ylabel("Reorder rate")
plt.tight_layout(); plt.savefig("reorder_by_position.png"); plt.close()

# Fig 15 - top products
products = pd.read_csv(f"{ARCHIVE}/products.csv")
top = (df["product_id"].value_counts().head(10).rename_axis("product_id")
       .reset_index(name="orders").merge(products[["product_id","product_name"]], on="product_id", how="left"))
fig, ax = plt.subplots(figsize=(9,6))
bars = ax.barh(top["product_name"][::-1], top["orders"][::-1], color="#2A6F97")
for b,v in zip(bars, top["orders"][::-1]):
    ax.text(v, b.get_y()+b.get_height()/2, f" {v:,}", va="center")
ax.set_title("Top 10 Most-Ordered Products", fontweight="bold")
ax.set_xlabel("Number of orders")
plt.tight_layout(); plt.savefig("top_products.png"); plt.close()
print("\nTop 10 products by name:\n", top[["product_name","orders"]].to_string(index=False))

# Fig 16 - feature importance
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(7,4))
bars = ax.barh(imp.index, imp.values, color="#EE9B00")
for b,v in zip(bars, imp.values):
    ax.text(v, b.get_y()+b.get_height()/2, f" {v:.3f}", va="center")
ax.set_title("Random Forest Feature Importance", fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout(); plt.savefig("feature_importance.png"); plt.close()

# Fig 18 - confusion matrix
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Not Reordered","Reordered"]); ax.set_yticklabels(["Not Reordered","Reordered"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Random Forest", fontweight="bold")
total = cm.sum()
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}\n({cm[i,j]/total*100:.1f}%)", ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black", fontweight="bold")
plt.colorbar(im); plt.tight_layout(); plt.savefig("confusion_matrix.png"); plt.close()

print("\nSaved 5 charts: reorder_distribution, reorder_by_position, top_products, feature_importance, confusion_matrix")
print("done")
