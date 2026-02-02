import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")
st.title("🛒 Customer Purchase Prediction – Decision Tree")

# --------------------------------
# Load Dataset
# --------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Retail_Cleaned.csv")

data = load_data()

# --------------------------------
# Feature & Target
# --------------------------------
features = ["Age", "Gender", "City", "ProductCategory", "Quantity", "TotalAmount"]

data["Will_Purchase_Next_Month"] = data["TotalAmount"].apply(
    lambda x: "Yes" if x > 5000 else "No"
)

X = data[features].copy()
y = data["Will_Purchase_Next_Month"]

# Encode categorical variables
label_encoders = {}
for col in ["Gender", "City", "ProductCategory"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# --------------------------------
# Train-Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --------------------------------
# Model Save / Load
# --------------------------------
MODEL_FILE = "model.pkl"

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        tree_model = pickle.load(f)
else:
    tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_model.fit(X_train, y_train)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(tree_model, f)

# --------------------------------
# Sidebar – Customer Input
# --------------------------------
st.sidebar.header("🧍 Customer Details")

age = st.sidebar.number_input("Age", 18, 80, 30)

gender = st.sidebar.selectbox(
    "Gender", label_encoders["Gender"].classes_
)

city = st.sidebar.selectbox(
    "City", label_encoders["City"].classes_
)

product = st.sidebar.selectbox(
    "Product Category", label_encoders["ProductCategory"].classes_
)

quantity = st.sidebar.number_input("Quantity", 1, 20, 1)

total_amount = st.sidebar.number_input(
    "Total Amount (₹)", 100, 50000, 1000
)

# Encode inputs
input_df = pd.DataFrame({
    "Age": [age],
    "Gender": [label_encoders["Gender"].transform([gender])[0]],
    "City": [label_encoders["City"].transform([city])[0]],
    "ProductCategory": [label_encoders["ProductCategory"].transform([product])[0]],
    "Quantity": [quantity],
    "TotalAmount": [total_amount]
})

if st.sidebar.button("🔮 Predict"):
    prediction = tree_model.predict(input_df)[0]

    if prediction == "Yes":
        st.sidebar.success("✅ Customer WILL purchase next month")
    else:
        st.sidebar.error("❌ Customer will NOT purchase next month")

# --------------------------------
# Main Page – Evaluation
# --------------------------------
y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Performance")
st.metric("Accuracy", f"{accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax_cm)
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# Classification Report
st.subheader("📄 Classification Report")
st.text(classification_report(y_test, y_pred))

# --------------------------------
# Feature Importance
# --------------------------------
st.subheader("📌 Feature Importance")
fi_df = pd.DataFrame({
    "Feature": features,
    "Importance": tree_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax_fi)
st.pyplot(fig_fi)

# --------------------------------
# Decision Tree
# --------------------------------
st.subheader("🌳 Decision Tree")
fig_tree, ax_tree = plt.subplots(figsize=(14, 8))
plot_tree(tree_model,
          feature_names=features,
          class_names=["No", "Yes"],
          filled=True,
          fontsize=8,
          ax=ax_tree)
st.pyplot(fig_tree)

# --------------------------------
# Customer Prediction Mapping
# --------------------------------
st.subheader("🧾 Customer Prediction Mapping")

customer_mapping = data.loc[X_test.index, ["CustomerID"]].copy()
customer_mapping["Predicted_Purchase_Label"] = y_pred

st.dataframe(customer_mapping.head(10))

csv = customer_mapping.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Download Mapping CSV",
    csv,
    "Customer_Purchase_Prediction_Mapping.csv",
    "text/csv"
)
