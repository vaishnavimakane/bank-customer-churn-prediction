
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load trained model
model = joblib.load("churn_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 European Bank Customer Churn Prediction")
st.write("Predict whether a bank customer is likely to leave the bank.")

st.divider()

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score", 350, 850, 650
    )

    geography = st.selectbox(
        "Geography", ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender", ["Female", "Male"]
    )

    age = st.number_input(
        "Age", 18, 100, 35
    )

    tenure = st.number_input(
        "Tenure", 0, 10, 5
    )

with col2:
    balance = st.number_input(
        "Balance", min_value=0.0, value=50000.0
    )

    products = st.selectbox(
        "Number of Products", [1, 2, 3, 4]
    )

    has_card = st.selectbox(
        "Has Credit Card", [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    active_member = st.selectbox(
        "Is Active Member", [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    salary = st.number_input(
        "Estimated Salary", min_value=0.0,
        value=50000.0
    )

st.divider()

if st.button("Predict Customer Churn", type="primary"):

    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": active_member,
        "EstimatedSalary": salary
    }])
    input_data["BalanceToSalaryRatio"] = input_data["Balance"] / (input_data["EstimatedSalary"] + 1)
    input_data["ProductDensity"] = input_data["NumOfProducts"] / 4
    input_data["EngagementProductInteraction"] = input_data["IsActiveMember"] * input_data["NumOfProducts"]
    input_data["AgeTenureInteraction"] = input_data["Age"] * input_data["Tenure"]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("⚠️ Customer is likely to leave the bank.")
    else:
        st.success("✅ Customer is likely to stay with the bank.")
import os

file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "European_Bank .xlsx"
)

df = pd.read_excel(file_path)

st.write("Dataset Loaded Successfully")
st.write(df.head())
st.write("Dataset Columns:", df.columns.tolist())


# Clean Dataset
df["Exited"] = pd.to_numeric(df["Exited"], errors="coerce")
df = df.dropna(subset=["Exited"]).copy()

# Dashboard Title
st.divider()
st.title("📊 Customer Churn Analytics Dashboard")

# Dashboard Filters
st.subheader("Filter Customers")

col1, col2 = st.columns(2)

with col1:
    selected_geo = st.selectbox(
        "Select Geography",
        ["All"] + sorted(df["Geography"].dropna().unique().tolist())
    )

with col2:
    selected_gender = st.selectbox(
        "Select Gender",
        ["All"] + sorted(df["Gender"].dropna().unique().tolist())
    )

# Apply Filters
filtered_df = df.copy()

if selected_geo != "All":
    filtered_df = filtered_df[
        filtered_df["Geography"] == selected_geo
    ]

if selected_gender != "All":
    filtered_df = filtered_df[
        filtered_df["Gender"] == selected_gender
    ]

# KPI Calculations
total_customers = len(filtered_df)

churned = (filtered_df["Exited"] == 1).sum()

retained = (filtered_df["Exited"] == 0).sum()

churn_rate = (
    churned / total_customers * 100
    if total_customers > 0 else 0
)

# Display KPI Cards
st.subheader("Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Customers", f"{total_customers:,}")
k2.metric("Churned Customers", f"{churned:,}")
k3.metric("Retained Customers", f"{retained:,}")
k4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.divider()

if total_customers == 0:
    st.warning("No customers found for the selected filters.")
    st.stop()


# 1. Customer Churn Distribution
st.subheader("1. Customer Churn Distribution")

churned = (filtered_df["Exited"] == 1).sum()
retained = (filtered_df["Exited"] == 0).sum()

churn_data = pd.DataFrame({
    "Customer Status": ["Retained", "Churned"],
    "Customers": [retained, churned]
})

fig1 = px.bar(
    churn_data,
    x="Customer Status",
    y="Customers",
    color="Customer Status",
    text="Customers",
    color_discrete_map={
        "Retained": "#2E8B57",
        "Churned": "#E74C3C"
    },
    title="Retained vs Churned Customers"
)

fig1.update_traces(textposition="outside")
fig1.update_layout(showlegend=False)

st.plotly_chart(fig1, use_container_width=True)


# 2. Geography-wise Churn Analysis
st.subheader("2. Geography-wise Churn Analysis")

geo_data = (
    filtered_df.groupby("Geography")["Exited"]
    .mean().mul(100)
    .reset_index(name="Churn Rate (%)")
)

fig2 = px.bar(
    geo_data,
    x="Geography",
    y="Churn Rate (%)",
    text="Churn Rate (%)",
    color="Churn Rate (%)",
    color_continuous_scale="Blues",
    title="Churn Rate by Geography"
)

fig2.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(fig2, use_container_width=True)


# 3. Gender-wise Churn Analysis
st.subheader("3. Gender-wise Churn Analysis")

gender_data = (
    filtered_df.groupby("Gender")["Exited"]
    .mean().mul(100)
    .reset_index(name="Churn Rate (%)")
)

fig3 = px.bar(
    gender_data,
    x="Gender",
    y="Churn Rate (%)",
    text="Churn Rate (%)",
    color="Gender",
    title="Churn Rate by Gender"
)

fig3.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(fig3, use_container_width=True)


# 4. Age-wise Churn Analysis
st.subheader("4. Age-wise Churn Analysis")

age_df = filtered_df.copy()

age_df["Age Group"] = pd.cut(
    age_df["Age"],
    bins=[0, 25, 35, 45, 55, 65, 120],
    labels=["18-25", "26-35", "36-45",
            "46-55", "56-65", "65+"]
)

age_data = (
    age_df.groupby("Age Group", observed=False)["Exited"]
    .mean().mul(100)
    .reset_index(name="Churn Rate (%)")
)

fig4 = px.bar(
    age_data,
    x="Age Group",
    y="Churn Rate (%)",
    text="Churn Rate (%)",
    color="Churn Rate (%)",
    color_continuous_scale="Oranges",
    title="Churn Rate by Age Group"
)

fig4.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(fig4, use_container_width=True)


# 5. Product-wise Churn Analysis
st.subheader("5. Product-wise Churn Analysis")

product_data = (
    filtered_df.groupby("NumOfProducts")["Exited"]
    .mean().mul(100)
    .reset_index(name="Churn Rate (%)")
)

fig5 = px.bar(
    product_data,
    x="NumOfProducts",
    y="Churn Rate (%)",
    text="Churn Rate (%)",
    color="Churn Rate (%)",
    color_continuous_scale="Purples",
    title="Churn Rate by Number of Products"
)

fig5.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(fig5, use_container_width=True)


# 6. Active vs Inactive Customer Analysis
st.subheader("6. Active vs Inactive Customer Analysis")

activity_data = (
    filtered_df.groupby("IsActiveMember")["Exited"]
    .mean().mul(100)
    .reset_index(name="Churn Rate (%)")
)

activity_data["Membership Status"] = (
    activity_data["IsActiveMember"].map({
        0: "Inactive Members",
        1: "Active Members"
    })
)

fig6 = px.bar(
    activity_data,
    x="Membership Status",
    y="Churn Rate (%)",
    text="Churn Rate (%)",
    color="Membership Status",
    title="Churn Rate: Active vs Inactive Members"
)

fig6.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

st.plotly_chart(fig6, use_container_width=True)


st.success("Customer Churn Dashboard Loaded Successfully!")

# 7. Feature Importance Dashboard
st.divider()
st.subheader("7. Feature Importance Analysis")

importance_df = pd.read_excel("feature_importance.xlsx")

st.write("Key factors influencing customer churn:")

fig7 = px.bar(
    importance_df.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Important Features",
    color="Importance"
)

fig7.update_layout(yaxis={"categoryorder": "total ascending"})

st.plotly_chart(fig7, use_container_width=True)

# 8. Churn Probability Distribution
st.divider()
st.subheader("8. Churn Probability Distribution")

# Prepare customer data for prediction
probability_data = df.drop(
    columns=["Exited", "CustomerId", "Surname", "Year"],
    errors="ignore"
).copy()

# Apply the same feature engineering used during model training
probability_data["BalanceToSalaryRatio"] = (
    probability_data["Balance"] /
    (probability_data["EstimatedSalary"] + 1)
)

probability_data["ProductDensity"] = (
    probability_data["NumOfProducts"] / 4
)

probability_data["EngagementProductInteraction"] = (
    probability_data["IsActiveMember"] *
    probability_data["NumOfProducts"]
)

probability_data["AgeTenureInteraction"] = (
    probability_data["Age"] *
    probability_data["Tenure"]
)

# Calculate churn probabilities
churn_probabilities = model.predict_proba(
    probability_data
)[:, 1] * 100

probability_df = pd.DataFrame({
    "Churn Probability (%)": churn_probabilities
})

# Display histogram
fig8 = px.histogram(
    probability_df,
    x="Churn Probability (%)",
    nbins=20,
    title="Distribution of Customer Churn Risk"
)

st.plotly_chart(fig8, use_container_width=True)

# 9. What-If Scenario Simulator
st.divider()
st.subheader("9. What-If Scenario Simulator")

st.write(
    "Change customer activity and product details "
    "to see how churn risk may change."
)

scenario_active = st.selectbox(
    "Scenario: Active Member",
    [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No",
    key="scenario_active"
)

scenario_products = st.selectbox(
    "Scenario: Number of Products",
    [1, 2, 3, 4],
    key="scenario_products"
)

scenario_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": scenario_products,
    "HasCrCard": has_card,
    "IsActiveMember": scenario_active,
    "EstimatedSalary": salary
}])

scenario_data["BalanceToSalaryRatio"] = (
    scenario_data["Balance"] /
    (scenario_data["EstimatedSalary"] + 1)
)
scenario_data["ProductDensity"] = (
    scenario_data["NumOfProducts"] / 4
)
scenario_data["EngagementProductInteraction"] = (
    scenario_data["IsActiveMember"] *
    scenario_data["NumOfProducts"]
)
scenario_data["AgeTenureInteraction"] = (
    scenario_data["Age"] * scenario_data["Tenure"]
)

scenario_probability = (
    model.predict_proba(scenario_data)[0][1] * 100
)

st.metric(
    "Scenario Churn Probability",
    f"{scenario_probability:.2f}%"
)

if scenario_probability >= 50:
    st.warning("This scenario shows a higher predicted churn risk.")
else:
    st.success("This scenario shows a lower predicted churn risk.")


# 10. SHAP Explainability Analysis
st.divider()
st.subheader("10. SHAP Explainability Analysis")

st.write(
    "This chart explains which customer features "
    "influence the model's churn predictions."
)

st.image(
    "shap_summary.png",
    caption="SHAP Summary Plot - Customer Churn Factors",
    use_container_width=True
)