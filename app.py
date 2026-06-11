import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Real Estate Analytics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Modern Real Estate Prediction & Analytics Dashboard")
st.write("Modify house characteristics in the sidebar and click 'Generate Dashboard' to evaluate valuations and market trends.")

# 2. Load Your Trained Model Pipeline
@st.cache_resource
def load_model():
    return joblib.load('house_price_xgb_model.pkl')

model = load_model()

# 3. Load Dataset & Pre-compute Metrics
@st.cache_data
def load_and_analyze_dataset():
    df = pd.read_csv('train.csv')
    
    # Calculate key dataset metrics for the KPI cards
    avg_price = df['SalePrice'].mean()
    max_price = df['SalePrice'].max()
    min_price = df['SalePrice'].min()
    
    # Prepare the 79-column structure template
    X_template = df.drop(columns=['Id', 'SalePrice'], errors='ignore')
    template_row = X_template.iloc[[0]].copy()
    
    # Create an aggregated dataframe for the historical trend chart
    trend_df = df.groupby('YearBuilt')['SalePrice'].mean().reset_index()
    
    return template_row, avg_price, max_price, min_price, trend_df

template_df, data_avg, data_max, data_min, historical_trends = load_and_analyze_dataset()

# ==========================================
# 🎛️ SIDEBAR FORM: Collects inputs securely
# ==========================================
# Using a form wraps all widgets together until the final click!
with st.sidebar.form(key="property_form"):
    st.header("📋 Property Specifications")
    st.write("Adjust the features below, then press the submit button at the bottom.")

    st.markdown("### 📐 Size & Location")
    gr_liv_area = st.number_input("Above Ground Living Area (Sq Ft)", min_value=300, max_value=6000, value=1500, step=50)
    total_bsmt_sf = st.number_input("Total Basement Area (Sq Ft)", min_value=0, max_value=6000, value=1000, step=50)

    zoning_options = {
        "Residential Low Density (RL)": "RL",
        "Residential Medium Density (RM)": "RM",
        "Floating Village Residential (FV)": "FV",
        "Residential High Density (RH)": "RH",
        "Commercial (C)": "C",
        "Agriculture (A)": "A",
        "Industrial (I)": "I",
        "Residential Low Density Park (RP)": "RP"
    }
    zoning_display = st.selectbox("General Zoning Classification", options=list(zoning_options.keys()), index=0)
    ms_zoning = zoning_options[zoning_display]

    air_options = {"Yes": "Y", "No": "N"}
    air_display = st.selectbox("Central Air Conditioning", options=list(air_options.keys()), index=0)
    central_air = air_options[air_display]

    st.markdown("---")
    st.markdown("### ✨ Quality & Amenities")
    overall_qual = st.slider("Overall Material & Finish Quality (1-10)", min_value=1, max_value=10, value=5)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2000, step=1)
    full_bath = st.selectbox("Number of Full Bathrooms", options=[0, 1, 2, 3, 4], index=2) 
    garage_cars = st.selectbox("Size of Garage (in cars)", options=[0, 1, 2, 3, 4], index=1)
    fireplaces = st.selectbox("Number of Fireplaces", options=[0, 1, 2, 3], index=1)

    garage_finish_options = {"Unfinished": "Unf", "Finished": "Fin", "Rough Finished": "RFn"}
    garage_finish_display = st.selectbox("Garage Finish", options=list(garage_finish_options.keys()), index=0)
    garage_finish = garage_finish_options[garage_finish_display]

    kitchen_qual_options = {"Poor": "Po", "Fair": "Fa", "Average": "TA", "Good": "Gd", "Excellent": "Ex"}
    kitchen_qual_display = st.selectbox("Kitchen Quality", options=list(kitchen_qual_options.keys()), index=2)
    kitchen_qual = kitchen_qual_options[kitchen_qual_display]
    
    # 💥 The Sidebar Submit Button!
    submit_button = st.form_submit_button(label="🚀 Generate Dashboard", use_container_width=True)


# ==========================================
# 📊 MAIN DISPLAY: Renders ONLY after submit
# ==========================================

if submit_button:
    # 🎆 Trigger cool load animations
    st.balloons()
    
    # Create 3 clean metric tracking cards across the top
    m1, m2, m3 = st.columns(3)
    m1.metric(label="📊 Dataset Historical Average", value=f"${data_avg:,.2f}")
    m2.metric(label="📈 Highest Record Sale Price", value=f"${data_max:,.2f}")
    m3.metric(label="📉 Lowest Record Sale Price", value=f"${data_min:,.2f}")

    st.markdown("---")

    # Prepare inputs for prediction engine
    input_df = template_df.copy()
    input_df['GrLivArea'] = gr_liv_area
    input_df['TotalBsmtSF'] = total_bsmt_sf
    input_df['OverallQual'] = overall_qual
    input_df['YearBuilt'] = year_built
    input_df['FullBath'] = full_bath
    input_df['MSZoning'] = ms_zoning
    input_df['CentralAir'] = central_air
    input_df['GarageCars'] = garage_cars
    input_df['GarageFinish'] = garage_finish
    input_df['KitchenQual'] = kitchen_qual
    input_df['Fireplaces'] = fireplaces

    try:
        prediction = model.predict(input_df)[0]
        
        st.subheader("🔮 ML Predictive Valuation")
        
        # Visual analysis check against dataset mean
        if prediction > data_avg:
            diff = prediction - data_avg
            comparison_text = f"This property aligns **${diff:,.2f} ABOVE** the historical market average."
        else:
            diff = data_avg - prediction
            comparison_text = f"This property aligns **${diff:,.2f} BELOW** the historical market average."
            
        st.success(f"### Estimated Property Valuation: **${prediction:,.2f}** \n*{comparison_text}*")
        
    except Exception as e:
        st.error(f"🚨 Model Prediction Stalled: {e}")

    st.markdown("---")

    # Render historical trend metrics visually using native line charts
    st.subheader("📈 Historical Market Growth Trends")
    st.write("This interactive chart traces the average price of residential transactions grouped by construction year across the entire training dataset.")

    # Rename columns for cleaner chart tooltips
    chart_data = historical_trends.rename(columns={'YearBuilt': 'Year Built', 'SalePrice': 'Average Sale Price'})
    st.line_chart(data=chart_data, x='Year Built', y='Average Sale Price', color="#2ca02c")

else:
    # This shows up when they first open the app to tell them what to do next
    st.info("👈 **The control room is ready!** Adjust the specifications inside the sidebar and click **'Generate Dashboard'** to compute your custom evaluation.")



