import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="CO₂ Emissions Estimation",
    page_icon="🌍",
    layout="wide"
)

# --- Caching Functions for Loading Data & Models ---
@st.cache_data
def load_source_data(file_path):
    """Loads and caches the source CSV data."""
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        df['name'] = df['name'].ffill()
        return df
    except Exception as e:
        st.error(f"Error loading source data file: {e}")
        return None

@st.cache_resource
def load_model(file_path):
    """Generic function to load a .pkl model file."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model file {file_path}: {e}")
        return None

# --- Main Application UI ---
st.image("Header_Colab.png")

# --- Load Data and Find Models ---
MODEL_DIR = 'province_models_fixed_order' 
source_df = load_source_data('cleaned_dataset.csv')
regression_model = load_model('gradient_boosting_model.pkl')

# --- Pre-calculate quantiles and define level function ---
quantiles = {}
if source_df is not None:
    # Define columns for which to calculate quantiles
    cols_for_quantiles = [
        'per_capita_gdp_yuan', 'total_population_million', 'urbanization_rate_percent',
        'proportion_of_primary_industry_percent', 'proportion_of_secondary_industry_percent',
        'proportion_of_the_tertiary_industry_percent', 'coal_proportion_percent',
        'total_carbon_dioxide_emissions_(_million_tons_)'
    ]
    # Calculate and store quantiles (25% and 75%)
    quantiles = source_df[cols_for_quantiles].quantile([0.25, 0.75]).to_dict()

def get_level(value, column_name):
    """Determines if a value is Low, Medium, or High based on pre-calculated dataset quantiles."""
    if not quantiles or column_name not in quantiles:
        return "" # Return empty if quantiles not available
    
    q1 = quantiles[column_name][0.25]
    q3 = quantiles[column_name][0.75]

    if value < q1:
        return "L"
    elif value > q3:
        return "H"
    else:
        return "M"

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")

# Initialize session state for page navigation if it doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Determine the type for each button based on the current page
home_btn_type = "primary" if st.session_state.page == 'Home' else "secondary"
est_btn_type = "primary" if st.session_state.page == 'Estimate' else "secondary"

# Create buttons and update session state on click
if st.sidebar.button("Home", use_container_width=True, type=home_btn_type):
    st.session_state.page = 'Home'
    st.rerun()
if st.sidebar.button("Estimate", use_container_width=True, type=est_btn_type):
    st.session_state.page = 'Estimate'
    st.rerun()


# ==========================================================================
# HOME PAGE
# ==========================================================================
if st.session_state.page == "Home":
    home_tab1, home_tab2 = st.tabs(["Description", "Dataset"])

    with home_tab1:
        st.header("Project Description")
        st.write("""
        This application provides two advanced tools for analyzing and predicting Carbon Dioxide (CO₂) emissions across various provinces in China. 
        It leverages historical data from 1999 to 2019 to build robust machine learning models.
        """)
        st.divider()
        st.subheader("Key Features")

        # --- Feature 1 with Image ---
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("Regression.png")
        with col2:
            st.markdown("""
            **1. Estimation with Specific Variables (Multivariate Regression):**
            - This tool uses a **Gradient Boosting Regressor** model, which has been trained on the entire dataset across all provinces.
            - It learns the complex relationships between CO₂ emissions and various socio-economic factors like GDP, population, urbanization rate, and industrial structure.
            - This allows you to create detailed "what-if" scenarios to see how specific policy and economic changes might impact emissions.
            """)

        st.divider()

        # --- Feature 2 with Image ---
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("forecasting.jpg", )
        with col2:
            st.markdown("""
            **2. Forecasting (Univariate Time-Series):**
            - This tool employs an **ARIMA (AutoRegressive Integrated Moving Average)** model.
            - A unique ARIMA model has been trained for each individual province, focusing solely on its historical emissions trend.
            - This provides a "baseline" forecast, showing where a province's emissions are headed if its historical momentum continues without major external changes.
            """)
        
        st.divider()
        st.markdown("By combining these two approaches, users can gain a comprehensive understanding of both the underlying drivers of emissions and the likely future trends.")

    with home_tab2:
        st.header("Dataset Preview")
        st.markdown("This is the `cleaned_dataset.csv` file used for training all models.")
        if source_df is not None:
            st.dataframe(source_df)
        else:
            st.error("Dataset file not found.")

# ==========================================================================
# ESTIMATE PAGE
# ==========================================================================
elif st.session_state.page == "Estimate":
    est_tab1, est_tab2 = st.tabs(["Estimate with Specific Variable", "Forecast"])

    # --- Regression Model Tab ---
    with est_tab1:
        if regression_model is None or source_df is None:
            st.error("Regression model or source data not found. Please check your files.")
        else:
            province_names = sorted(source_df['name'].unique())
            
            # --- Input Form ---
            with st.form(key='prediction_form'):
                st.subheader("Input the correct values into the following boxes to estimate total carbon dioxide emissions")
                
                # Input fields laid out in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    province = st.selectbox("Province", options=province_names)
                    year = st.number_input("Year", min_value=2020, max_value=2050, value=2025)
                    gdp = st.number_input("gdp per capita (yuan)", min_value=0)
                with col2:
                    population = st.number_input("Total Population (million)", min_value=0.0, format="%.2f")
                    urbanization = st.number_input("Urbanization Rate (%)", min_value=0.0, max_value=100.0, format="%.1f")
                    primary_ind = st.number_input("Proportion of Primary Industry (%)", min_value=0.0, max_value=100.0, format="%.1f")
                with col3:
                    secondary_ind = st.number_input("Proportion of Secondary Industry (%)", min_value=0.0, max_value=100.0, format="%.1f")
                    tertiary_ind = st.number_input("Proportion of the Tertiary Industry (%)", min_value=0.0, max_value=100.0, format="%.1f")
                    coal_prop = st.number_input("Coal Proportion (%)", min_value=0.0, max_value=100.0, format="%.1f")
                
                submit_button = st.form_submit_button(label='Estimate Total Carbon Dioxide Emissions')

            # --- Results Display ---
            if submit_button:
                st.divider()
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.subheader("Result")
                    # Create DataFrame from inputs
                    input_data = pd.DataFrame({
                        'name': [province], 'year': [year], 'per_capita_gdp_yuan': [gdp],
                        'total_population_million': [population], 'urbanization_rate_percent': [urbanization],
                        'proportion_of_primary_industry_percent': [primary_ind], 'proportion_of_secondary_industry_percent': [secondary_ind],
                        'proportion_of_the_tertiary_industry_percent': [tertiary_ind], 'coal_proportion_percent': [coal_prop]
                    })
                    # Make prediction
                    prediction = regression_model.predict(input_data)[0]
                    st.metric(label=f"Estimated CO₂ Emissions for {province}", value=f"{prediction:,.2f} million tons")
                
                with res_col2:
                    st.subheader("Explanation")
                    
                    # Determine levels for each input and the prediction
                    gdp_level = get_level(gdp, 'per_capita_gdp_yuan')
                    pop_level = get_level(population, 'total_population_million')
                    urban_level = get_level(urbanization, 'urbanization_rate_percent')
                    primary_level = get_level(primary_ind, 'proportion_of_primary_industry_percent')
                    secondary_level = get_level(secondary_ind, 'proportion_of_secondary_industry_percent')
                    tertiary_level = get_level(tertiary_ind, 'proportion_of_the_tertiary_industry_percent')
                    coal_level = get_level(coal_prop, 'coal_proportion_percent')
                    prediction_level = get_level(prediction, 'total_carbon_dioxide_emissions_(_million_tons_)')

                    explanation_text = f"""
                    Based on the values you input for **{province}** in the year **{year}**:

                    - A GDP per capita of **{gdp:,.0f} yuan** is at **Level {gdp_level}**.
                    - A total population of **{population:,.2f} million** is at **Level {pop_level}**.
                    - An urbanization rate of **{urbanization:.1f}%** is at **Level {urban_level}**.
                    - A primary industry proportion of **{primary_ind:.1f}%** is at **Level {primary_level}**.
                    - A secondary industry proportion of **{secondary_ind:.1f}%** is at **Level {secondary_level}**.
                    - A tertiary industry proportion of **{tertiary_ind:.1f}%** is at **Level {tertiary_level}**.
                    - A coal proportion of **{coal_prop:.1f}%** is at **Level {coal_level}**.

                    ---
                    The model estimates that these factors would result in carbon dioxide emissions of **{prediction:,.2f} million tons**, which corresponds to **Level {prediction_level}** compared to historical data.
                    """
                    st.markdown(explanation_text)
                    st.caption("Level L: Low (below 25th percentile), M: Medium (between 25th-75th), H: High (above 75th percentile)")


    # --- ARIMA Forecast Tab ---
    with est_tab2:
        if not os.path.exists(MODEL_DIR) or source_df is None:
            st.error(f"ARIMA model directory '{MODEL_DIR}' or source data not found.")
        else:
            st.subheader("Select a province and the number of years to forecast its CO₂ emissions trend.")
            
            # Create a list of province names from the model filenames
            arima_province_names = sorted([
                f.replace('arima_model_', '').replace('.pkl', '').replace('_', ' ') 
                for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')
            ])
            
            fc_province = st.selectbox("Province", options=arima_province_names, key="forecast_province")
            fc_years = st.slider("Select number of years to forecast", 1, 5, 3, key="forecast_years")

            # Load the correct model and generate forecast
            safe_province_name = fc_province.replace(" ", "_")
            model_path = os.path.join(MODEL_DIR, f'arima_model_{safe_province_name}.pkl')
            arima_model = load_model(model_path)
            
            if arima_model:
                province_ts = source_df[source_df['name'] == fc_province][['year', 'total_carbon_dioxide_emissions_(_million_tons_)']].copy()
                province_ts = province_ts.set_index('year')

                forecast = arima_model.get_forecast(steps=fc_years)
                forecast_df = forecast.summary_frame(alpha=0.05)
                
                last_year = province_ts.index.max()
                forecast_df.index = range(last_year + 1, last_year + 1 + len(forecast_df))
                forecast_df.index.name = 'Year'

                # FIX: Cap negative predictions at zero
                forecast_df['mean'] = forecast_df['mean'].clip(lower=0)
                
                # Create Plotly figure
                fig = go.Figure()
                
                # Historical data trace
                fig.add_trace(go.Scatter(x=province_ts.index, y=province_ts.iloc[:, 0], mode='lines+markers', name='Historical Emissions', line=dict(color='royalblue')))
                
                # Forecast trace
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Forecast', line=dict(dash='dash', color='firebrick')))
                
                # Confidence interval trace
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_upper'], fill='tonexty', fillcolor='rgba(255, 82, 82, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='95% Confidence Interval', showlegend=True))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_lower'], fill='tonexty', fillcolor='rgba(255, 82, 82, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='Lower Bound', showlegend=False))
                
                fig.update_layout(
                    title=f"CO₂ Emissions Forecast for {fc_province}",
                    xaxis_title="Year",
                    yaxis_title="Emissions (Million Tons)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Forecasted Data")
                st.dataframe(forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].rename(columns={
                    'mean': 'Mean Forecast',
                    'mean_ci_lower': 'Lower 95% CI',
                    'mean_ci_upper': 'Upper 95% CI'
                }).style.format("{:,.2f}"))
            else:
                st.error(f"Could not load forecast model for {fc_province}.")