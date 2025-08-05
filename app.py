import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="CO₂ Emissions Estimation",
    page_icon="🌍",
    layout="wide"
)

# --- Caching Functions for Loading Data & Models ---
@st.cache_data
def load_raw_data(file_path):
    """Loads the original, raw CSV data."""
    if not os.path.exists(file_path):
        return None
    try:
        # Based on the notebook, the delimiter is ';'
        df = pd.read_csv(file_path, delimiter=';')
        return df
    except Exception as e:
        st.error(f"Error loading raw data file ('{file_path}'): {e}")
        return None

@st.cache_data
def load_cleaned_data(file_path):
    """Loads and caches the cleaned CSV data."""
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading cleaned data file ('{file_path}'): {e}")
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

# --- Load Data and Models ---
MODEL_DIR = 'province_models_fixed_order' 
raw_df = load_raw_data('dataset.csv')
source_df = load_cleaned_data('cleaned_dataset.csv')
regression_model = load_model('gradient_boosting_model.pkl')

# --- Pre-calculate quantiles and define level function ---
quantiles = {}
if source_df is not None:
    cols_for_quantiles = [
        'per_capita_gdp_yuan', 'total_population_million', 'urbanization_rate_percent',
        'proportion_of_primary_industry_percent', 'proportion_of_secondary_industry_percent',
        'proportion_of_the_tertiary_industry_percent', 'coal_proportion_percent',
        'total_emissions' 
    ]
    # Update target column name to match cleaned file
    if 'total_carbon_dioxide_emissions_(_million_tons_)' in source_df.columns:
         source_df.rename(columns={'total_carbon_dioxide_emissions_(_million_tons_)': 'total_emissions'}, inplace=True)
    
    quantiles = source_df[cols_for_quantiles].quantile([0.25, 0.75]).to_dict()

def get_level(value, column_name):
    """Determines if a value is Low, Medium, or High based on pre-calculated dataset quantiles."""
    if not quantiles or column_name not in quantiles:
        return "" 
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
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

home_btn_type = "primary" if st.session_state.page == 'Home' else "secondary"
est_btn_type = "primary" if st.session_state.page == 'Estimate' else "secondary"

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
    home_tab1, home_tab2 = st.tabs(["Description", "Dataset Overview"])

    with home_tab1:
        st.header("Project Description")
        st.write("""
        This application provides two advanced tools for analyzing and predicting Carbon Dioxide (CO₂) emissions across various provinces in China. 
        It leverages historical data from 1999 to 2019 to build robust machine learning models.

        ### Key Features:

        **1. Estimation with Specific Variables (Multivariate Regression):**
        - This tool uses a **Gradient Boosting Regressor** model, which has been trained on the entire dataset across all provinces.
        - It learns the complex relationships between $CO_2$ emissions and various socio-economic factors like GDP, population, urbanization rate, and industrial structure.
        - This allows you to create detailed "what-if" scenarios to see how specific policy and economic changes might impact emissions.

        **2. Forecasting (Univariate Time-Series):**
        - This tool employs an **ARIMA (AutoRegressive Integrated Moving Average)** model.
        - A unique ARIMA model has been trained for each individual province, focusing solely on its historical emissions trend.
        - This provides a "baseline" forecast, showing where a province's emissions are headed if its historical momentum continues without major external changes.

        By combining these two approaches, users can gain a comprehensive understanding of both the underlying drivers of emissions and the likely future trends.
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
        st.header("Dataset Overview")
        
        if raw_df is None or source_df is None:
            st.error("One or more dataset files (`dataset.csv`, `cleaned_dataset.csv`) not found. Cannot display overview.")
        else:
            with st.expander("About Dataset", expanded=False):
                st.markdown("""
                ### Dataset Source
                - **Source**: [Data for: Spatial Characteristics and Future Forecasting of Carbon Dioxide Emissions in China: A Provincial-Level Analysis](https://data.mendeley.com/datasets/rp3f7mdjxz/1)
                - **Published Date**: 30 Jul 2024
                - **DOI**: [10.17632/rp3f7mdjxz.1](https://doi.org/10.17632/rp3f7mdjxz.1)
                
                ### General Description
                This is a panel dataset containing socio-economic and carbon dioxide emissions data from 31 provinces in China over a 21-year period (1999 to 2019). In total, there are 651 rows of data, which is the product of 31 provinces multiplied by 21 years of observations.

                ### Column Descriptions
                - **`Name`**: The name of the province in China.
                - **`Year`**: The year the data was recorded (1999-2019).
                - **`per capita gdp(yuan)`**: Per Capita GDP in Chinese Yuan.
                - **`total population(million)`**: Total Population in millions.
                - **`urbanization rate(%)`**: The percentage of the population living in urban areas.
                - **`proportion of primary industry(%)`**: The percentage contribution of the primary sector (e.g., agriculture) to GDP.
                - **`proportion of secondary industry(%)`**: The percentage contribution of the secondary sector (e.g., industry) to GDP.
                - **`proportion of the tertiary industry(%)`**: The percentage contribution of the tertiary sector (e.g., services) to GDP.
                - **`coal proportion(%)`**: The percentage of coal use in total energy consumption.
                - **`Total carbon dioxide emissions (million tons)`**: The total CO2 emissions in million tons. This is the **target variable**.
                """)

            with st.expander("1. Raw Dataset Preview", expanded=False):
                st.markdown("This is the original, unprocessed data as loaded from the source file. Note the inconsistent column names, missing values (`NaN`), and numerical data formatted as text with commas.")
                st.dataframe(raw_df.head(10))

            with st.expander("2. Data Preprocessing Steps", expanded=False):
                st.markdown("""
                The raw dataset required several cleaning and transformation steps to be suitable for machine learning. The following process, mirroring the analysis notebook, was applied:
                
                #### a. Column Name Standardization
                The original column names contained spaces, parentheses, and mixed casing. They were standardized to a consistent `snake_case` format for easier access.
                - **Example Before**: `per capita gdp(yuan)`
                - **Example After**: `per_capita_gdp_yuan`
                - The target column `total_carbon_dioxide_emissions_(_million_tons_)` was also renamed to `total_emissions`.
                
                #### b. Handling Missing Values
                The `name` and `year` columns had missing values (`NaN`) for certain rows. These were filled using the **forward-fill (`ffill`)** method. This method propagates the last valid observation forward, which is appropriate for this panel data where missing names belong to the province mentioned previously.
                
                #### c. Data Type Conversion
                Numerical columns were incorrectly loaded as text (`object`) because they used commas as decimal separators. Each of these columns was converted:
                1.  The comma (`,`) was replaced with a period (`.`).
                2.  The resulting string was converted to a floating-point number (`float`).
                
                #### d. Handling Extreme Outliers
                Boxplots revealed one extreme outlier in `proportion_of_primary_industry_percent` and one in `proportion_of_secondary_industry_percent`. These outliers could skew the model.
                - **Strategy**: Instead of removing the rows, the outlier values were replaced with the **average value for that specific province**, calculated from its other years of data. This preserves the data point while correcting the anomalous value.
                  - For **Beijing** (1999), the primary industry proportion of `0.00` was replaced with its provincial average of `1.03`.
                  - For **Shanghai** (1999), the secondary industry proportion of `98.90` was replaced with its provincial average of `51.99`.

                After these steps, the dataset was clean, complete, and ready for analysis.
                """)
            
            with st.expander("3. Processed Dataset Preview"):
                st.markdown("Below is a preview of the final, cleaned dataset used for all models and visualizations in this application. All data types are correct, and missing values/outliers have been handled.")
                st.dataframe(source_df.head(10))

            with st.expander("4. Data Visualizations"):
                numeric_cols = source_df.select_dtypes(include=np.number).drop(columns='year')

                st.subheader("Correlation Heatmap")
                st.markdown("The heatmap below shows the correlation coefficient between different numerical variables. A value close to 1 (bright color) indicates a strong positive correlation.")
                
                corr = numeric_cols.corr()
                fig_heatmap = go.Figure(data=go.Heatmap(
                                   z=corr.values,
                                   x=corr.index.values,
                                   y=corr.columns.values,
                                   colorscale='Viridis',
                                   colorbar=dict(title='Correlation')))
                fig_heatmap.update_layout(title='Correlation Matrix of Numerical Features', yaxis_autorange='reversed')
                st.plotly_chart(fig_heatmap, use_container_width=True)


                st.subheader("National $CO_2$ Emissions Trend (1999-2019)")
                st.markdown("This chart aggregates the emissions from all provinces to show the overall trend in China over two decades.")

                total_emissions_by_year = source_df.groupby('year')['total_emissions'].sum().reset_index()
                fig_total_trend = px.line(total_emissions_by_year, x='year', y='total_emissions',
                                          title='Total National $CO_2$ Emissions Over Time', markers=True,
                                          labels={'year': 'Year', 'total_emissions': 'Total Emissions (Million Tons)'})
                st.plotly_chart(fig_total_trend, use_container_width=True)


                st.subheader("$CO_2$ Emissions Trend per Province")
                st.markdown("This chart compares the emissions trends of all provinces. You can click on items in the legend to hide or show specific provinces.")
                
                fig_all_provinces = px.line(source_df, x='year', y='total_emissions', color='name',
                                   title='$CO_2$ Emissions for All Provinces (1999-2019)', markers=False,
                                   labels={'year': 'Year', 'total_emissions': 'Emissions (Million Tons)', 'name': 'Province'})
                
                fig_all_provinces.update_xaxes(
                    dtick=2,
                    tickangle=45
                )

                # fig_all_provinces.update_layout(legend=dict(
                #     orientation="h",
                #     yanchor="bottom",
                #     y=-0.4, 
                #     xanchor="right",
                #     x=1
                # ))
                st.plotly_chart(fig_all_provinces, use_container_width=True)

                st.subheader("Feature Distributions")
                st.markdown("These histograms show the distribution of each numerical feature in the dataset, helping to understand their range and common values.")
                
                cols_to_plot = numeric_cols.columns
                fig_hist = make_subplots(rows=(len(cols_to_plot) + 2) // 3, cols=3, subplot_titles=[col.replace("_", " ").title() for col in cols_to_plot])
                for i, col in enumerate(cols_to_plot):
                    row = i // 3 + 1
                    col_num = i % 3 + 1
                    fig_hist.add_trace(go.Histogram(x=source_df[col], name=col), row=row, col=col_num)
                
                fig_hist.update_layout(height=800, showlegend=False, title_text="Distributions of Numerical Features")
                st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================================================
# ESTIMATE PAGE
# ==========================================================================
elif st.session_state.page == "Estimate":
    est_tab1, est_tab2 = st.tabs(["Estimate with Specific Variable", "Forecast"])

    with est_tab1:
        if regression_model is None or source_df is None:
            st.error("Regression model or source data not found. Please check your files.")
        else:
            province_names = sorted(source_df['name'].unique())
            
            with st.form(key='prediction_form'):
                st.subheader("Input the values below to estimate total carbon dioxide emissions")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    province = st.selectbox("Province", options=province_names)
                    year = st.number_input("Year", min_value=2020, max_value=2050, value=2025)
                    gdp = st.number_input("GDP per Capita (yuan)", min_value=0)
                with col2:
                    population = st.number_input("Total Population (million)", min_value=0.0, format="%.2f")
                    urbanization = st.number_input("Urbanization Rate (%)", min_value=0.0, max_value=100.0, format="%.1f")
                    primary_ind = st.number_input("Proportion of Primary Industry (%)", min_value=0.0, max_value=100.0, format="%.1f")
                with col3:
                    secondary_ind = st.number_input("Proportion of Secondary Industry (%)", min_value=0.0, max_value=100.0, format="%.1f")
                    tertiary_ind = st.number_input("Proportion of Tertiary Industry (%)", min_value=0.0, max_value=100.0, format="%.1f")
                    coal_prop = st.number_input("Coal Proportion (%)", min_value=0.0, max_value=100.0, format="%.1f")
                
                submit_button = st.form_submit_button(label='Estimate Emissions')

            if submit_button:
                st.divider()
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.subheader("Result")
                    input_data_dict = {
                        'name': [province], 'year': [year], 'per_capita_gdp_yuan': [gdp],
                        'total_population_million': [population], 'urbanization_rate_percent': [urbanization],
                        'proportion_of_primary_industry_percent': [primary_ind], 
                        'proportion_of_secondary_industry_percent': [secondary_ind],
                        'proportion_of_the_tertiary_industry_percent': [tertiary_ind], 
                        'coal_proportion_percent': [coal_prop]
                    }
                    # Rename the columns in the prediction input to match the model's training columns
                    input_data_df = pd.DataFrame(input_data_dict)
                    
                    prediction = regression_model.predict(input_data_df)[0]
                    st.metric(label=f"Estimated CO₂ Emissions for {province}", value=f"{prediction:,.2f} million tons")
                
                with res_col2:
                    st.subheader("Explanation")
                    
                    gdp_level = get_level(gdp, 'per_capita_gdp_yuan')
                    pop_level = get_level(population, 'total_population_million')
                    urban_level = get_level(urbanization, 'urbanization_rate_percent')
                    primary_level = get_level(primary_ind, 'proportion_of_primary_industry_percent')
                    secondary_level = get_level(secondary_ind, 'proportion_of_secondary_industry_percent')
                    tertiary_level = get_level(tertiary_ind, 'proportion_of_the_tertiary_industry_percent')
                    coal_level = get_level(coal_prop, 'coal_proportion_percent')
                    prediction_level = get_level(prediction, 'total_emissions')

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

    with est_tab2:
        if not os.path.exists(MODEL_DIR) or source_df is None:
            st.error(f"ARIMA model directory '{MODEL_DIR}' or source data not found.")
        else:
            st.subheader("Select a province and the number of years to forecast its $CO_2$ emissions trend.")
            
            arima_province_names = sorted([
                f.replace('arima_model_', '').replace('.pkl', '').replace('_', ' ') 
                for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')
            ])
            
            fc_province = st.selectbox("Province", options=arima_province_names, key="forecast_province")
            fc_years = st.slider("Select number of years to forecast", 1, 5, 3, key="forecast_years")

            safe_province_name = fc_province.replace(" ", "_")
            model_path = os.path.join(MODEL_DIR, f'arima_model_{safe_province_name}.pkl')
            arima_model = load_model(model_path)
            
            if arima_model:
                province_ts = source_df[source_df['name'] == fc_province][['year', 'total_emissions']].copy().set_index('year')
                
                forecast = arima_model.get_forecast(steps=fc_years)
                forecast_df = forecast.summary_frame(alpha=0.05)
                
                last_year = province_ts.index.max()
                forecast_df.index = range(last_year + 1, last_year + 1 + len(forecast_df))
                
                forecast_df['mean'] = forecast_df['mean'].clip(lower=0)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=province_ts.index, y=province_ts.iloc[:, 0], mode='lines+markers', name='Historical Emissions', line=dict(color='royalblue')))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Forecast', line=dict(dash='dash', color='firebrick')))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_upper'], fill='tonexty', fillcolor='rgba(255, 82, 82, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='95% Confidence Interval', showlegend=True))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_lower'], fill='tonexty', fillcolor='rgba(255, 82, 82, 0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
                
                fig.update_layout(
                    title=f"$CO_2$ Emissions Forecast for {fc_province}",
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