import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="CO₂ Emissions Forecaster",
    page_icon="📈",
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
    """Loads a single .pkl model file."""
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
st.title("📈 CO₂ Emissions: Forecast & Performance Dashboard")

# --- Load Data and Find Models ---
# IMPORTANT: Assumes your .pkl files are in a sub-directory named 'province_models_fixed_order'
MODEL_DIR = 'province_models_fixed_order' 
source_df = load_source_data('cleaned_dataset.csv')

if source_df is None or not os.path.exists(MODEL_DIR):
    st.error(f"Required files not found. Please make sure `cleaned_dataset.csv` and the `{MODEL_DIR}` directory exist in the same folder as this app.")
else:
    # --- Sidebar for User Controls ---
    st.sidebar.header("Controls")
    province_names = sorted([f.replace('arima_model_', '').replace('.pkl', '').replace('_', ' ') for f in os.listdir(MODEL_DIR)])
    selected_province = st.sidebar.selectbox("Select a Province", options=province_names)
    
    # --- Prepare file paths and data for the selected province ---
    safe_province_name = selected_province.replace(" ", "_")
    model_path = os.path.join(MODEL_DIR, f'arima_model_{safe_province_name}.pkl')
    
    final_model = load_model(model_path)
    province_ts = source_df[source_df['name'] == selected_province][['year', 'total_carbon_dioxide_emissions_(_million_tons_)']].copy()
    province_ts = province_ts.set_index('year')

    if final_model is None:
        st.error(f"Could not load the model for {selected_province}. Check if the file exists at: {model_path}")
    else:
        # --- Create Tabs for each visualization ---
        tab1, tab2 = st.tabs(["Final Forecast", "Model Performance Evaluation"])

        # ==========================================================================
        # Tab 1: Final Forecast
        # ==========================================================================
        with tab1:
            st.header(f"Official Forecast for {selected_province}")
            st.markdown("This chart uses the final, pre-trained model (trained on all historical data from 1999-2019) to project future emissions.")
            
            forecast_years = st.slider("Select number of years to forecast", 5, 20, 10, key="final_fc")
            
            # Generate forecast from the final model
            forecast = final_model.get_forecast(steps=forecast_years)
            forecast_df = forecast.summary_frame(alpha=0.05)
            
            # Create correct index for the forecast
            last_year = province_ts.index.max()
            forecast_df.index = range(last_year + 1, last_year + 1 + len(forecast_df))
            forecast_df.index.name = 'Year'

            # Create Plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=province_ts.index, y=province_ts.iloc[:, 0], mode='lines', name='Historical Emissions'))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Forecasted Emissions', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_upper'], fill='tonexty', mode='none', name='95% Confidence Interval', fillcolor='rgba(255, 75, 75, 0.2)'))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_lower'], fill='tonexty', mode='none', showlegend=False, fillcolor='rgba(255, 75, 75, 0.2)'))

            fig.update_layout(title=f"Final Forecast for {selected_province}", xaxis_title="Year", yaxis_title="Emissions (Million Tons)")
            st.plotly_chart(fig, use_container_width=True)

        # ==========================================================================
        # Tab 2: Model Performance Evaluation
        # ==========================================================================
        with tab2:
            st.header(f"Performance Evaluation for {selected_province}")
            st.markdown("This chart shows how the model would have performed on the last 3 years of data (2017-2019). The model is trained only on data up to 2016 and then used to forecast, allowing us to compare its prediction to the known, actual values.")

            # Split data into training and test sets
            train_data = province_ts.iloc[:-3]
            test_data = province_ts.iloc[-3:]

            # Temporarily train a model on the training data only
            # This is done live for demonstration purposes
            from statsmodels.tsa.arima.model import ARIMA
            eval_model = ARIMA(train_data, order=(1, 1, 1))
            eval_model_fit = eval_model.fit()

            # Forecast for 8 steps: 3 for the test period + 5 future years
            forecast_steps = len(test_data) + 5
            eval_forecast = eval_model_fit.get_forecast(steps=forecast_steps)
            eval_forecast_df = eval_forecast.summary_frame(alpha=0.05)
            
            # Create a proper index for the forecast
            last_train_year = train_data.index.max()
            eval_forecast_df.index = range(last_train_year + 1, last_train_year + 1 + forecast_steps)
            eval_forecast_df.index.name = 'Year'

            # Create the plot
            fig_eval = go.Figure()
            fig_eval.add_trace(go.Scatter(x=train_data.index, y=train_data.iloc[:, 0], mode='lines', name='Historical (Train)'))
            fig_eval.add_trace(go.Scatter(x=test_data.index, y=test_data.iloc[:, 0], mode='lines+markers', name='Actual (Test)', line=dict(color='green')))
            fig_eval.add_trace(go.Scatter(x=eval_forecast_df.index, y=eval_forecast_df['mean'], mode='lines', name='Forecast', line=dict(dash='dash', color='red')))
            
            fig_eval.update_layout(title=f"Model Evaluation for {selected_province}", xaxis_title="Year", yaxis_title="Emissions (Million Tons)")
            st.plotly_chart(fig_eval, use_container_width=True)
