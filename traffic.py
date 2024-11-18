import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

# Title and Header
st.title('Traffic Volume Predictor')
st.image('traffic_image.gif', use_column_width=True, caption="Predict traffic volume!")

st.write("This app uses multiple inputs to predict the volume of traffic.")

# Utility functions
@st.cache_resource
def load_pickle(file_name):
    with open(file_name, 'rb') as model_pickle:
        return pickle.load(model_pickle)

@st.cache_resource
def load_csv(file_name):
    return pd.read_csv(file_name)

# Load the model
reg_model = load_pickle('mapie_traffic.pickle')

# Load dataset for defaults
df = load_csv('Traffic_Volume.csv')
df['date_time'] = pd.to_datetime(df['date_time'], format='%m/%d/%y %H:%M')

# Extract day, month, and weekday
df['weekday'] = df['date_time'].dt.strftime('%A')
df['month'] = df['date_time'].dt.strftime('%B')
df['hour'] = df['date_time'].dt.hour

# Drop the original 'date_time' column
df = df.drop(columns=['date_time'])
sample_df = df.copy()

# Confidence Interval Slider
confidence_interval = st.slider('Set confidence interval', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

# Sidebar User Inputs
with st.sidebar.form("user_inputs_form"):
    st.header("User Input")
    st.image('traffic_sidebar.jpg', use_column_width=True)

    selected_model = st.selectbox('Select Model', options=['XGBoost Regressor', 'MAPIE Regressor'])

    with st.expander('Option 1: Upload CSV File'):
        input_CSV = st.file_uploader('Upload CSV')

    with st.expander('Option 2: Fill Out Form'):
        holiday_options = sample_df['holiday'].unique()
        input_holiday = st.selectbox('Holiday', options=holiday_options)
        
        input_temp = st.number_input('Temp (F)', value=sample_df['temp'].mean())
        input_rain = st.number_input('Rain (mm) in past hour', value=sample_df['rain_1h'].mean())
        input_snow = st.number_input('Snow (mm) in past hour', value=sample_df['snow_1h'].mean())
        input_clouds = st.number_input('Cloud coverage percent', value=sample_df['clouds_all'].mean())

        weather_options = sample_df['weather_main'].unique()
        input_weather = st.selectbox('Weather description', options=weather_options)

        day_options = sample_df['weekday'].unique()
        input_day = st.selectbox('Weekday', options=day_options)

        month_options = sample_df['month'].unique()
        input_month = st.selectbox('Month', options=month_options)
        input_time = st.number_input('Time (24 Hour)', value=sample_df['hour'].mean())

    submit_button = st.form_submit_button("Predict")

# Process User Inputs
if submit_button:
    st.write("Processing prediction...")

    try:
        # Handle CSV Upload
        if input_CSV:
            df = pd.read_csv(input_CSV)

            # Normalize column names
            df.columns = df.columns.str.strip().str.lower()
        else:
            # Use manually entered form data
            df = pd.DataFrame([{
                'holiday': input_holiday,
                'temp': input_temp,
                'rain_1h': input_rain,
                'snow_1h': input_snow,
                'clouds_all': input_clouds,
                'weather_main': input_weather,
                'weekday': input_day,
                'month': input_month,
                'hour': input_time
            }])

        # Encode categorical variables
        encoded_df = pd.get_dummies(df, columns=['holiday', 'weather_main', 'weekday', 'month'], drop_first=True)

        # Prediction logic with confidence intervals
        alpha = confidence_interval
        predictions, intervals = reg_model.predict(encoded_df, alpha=alpha)

        # Extract intervals
        lower_bounds = intervals[:, 0]
        upper_bounds = intervals[:, 1]

        # Add predictions and intervals to DataFrame
        encoded_df['predicted_volume'] = predictions
        encoded_df['lower_bound'] = lower_bounds
        encoded_df['upper_bound'] = upper_bounds

        # Display results
        st.metric(label="Predicted Traffic Volume", value=f"{predictions[0]:.2f}")
        st.write(f"Confidence Interval: {lower_bounds[0]:.2f} to {upper_bounds[0]:.2f}")
        st.write(encoded_df)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display insights
st.subheader("Sample Dataframe")
st.write(sample_df.tail(10))

st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                                  "Histogram of Residuals", 
                                  "Predicted Vs. Actual", 
                                  "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.png')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('distribution_plot.png')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Predicted Vs. Actual")
    st.image('scatter_plot.png')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('prediction_interval_coverage.png')
    st.caption("Range of predictions with confidence intervals.")
