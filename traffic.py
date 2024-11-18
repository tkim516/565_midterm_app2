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
df = df.drop(columns=['date_time', 'traffic_volume'])
sample_df = df.copy()

# Confidence Interval Slider
confidence_interval = st.slider('Set confidence interval', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

# Sidebar User Inputs
with st.sidebar.form("user_inputs_form"):
    st.header("User Input")
    st.image('traffic_sidebar.jpg', use_column_width=True)

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

st.subheader("Sample Dataframe")
st.write(sample_df.tail(4))

# Process User Inputs
if submit_button:
    try:
        # Handle CSV Upload
        if input_CSV:
            df = pd.read_csv(input_CSV)
            user_encoded_df = pd.get_dummies(df)
            sample_df_encoded = pd.get_dummies(sample_df)            
            missing_columns = [col for col in sample_df_encoded.columns if col not in user_encoded_df.columns]

            # Add missing columns to df1 and initialize them with None
            for col in missing_columns:
                user_encoded_df[col] = False
            
            missing_columns2 = [col for col in user_encoded_df.columns if col not in sample_df_encoded.columns]
            user_encoded_df = user_encoded_df[sample_df_encoded.columns]

        else:
            user_input = pd.DataFrame([{
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
            
            combined_df = pd.concat([sample_df, user_input], ignore_index=True)
            
            # Apply one-hot encoding
            encoded_combined_df = pd.get_dummies(combined_df)

            # Extract only the user row
            user_encoded_df = encoded_combined_df.tail()
        
        # Prediction logic with confidence intervals
        alpha = confidence_interval
        st.write(f'Alpha: {alpha}')
        predictions, intervals = reg_model.predict(user_encoded_df, alpha=alpha)

        # Extract intervals
        lower_bounds = np.maximum(intervals[:, 0], 0)
        upper_bounds = intervals[:, 1]

        # Add predictions and intervals to DataFrame
        user_encoded_df['predicted_volume'] = predictions
        user_encoded_df['lower_bound'] = lower_bounds
        user_encoded_df['upper_bound'] = upper_bounds

        columns_to_move = ['predicted_volume', 'lower_bound', 'upper_bound']
        remaining_columns = [col for col in user_encoded_df.columns if col not in columns_to_move]
        new_column_order = columns_to_move + remaining_columns

        # Reorder the DataFrame
        user_encoded_df = user_encoded_df[new_column_order]

        # Display the reordered DataFrame        
        user_encoded_df = user_encoded_df[new_column_order]

        # Highlight the last row if CSV is not uploaded
        if not input_CSV:
            st.metric(label="Predicted Traffic Volume", value=f"{predictions[0]:.2f}")
            st.write(f"With a {alpha * 100}% confidence interval:")
            st.subheader("User Inputs with Predictions and Intervals")
            # Apply styling to highlight the last row in light blue
            styled_df = user_encoded_df.style.apply(
                lambda x: ['background-color: yellow' if i == len(x) - 1 else '' for i in range(len(x))], 
                axis=0
            )
            
            # Use `st.dataframe` for better interaction
            st.dataframe(styled_df)
        else:
            # If CSV is uploaded, simply display the DataFrame
            st.write(f"With a {alpha * 100}% confidence interval:")
            st.subheader("User Dataframe with Predictions and Intervals")
            st.write(user_encoded_df)

        

    except Exception as err:
        st.error(f"An error occurred: {err}")

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
