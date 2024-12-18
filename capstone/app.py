import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import os

# Define the input fields for the app
def user_input_features():
    age = st.number_input('Age', min_value=1, max_value=120, value=54)
    sex = st.selectbox('Sex', ('M', 'F'))
    chest_pain_type = st.selectbox('Chest Pain Type', ('TA', 'ATA', 'NAP', 'ASY'))
    resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=200, value=125)
    cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, value=224)
    fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1))
    resting_ecg = st.selectbox('Resting ECG', ('Normal', 'ST', 'LVH'))
    max_hr = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=122)
    exercise_angina = st.selectbox('Exercise-Induced Angina', ('Y', 'N'))
    oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=2.0)
    st_slope = st.selectbox('ST Slope', ('Up', 'Flat', 'Down'))

    data = {
        'age': age,
        'sex': sex,
        'chestpaintype': chest_pain_type,
        'restingbp': resting_bp,
        'cholesterol': cholesterol,
        'fastingbs': fasting_bs,
        'restingecg': resting_ecg,
        'maxhr': max_hr,
        'exerciseangina': exercise_angina,
        'oldpeak': oldpeak,
        'st_slope': st_slope
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Main function to run the app
def main():
    st.title('Heart Disease Prediction App')
    st.write("""
    This app predicts whether a patient has heart disease based on various health metrics and demographic information.
    """)

    # Display an image
    st.image('images/heart_image.jpeg', caption='Heart Disease Prediction', width=400)

    input_df = user_input_features()

    # Convert categorical features to 'category' dtype
    for column in input_df.select_dtypes(include=['object']).columns:
        input_df[column] = input_df[column].astype('category')

    # Load the trained model
    model_directory = 'model'
    model_filename = 'best_xgboost_model.joblib'
    model_path = os.path.join(model_directory, model_filename)
    model = joblib.load(model_path)

    # Make predictions
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    st.subheader('Prediction')
    heart_disease = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f'Heart Disease: {heart_disease}')

    st.subheader('Prediction Probability')
    st.write(f'Probability of Heart Disease: {prediction_proba[0]:.2f}')

if __name__ == '__main__':
    main()