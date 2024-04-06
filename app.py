import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('./model/model_titanic.pkl')

def main():
    st.title('Mon application Streamlit')

    st.write('Bienvenue ! Utilisez cette application pour faire des prédictions.')

    # Add input fields for user input
    sex = st.selectbox('Sexe', ['male', 'female'])
    age = st.slider('Âge', min_value=0, max_value=100, value=30, step=1)
    pclass = st.selectbox('Classe', [1, 2, 3])
    fare = st.number_input('Fare', value=50.0)
    embarked = st.selectbox('Port d\'embarquement', ['C', 'Q', 'S'])

    # Prepare input data with correct column order
    input_data = pd.DataFrame({'Sex': [sex], 'Age': [age], 'Pclass': [pclass], 'Fare': [fare], 'Embarked': [embarked], 'Parch': [0], 'SibSp': [0]}, columns=['Sex', 'Age', 'Pclass', 'Fare', 'Embarked', 'Parch', 'SibSp'])

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display prediction
    if prediction[0] == 0:
        st.write('Le passager est prédit comme non survivant.')
    else:
        st.write('Le passager est prédit comme survivant.')

    # Display prediction probability
    st.write(f'Probabilité de non survie: {prediction_proba[0][0]:.2f}')
    st.write(f'Probabilité de survie: {prediction_proba[0][1]:.2f}')

if __name__ == '__main__':
    main()
