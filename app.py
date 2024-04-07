import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from plots_predict import PlotPredict

# Load the model
model = joblib.load('./model/SVC.pkl')

# Load the cleaned Titanic dataset
cleaned_titanic = pd.read_csv('./data/cleaned_titanic.csv').drop('Survived', axis=1)

# Function to display the main page
def main():
    st.title('Prédire les Survivants du Titanic')
    st.write('Veuillez remplir les informations suivantes :')

    plots = PlotPredict(cleaned_titanic, model.predict(cleaned_titanic))

    sex = st.radio('Sexe', ['male', 'female'])
    age = st.slider('Âge', min_value=0, max_value=100, value=30, step=1)
    pclass = st.selectbox('Classe', [1, 2, 3])
    fare = st.number_input('Fare', value=50.0)
    parch = st.number_input('Parch', value=0)
    sibsp = st.number_input('SibSp', value=0)
    embarked = st.radio('Port d\'embarquement', [0, 1, 2])

    prediction = model.predict(cleaned_titanic)

    # Display the input data
    df_input_data = pd.DataFrame({'Sex': [sex], 'Age': [age], 'Pclass': [pclass], 'Fare': [fare], 'Parch':[parch], 'Sibsp':[sibsp] , 'Embarked': [embarked]})
    st.write('Données utilisateur :')
    st.write(df_input_data)

    st.write('Visualisation en temps réel des données utilisateur :')
    plots.visualize_data(df_input_data)

    if prediction[0] == 0:
        st.write('Le passager est prédit comme non survivant.')
    else:
        st.write('Le passager est prédit comme survivant.')

    # Plot survived passengers with PassengerId
    # st.write('Visualisation des passagers survivants avec leur PassengerId :')
    # plots.plot_survived_passengers()

if __name__ == '__main__':
    main()