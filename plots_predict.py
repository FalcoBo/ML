import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Class to plot the predictions

class PlotPredict:
    def __init__(self, cleaned_titanic, predictions):
        self.cleaned_titanic = cleaned_titanic
        self.predictions = predictions

    # Method to visualize the data
    def visualize_data(self, data):
        fig, ax = plt.subplots()
        data.plot(kind='bar', ax=ax)
        ax.set_title('Données utilisateur')
        ax.set_xlabel('Caractéristiques')
        ax.set_ylabel('Valeurs')
        st.pyplot(fig)

    # Method to plot the survived passengers with their PassengerId
    def plot_survived_passengers(self):
        survived_passengers = self.cleaned_titanic[self.predictions == 1]
        survived_passengers_df = pd.DataFrame(survived_passengers)
        sns.scatterplot(data=survived_passengers_df, x='PassengerId', y='Survived', hue='Survived', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Survived')
        plt.title('Passagers survivants avec leur PassengerId')
        plt.show()

    # Method to plot the survived passengers with their PassengerId by class
    def by_class(self, data):
        if 'Survived' in data.columns:
            survived_passengers = data[data['Survived'] == 1]
        else:
            survived_passengers = data
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Pclass', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Classe')
        plt.title('Passagers survivants avec leur PassengerId par classe')
        plt.show()

    # Method to plot the survived passengers with their PassengerId by age
    def by_age(self, data):
        if 'Survived' in data.columns:
            survived_passengers = data[data['Survived'] == 1]
        else:
            survived_passengers = data
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Age', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Âge')
        plt.title('Passagers survivants avec leur PassengerId par âge')
        plt.show()

    # Method to plot the survived passengers with their PassengerId by sex
    def by_sex(self, data):
        if 'Survived' in data.columns:
            survived_passengers = data[data['Survived'] == 1]
        else:
            survived_passengers = data
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Sex', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Sexe')
        plt.title('Passagers survivants avec leur PassengerId par sexe')
        plt.show()

    # Method to plot the survived passengers with their PassengerId by fare
    def by_fare(self, data):
        if 'Survived' in data.columns:
            survived_passengers = data[data['Survived'] == 1]
        else:
            survived_passengers = data
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Fare', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Tarif')
        plt.title('Passagers survivants avec leur PassengerId par tarif')
        plt.show()

    # Method to plot the survived passengers with their PassengerId by Parch
    def by_parch(self, data):
        if 'Survived' in data.columns:
            survived_passengers = data[data['Survived'] == 1]
        else:
            survived_passengers = data
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Parch', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Parch')
        plt.title('Passagers survivants avec leur PassengerId par Parch')
        plt.show()

    # Method to plot the survived passengers with their PassengerId by Sibsp
    def by_Sibsp(self, data):
        if 'Survived' in data.columns:
            survived_passengers = data[data['Survived'] == 1]
        else:
            survived_passengers = data
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Sibsp', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Sibsp')
        plt.title('Passagers survivants avec leur PassengerId par Sibsp')
        plt.show()

    # Method to plot the survived passengers with their PassengerId by embarked
    def by_embarked(self, data):
        if 'Survived' in data.columns:
            survived_passengers = data[data['Survived'] == 1]
        else:
            survived_passengers = data
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Embarked', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Embarked')
        plt.title('Passagers survivants avec leur PassengerId par port d\'embarquement')
        plt.show()

    # Method to plot the survived passengers with their PassengerId by family size
    def by_family_size(self, data):
        if 'Survived' in data.columns:
            data['FamilySize'] = data['Sibsp'] + data['Parch']
            survived_passengers = data[data['Survived'] == 1]
        else:
            data['FamilySize'] = data['Sibsp'] + data['Parch']
            survived_passengers = data
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='FamilySize', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Taille de famille')
        plt.title('Passagers survivants avec leur PassengerId par taille de famille')
        plt.show()
