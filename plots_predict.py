import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class PlotPredict:
    def __init__(self, cleaned_titanic, predictions):
        self.cleaned_titanic = cleaned_titanic
        self.predictions = predictions

    def visualize_data(self, data):
        fig, ax = plt.subplots()
        data.plot(kind='bar', ax=ax)
        ax.set_title('Données utilisateur')
        ax.set_xlabel('Caractéristiques')
        ax.set_ylabel('Valeurs')
        st.pyplot(fig)

    def plot_survived_passengers(self):
        survived_passengers = self.cleaned_titanic[self.predictions == 1]
        survived_passengers_df = pd.DataFrame(survived_passengers)
        sns.scatterplot(data=survived_passengers_df, x='PassengerId', y='Survived', hue='Survived', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Survived')
        plt.title('Passagers survivants avec leur PassengerId')
        plt.show()

    def by_class(self, data):
        survived_passengers = data[self.predictions == 1]
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Survived', hue='Pclass', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Survived')
        plt.title('Passagers survivants avec leur PassengerId par classe')
        plt.show()
    
    def by_age(self, data):
        survived_passengers = data[self.predictions == 1]
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Survived', hue='Age', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Survived')
        plt.title('Passagers survivants avec leur PassengerId par âge')
        plt.show()

    def by_sex(self, data):
        survived_passengers = data[self.predictions == 1]
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Survived', hue='Sex', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Survived')
        plt.title('Passagers survivants avec leur PassengerId par sexe')
        plt.show()

    def by_fare(self, data):
        survived_passengers = data[self.predictions == 1]
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Survived', hue='Fare', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Survived')
        plt.title('Passagers survivants avec leur PassengerId par tarif')
        plt.show()

    def by_parch(self, data):
        survived_passengers = data[self.predictions == 1]
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Survived', hue='Parch', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Survived')
        plt.title('Passagers survivants avec leur PassengerId par Parch')
        plt.show()

    def by_sibsp(self, data):
        survived_passengers = data[self.predictions == 1]
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Survived', hue='Sibsp', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Survived')
        plt.title('Passagers survivants avec leur PassengerId par Sibsp')
        plt.show()

    def by_embarked(self, data):
        survived_passengers = data[self.predictions == 1]
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Survived', hue='Embarked', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Survived')
        plt.title('Passagers survivants avec leur PassengerId par port d\'embarquement')
        plt.show()
    
    def by_family_size(self, data):
        data['FamilySize'] = data['SibSp'] + data['Parch']
        survived_passengers = data[self.predictions == 1]
        sns.scatterplot(data=survived_passengers, x='PassengerId', y='Survived', hue='FamilySize', palette='deep', marker='o')
        plt.xlabel('PassengerId')
        plt.ylabel('Survived')
        plt.title('Passagers survivants avec leur PassengerId par taille de famille')
        plt.show()