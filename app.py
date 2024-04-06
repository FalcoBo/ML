import streamlit as st
import joblib

model = joblib.load('./model/model_titanic.pkl')

def main():
    st.title('Mon application Streamlit')

    st.write('Bienvenue ! Utilisez cette application pour faire des prédictions.')


if __name__ == '__main__':
    main()