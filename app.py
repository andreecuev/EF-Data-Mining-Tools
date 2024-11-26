#importar librerias
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model.joblib')
pipeline = joblib.load('pipeline.joblib')


#funcion para clasificar las plantas 
def classify(num):
    if num == 0:
        return 'No Survived'
    else:
        return 'Survived'

def main():
    #titulo
    st.title('Modelamiento de Supervivencia del Titanic')
    #titulo de sidebar
    st.sidebar.header('User Input Parameters')

    #funcion para poner los parametrso en el sidebar
    def user_input_parameters():
        pclas = st.sidebar.slider('Pclass', 1, 2, 3)
        sex = st.sidebar.selectbox('Sex', ['male', 'female'])
        age = st.sidebar.slider('Age', 0, 39, 1)
        sibsp = st.sidebar.slider('SibSp', 0, 1)
        parch = st.sidebar.slider('Parch', 0, 1)
        fare = st.sidebar.slider('Fare', 0.0,  513.0, 32.0)
        embarked = st.sidebar.selectbox('Embarked', ['C', 'Q', 'S'])
        data = {'Pclass': pclas,
                'Sex': sex,
                'Age': age,
                'SibSp': sibsp,
                'Parch': parch,
                'Fare': fare,
                'Embarked': embarked
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()


    st.subheader('User Input Parameters')
    st.write(df)

    if st.button('RUN'):
        prediction = model.predict(pipeline.transform(df))
        st.subheader('Prediction')
        st.write(classify(prediction[0]))


if __name__ == '__main__':
    main()
    