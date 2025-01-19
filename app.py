import streamlit as st
import numpy as np
import joblib

# 1) Load the trained model
model = joblib.load("model.joblib")

st.title("Определитель типа питания")

st.write("""
Введите значения показателей микробиоты
""")
feature_list = ['Общая бактериальная масса',
 'Доля нормальной микробиоты',
 'Разнообразие микробиоты',
 'Общее количество  Bifidobacterium',
 'Общее количество Lactobacterium',
 'Bacteroides/fermicutes',
 'Дрожжи',
 'Условнопатогенная',
 'Патогенная флора']
foods = ['МЯСОЕДЫ', 'МОЛОКОЕДЫ', 'СЛАДКОЕЖКИ', 'ХЛЕБОЕДЫ', 'ОВОЩЕЕДЫ']

input_values = []
for feature_name in feature_list:
    value = st.number_input(feature_name, value=0.0)
    input_values.append(value)

# 3) When user clicks, do the prediction
if st.button("Рассчёт"):
    X_new = np.array([input_values])
    predicted_group = model.predict(X_new)[0]
    # st.write("Вы ввели:", input_values)
    st.write(f"**Ваша группа:** {foods[predicted_group].lower()}")
