import streamlit as st
import pandas as pd
import joblib

# Завантаження моделі
model_path = "models/aussie_rain.joblib"
loaded_model = joblib.load(model_path)

# Отримання компонентів з завантаженого словника
model = loaded_model['model']
imputer = loaded_model['imputer']
scaler = loaded_model['scaler']
encoder = loaded_model['encoder']
numeric_cols = loaded_model['numeric_cols']
categorical_cols = loaded_model['categorical_cols']

# Завантаження даних для отримання міст
data_path = "data/weatherAUS.csv"
data = pd.read_csv(data_path)

# Заголовок вашого застосунку
st.title("Прогноз дощу в Австралії")

# Створити колонки для вибору даних
col1, col2 = st.columns(2)

# Групування вибору даних у колонки
with col1:
    date = st.date_input("Виберіть дату")
    min_temp = st.slider("Мінімальна температура (°C)", float(data['MinTemp'].min()), float(data['MinTemp'].max()), float(data['MinTemp'].mean()))

with col2:
    location = st.selectbox("Виберіть локацію", data['Location'].unique())
    max_temp = st.slider("Максимальна температура (°C)", float(data['MaxTemp'].min()), float(data['MaxTemp'].max()), float(data['MaxTemp'].mean()))

# Решта компонентів залишаємо як є
rainfall = st.slider("Кількість опадів (мм)", float(data['Rainfall'].min()), float(data['Rainfall'].max()), float(data['Rainfall'].mean()))
evaporation = st.slider("Випаровування (мм)", float(data['Evaporation'].min()), float(data['Evaporation'].max()), float(data['Evaporation'].mean()))
sunshine = st.slider("Кількість сонячних годин", float(data['Sunshine'].min()), float(data['Sunshine'].max()), float(data['Sunshine'].mean()))
wind_gust_dir = st.selectbox("Напрямок сильного вітру", data['WindGustDir'].dropna().unique())
wind_gust_speed = st.slider("Швидкість сильного вітру (км/год)", float(data['WindGustSpeed'].min()), float(data['WindGustSpeed'].max()), float(data['WindGustSpeed'].mean()))

# Створити колонки для вітру, вологості, тиску, хмарності та температури о 9 та 15 
col3, col4 = st.columns(2)

with col3:
    wind_dir_9am = st.selectbox("Напрямок вітру о 9:00", data['WindDir9am'].dropna().unique())
    wind_speed_9am = st.slider("Швидкість вітру о 9:00 (км/год)", float(data['WindSpeed9am'].min()), float(data['WindSpeed9am'].max()), float(data['WindSpeed9am'].mean()))
    humidity_9am = st.slider("Вологість о 9:00 (%)", float(data['Humidity9am'].min()), float(data['Humidity9am'].max()), float(data['Humidity9am'].mean()))
    pressure_9am = st.slider("Тиск о 9:00 (hPa)", float(data['Pressure9am'].min()), float(data['Pressure9am'].max()), float(data['Pressure9am'].mean()))
    cloud_9am = st.slider("Хмарність о 9:00 (октави)", min_value=0, max_value=10, value=5, step=1)
    temp_9am = st.slider("Температура о 9:00 (°C)", float(data['Temp9am'].min()), float(data['Temp9am'].max()), float(data['Temp9am'].mean()))

with col4:
    wind_dir_3pm = st.selectbox("Напрямок вітру о 15:00", data['WindDir3pm'].dropna().unique())
    wind_speed_3pm = st.slider("Швидкість вітру о 15:00 (км/год)", float(data['WindSpeed3pm'].min()), float(data['WindSpeed3pm'].max()), float(data['WindSpeed3pm'].mean()))
    humidity_3pm = st.slider("Вологість о 15:00 (%)", float(data['Humidity3pm'].min()), float(data['Humidity3pm'].max()), float(data['Humidity3pm'].mean()))
    pressure_3pm = st.slider("Тиск о 15:00 (hPa)", float(data['Pressure3pm'].min()), float(data['Pressure3pm'].max()), float(data['Pressure3pm'].mean()))
    cloud_3pm = st.slider("Хмарність о 15:00 (октави)", min_value=0, max_value=10, value=5, step=1)
    temp_3pm = st.slider("Температура о 15:00 (°C)", float(data['Temp3pm'].min()), float(data['Temp3pm'].max()), float(data['Temp3pm'].mean()))

rain_today = st.selectbox("Чи був дощ сьогодні?", ["Yes", "No"])

# Створення DataFrame для передбачення
input_data = {
    "Location": location,
    "MinTemp": min_temp,
    "MaxTemp": max_temp,
    "Rainfall": rainfall,
    "Evaporation": evaporation,
    "Sunshine": sunshine,
    "WindGustDir": wind_gust_dir,
    "WindGustSpeed": wind_gust_speed,
    "WindDir9am": wind_dir_9am,
    "WindDir3pm": wind_dir_3pm,
    "WindSpeed9am": wind_speed_9am,
    "WindSpeed3pm": wind_speed_3pm,
    "Humidity9am": humidity_9am,
    "Humidity3pm": humidity_3pm,
    "Pressure9am": pressure_9am,
    "Pressure3pm": pressure_3pm,
    "Cloud9am": cloud_9am,
    "Cloud3pm": cloud_3pm,
    "Temp9am": temp_9am,
    "Temp3pm": temp_3pm,
    "RainToday": rain_today
}

input_df = pd.DataFrame([input_data])

# Підготовка даних для передбачення
# 1. Заповнення пропущених значень
input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])

# 2. Масштабування числових змінних
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# 3. Кодування категоріальних змінних з використанням збереженого encoder
input_encoded = encoder.transform(input_df[categorical_cols])

# Перевірка, чи є input_encoded розрідженим
if hasattr(input_encoded, "toarray"):
    input_encoded = input_encoded.toarray()

# Створення DataFrame з закодованих категоріальних змінних
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_feature_names, index=input_df.index)

# Об'єднання числових та закодованих категоріальних змінних
final_input = pd.concat([input_df[numeric_cols], input_encoded_df], axis=1)

# **Переконатися, що 'input_cols' містить числові колонки та закодовані категоріальні колонки**
# Для цього ми ігноруємо 'input_cols' з моделі і створюємо його власноруч
input_cols_reconstructed = numeric_cols + list(encoded_feature_names)

# Забезпечення відповідності стовпців з навчальними даними
final_input = final_input.reindex(columns=input_cols_reconstructed, fill_value=0)

# Логування проміжних результатів для відлагодження
#st.write("Numerical Data after Imputation and Scaling:")
#st.write(input_df[numeric_cols])

#st.write("Encoded Categorical Data:")
#st.write(input_encoded_df)

#st.write("Final Input for Prediction:")
#st.write(final_input)

#st.write("Model expects the following input columns:")
#st.write(input_cols_reconstructed)

# Прогнозування
if st.button("Передбачити"):
    if not final_input.empty:
        try:
            prediction = model.predict(final_input)
            st.write(f"Прогноз: {'Так, буде дощ' if prediction[0] == 1 else 'Ні, дощу не буде'}")
        except ValueError as e:
            st.error(f"Помилка під час прогнозування: {e}")
    else:
        st.write("Будь ласка, введіть значення для прогнозу")

# Додатково показати приклад з датасету
if st.checkbox("Показати приклад даних"):
    st.write(data.head())
