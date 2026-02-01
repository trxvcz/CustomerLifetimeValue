import joblib
import pandas as pd
import streamlit as st


# 1. Ładowanie modelu
# Używamy cache, żeby nie ładować modelu przy każdym kliknięciu
@st.cache_resource
def load_model():
    return joblib.load("models/RandomForestRegressor.joblib")

artifacts = load_model()
model = artifacts["model"]
encoder = artifacts["encoder"]

# 2. Tytuł i opis
st.title("Przewidywanie Wartości Klienta (CLV)")
st.write("Wprowadź dane klienta, aby oszacować jego wartość życiową.")

# 3. Formularz dla użytkownika (dostosuj do swoich kolumn!)
# To musi pasować do Twojego X_train przed transformacją
col1, col2 = st.columns(2)

with col1:
    monthly_premium = st.number_input("Miesięczna Składka ($)", value=100)
    total_claim = st.number_input("Całkowita Kwota Roszczeń ($)", value=0)
    income = st.number_input("Roczny Dochód ($)", value=50000)

with col2:
    state = st.selectbox("Stan", ["Washington", "California", "Arizona", "Oregon", "Nevada"])
    policy = st.selectbox("Rodzaj Polisy", ["Personal L3", "Personal L2", "Personal L1", "Corporate L3"])
    coverage = st.selectbox("Zakres", ["Basic", "Extended", "Premium"])

# 4. Przycisk "Oblicz"
if st.button("Oblicz CLV"):
    # Tworzymy DataFrame z danych wejściowych
    # UWAGA: Nazwy kolumn muszą być IDENTYCZNE jak w Twoim treningowym DataFrame
    input_data = pd.DataFrame({
        'State': [state],
        'Customer Lifetime Value': [0], # Placeholder, nieużywany
        'Response': ['No'], # Wartość domyślna
        'Coverage': [coverage],
        'Education': ['Bachelor'], # Wartość domyślna lub dodaj selectbox
        'Effective To Date': ['1/1/11'],
        'EmploymentStatus': ['Employed'],
        'Gender': ['F'],
        'Income': [income],
        'Location Code': ['Suburban'],
        'Marital Status': ['Married'],
        'Monthly Premium Auto': [monthly_premium],
        'Months Since Last Claim': [10],
        'Months Since Policy Inception': [50],
        'Number of Open Complaints': [0],
        'Number of Policies': [1], # Ważne! Możesz dodać slider
        'Policy Type': ['Personal Auto'],
        'Policy': [policy],
        'Renew Offer Type': ['Offer1'],
        'Sales Channel': ['Agent'],
        'Total Claim Amount': [total_claim],
        'Vehicle Class': ['Four-Door Car'],
        'Vehicle Size': ['Medsize']
    })

    # Tutaj musisz odtworzyć kroki preprocessingu z notebooka
    # 1. Usuwanie kolumn (jeśli usuwałeś w notebooku)
    input_data = input_data.drop(columns=['Customer Lifetime Value']) # Np. to

    # 2. Transformacja (OneHotEncoder)
    input_data_transformed = encoder.transform(input_data)

    # 3. Predykcja
    prediction = model.predict(input_data_transformed)

    # 4. Wynik
    st.success(f"Przewidywana Wartość Klienta: ${prediction[0]:,.2f}")