import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 2rem auto;
        max-width: 1200px;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 3rem 2rem;
        margin: -2rem -2rem 3rem -2rem;
        border-radius: 24px 24px 0 0;
        text-align: center;
        color: white;
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Section styling */
    .section-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
    }
    
    .section-title {
        display: flex;
        align-items: center;
        font-size: 1.5rem;
        font-weight: 600;
        color: #0E0F0F;
        margin-bottom: 1rem;
        padding-left: 1rem;
        border-left: 4px solid #4facfe;
    }

    
    /* Custom button styling */
    .predict-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 3rem !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        border-radius: 50px !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        margin: 2rem auto !important;
        display: block !important;
        width: fit-content !important;
    }
    
    .predict-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Result styling */
    .result-card {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(72, 187, 120, 0.3);
    }
    
    .result-text {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(0, 0, 0, 0.1);
        color: #CCFF66;
        font-style: italic;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-shadow: 0 0 5px #ccff66, 0 0 10px #ccff66;
    }

    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin: 1rem 0;
    }

    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Slider customization */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #4facfe 0%, #e2e8f0 0%);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #4facfe;
        box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('random_forest_obesity_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("❌ File model atau scaler tidak ditemukan. Pastikan file 'random_forest_obesity_model.pkl' dan 'scaler.pkl' ada di direktori yang sama.")
        st.stop()

model, scaler = load_models()

st.markdown("""
<div class="header-container">
    <h1 class="header-title">🏥 Prediksi Tingkat Obesitas</h1>
    <p class="header-subtitle">Masukkan data di bawah ini untuk memprediksi tingkat obesitas Anda</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">👤 Atribut Fisik</div>
    </div>
    """, unsafe_allow_html=True)
    
    age = st.slider("🎂 Usia (Tahun)", 10, 70, 25, help="Pilih usia Anda dalam tahun")
    height = st.slider("📏 Tinggi Badan (Meter)", 1.40, 2.00, 1.75, 0.01, help="Masukkan tinggi badan dalam meter")
    weight = st.slider("⚖️ Berat Badan (Kg)", 30.0, 200.0, 70.0, 0.5, help="Masukkan berat badan dalam kilogram")
    gender = st.selectbox("👥 Jenis Kelamin", ['Pria', 'Wanita'], help="Pilih jenis kelamin Anda")
    family_history = st.selectbox("👨‍👩‍👧‍👦 Riwayat obesitas dalam keluarga?", ['Ya', 'Tidak'], help="Apakah ada anggota keluarga yang mengalami obesitas?")

with col2:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🍽️ Kebiasaan Gaya Hidup</div>
    </div>
    """, unsafe_allow_html=True)
    
    favc = st.selectbox("🍔 Sering makan makanan berkalori tinggi (FAVC)?", ['Ya', 'Tidak'], help="Apakah Anda sering mengonsumsi makanan berkalori tinggi?")
    fcvc = st.slider("🥬 Frekuensi makan sayur (FCVC)", 1, 3, 2, help="Seberapa sering Anda makan sayuran? (1=Jarang, 3=Sering)")
    ncp = st.slider("🍽️ Jumlah makan utama per hari (NCP)", 1, 4, 3, help="Berapa kali Anda makan utama dalam sehari?")
    caec = st.selectbox("🍿 Makan di antara waktu makan utama (CAEC)?", ['Tidak', 'Kadang-kadang', 'Sering', 'Selalu'], help="Seberapa sering Anda ngemil di luar waktu makan?")
    smoke = st.selectbox("🚬 Apakah Anda merokok (SMOKE)?", ['Ya', 'Tidak'], help="Apakah Anda seorang perokok?")

st.markdown("<br>", unsafe_allow_html=True)
col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">💧 Aktivitas dan Konsumsi</div>
    </div>
    """, unsafe_allow_html=True)
    
    ch2o = st.slider("💧 Konsumsi air harian (Liter) (CH2O)", 1, 3, 2, help="Berapa liter air yang Anda minum per hari?")
    scc = st.selectbox("📊 Memantau konsumsi kalori (SCC)?", ['Ya', 'Tidak'], help="Apakah Anda memantau asupan kalori harian?")
    faf = st.slider("🏃‍♂️ Frekuensi aktivitas fisik per minggu (FAF)", 0, 3, 1, help="Berapa kali Anda berolahraga dalam seminggu?")

with col4:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📱 Gaya Hidup Digital & Transportasi</div>
    </div>
    """, unsafe_allow_html=True)
    
    tue = st.slider("📱 Waktu penggunaan gawai per hari (Jam) (TUE)", 0, 2, 1, help="Berapa jam Anda menggunakan gadget per hari?")
    calc = st.selectbox("🍷 Konsumsi alkohol (CALC)?", ['Tidak', 'Kadang-kadang', 'Sering', 'Selalu'], help="Seberapa sering Anda mengonsumsi alkohol?")
    mtrans = st.selectbox("🚗 Transportasi yang digunakan (MTRANS)", 
                         ['Mobil', 'Motor', 'Sepeda', 'Transportasi Umum', 'Jalan Kaki'], 
                         help="Apa mode transportasi utama yang Anda gunakan?")

st.markdown("<br><br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Prediksi Tingkat Obesitas", key="predict", help="Klik untuk memprediksi tingkat obesitas Anda")

if predict_clicked:
    with st.spinner('🔄 Sedang memproses prediksi...'):
        input_data = {
            'Age': age, 'Height': height, 'Weight': weight, 'FCVC': fcvc, 'NCP': ncp,
            'CH2O': ch2o, 'FAF': faf, 'TUE': tue,
            'Gender': 'Male' if gender == 'Pria' else 'Female',
            'family_history_with_overweight': 'yes' if family_history == 'Ya' else 'no',
            'FAVC': 'yes' if favc == 'Ya' else 'no',
            'CAEC': caec.replace('Sering', 'Frequently').replace('Kadang-kadang', 'Sometimes').replace('Selalu', 'Always').replace('Tidak', 'no'),
            'SMOKE': 'yes' if smoke == 'Ya' else 'no',
            'SCC': 'yes' if scc == 'Ya' else 'no',
            'CALC': calc.replace('Sering', 'Frequently').replace('Kadang-kadang', 'Sometimes').replace('Selalu', 'Always').replace('Tidak', 'no'),
            'MTRANS': mtrans.replace('Mobil', 'Automobile').replace('Motor', 'Motorbike').replace('Sepeda', 'Bike').replace('Transportasi Umum', 'Public_Transportation').replace('Jalan Kaki', 'Walking')
        }

        input_df_raw = pd.DataFrame([input_data])
        numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        input_df_raw[numerical_cols] = scaler.transform(input_df_raw[numerical_cols])

        categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        input_df_encoded = pd.get_dummies(input_df_raw, columns=categorical_cols)

        training_columns = [
            'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
            'Gender_Female', 'Gender_Male',
            'family_history_with_overweight_no', 'family_history_with_overweight_yes',
            'FAVC_no', 'FAVC_yes',
            'CAEC_Always', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no',
            'SMOKE_no', 'SMOKE_yes',
            'SCC_no', 'SCC_yes',
            'CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
            'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
            'MTRANS_Public_Transportation', 'MTRANS_Walking'
        ]

        input_final = input_df_encoded.reindex(columns=training_columns, fill_value=0)

        prediction = model.predict(input_final)
        prediction_proba = model.predict_proba(input_final)

        st.markdown("<br>", unsafe_allow_html=True)
        
        result = prediction[0].replace("_", " ").title()
        
        st.markdown(f"""
        <div class="result-card">
            <h2 class="result-text">✅ Hasil Prediksi: {result}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        
        st.markdown("### 📈 Probabilitas untuk Setiap Kategori")
        
        proba_data = pd.DataFrame({
            'Kategori': model.classes_,
            'Probabilitas': prediction_proba[0]
        })
        
        fig = px.bar(
            proba_data, 
            x='Kategori', 
            y='Probabilitas',
            color='Probabilitas',
            color_continuous_scale='Viridis',
            title="Distribusi Probabilitas Tingkat Obesitas"
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif"),
            title_font_size=16,
            showlegend=False
        )
        
        fig.update_traces(
            texttemplate='%{y:.1%}',
            textposition='outside'
        )
        
        st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="footer">
    Made with 💜 by Muhammad Satriya Pratama Manggala Kusuma - A11.2022.14225
</div>
""", unsafe_allow_html=True)
