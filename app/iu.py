import streamlit as st
import requests
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Suscripción Bancaria",
    page_icon="🏦",
    layout="wide"
)

# Título y descripción
st.title("🏦 Predicción de Suscripción a Depósito a Plazo Fijo")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning para predecir si un cliente bancario 
suscribirá un depósito a plazo fijo basándose en sus características personales y de contacto.
""")

# URL de la API
API_URL = st.sidebar.text_input("URL de la API", "http://localhost:8000")

# VERIFICAR CONEXIÓN A LA API
st.sidebar.markdown("---")
st.sidebar.subheader("Estado de la API")

try:
    health_response = requests.get(f"{API_URL}/", timeout=2)  # FIX: /health → /
    if health_response.status_code == 200:
        health_data = health_response.json()
        st.sidebar.success("✅ API Conectada")
        st.sidebar.json(health_data)
    else:
        st.sidebar.error("❌ API no responde correctamente")
except Exception as e:
    st.sidebar.error(f"❌ No se puede conectar a la API: {str(e)}")

# CREAR PESTAÑAS
tab1, tab2 = st.tabs(["📝 Predicción Individual", "📊 Información del Modelo"])

with tab1:
    st.header("Información del Cliente")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Datos Personales")
        age = st.number_input("Edad", min_value=18, max_value=100, value=35)

        job = st.selectbox("Tipo de Trabajo", [
            "admin.", "blue-collar", "entrepreneur", "housemaid",
            "management", "retired", "self-employed", "services",
            "student", "technician", "unemployed", "unknown"
        ])

        marital = st.selectbox("Estado Civil", [
            "married", "single", "divorced", "unknown"
        ])

        education = st.selectbox("Nivel Educativo", [
            "basic.4y", "basic.6y", "basic.9y", "high.school",
            "illiterate", "professional.course", "university.degree", "unknown"
        ])

        default = st.selectbox("¿Tiene Crédito en Default?", ["no", "yes", "unknown"])  # FIX: campo faltante
        housing = st.selectbox("¿Tiene Crédito Hipotecario?", ["yes", "no", "unknown"])
        loan = st.selectbox("¿Tiene Préstamo Personal?", ["yes", "no", "unknown"])

    with col2:
        st.subheader("Datos de Contacto")
        contact = st.selectbox("Tipo de Comunicación", ["cellular", "telephone"])

        month = st.selectbox("Mes del Último Contacto", [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ])

        day_of_week = st.selectbox("Día de la Semana", [
            "mon", "tue", "wed", "thu", "fri"
        ])

        duration = st.number_input("Duración del Último Contacto (segundos)",
                                   min_value=0, value=200)

        campaign = st.number_input("Número de Contactos (Campaña Actual)",
                                   min_value=1, value=2)

        previous = st.number_input("Número de Contactos (Campañas Anteriores)",
                                   min_value=0, value=0)

        poutcome = st.selectbox("Resultado Campaña Anterior", [
            "nonexistent", "failure", "success"
        ])

        contacted_before = st.selectbox("¿Contactado Anteriormente?", ["no", "yes"])

    with col3:
        st.subheader("Indicadores Económicos")
        emp_var_rate = st.number_input("Tasa de Variación de Empleo",
                                       value=1.1, format="%.2f")

        cons_price_idx = st.number_input("Índice de Precios al Consumidor",
                                         value=93.994, format="%.3f")

        cons_conf_idx = st.number_input("Índice de Confianza del Consumidor",
                                        value=-36.4, format="%.1f")

        euribor3m = st.number_input("Tasa Euribor 3 meses",
                                    value=4.857, format="%.3f")

        nr_employed = st.number_input("Número de Empleados",
                                      value=5191.0, format="%.1f")

    # BOTÓN PARA REALIZAR PREDICCIÓN
    st.markdown("---")
    if st.button("🔮 Realizar Predicción", type="primary", use_container_width=True):

        payload = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,          # FIX: campo agregado
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day_of_week": day_of_week,
            "duration": duration,
            "campaign": campaign,
            "previous": previous,
            "poutcome": poutcome,
            "emp_var_rate": emp_var_rate,
            "cons_price_idx": cons_price_idx,
            "cons_conf_idx": cons_conf_idx,
            "euribor3m": euribor3m,
            "nr_employed": nr_employed,
            "contacted_before": contacted_before
        }

        try:
            with st.spinner("Consultando el modelo..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()

                st.success("✅ Predicción realizada exitosamente")

                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.markdown("### 🎯 Resultado de la Predicción")
                    prediction = result["prediction"]

                    if prediction == "yes":
                        st.success("### ✅ El Cliente SUSCRIBIRÁ el depósito")
                    else:
                        st.error("### ❌ El cliente NO suscribirá el depósito")

                    st.markdown("### 📋 Información del Modelo")
                    st.json(result["model_info"])

                with res_col2:
                    st.markdown("### 📊 Probabilidades")

                    probabilities = result["probability"]

                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(probabilities.keys()),
                            y=list(probabilities.values()),
                            text=[f"{v:.2%}" for v in probabilities.values()],
                            textposition="auto",
                            marker_color=["#ff4b4b" if k == "no" else "#00cc96" for k in probabilities.keys()]
                        )
                    ])

                    fig.update_layout(
                        title="Probabilidad de Suscripción",
                        xaxis_title="Clase",
                        yaxis_title="Probabilidad",
                        yaxis=dict(tickformat=".0%"),  # FIX: Tickformat → tickformat
                        height=300
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    prob_col1, prob_col2 = st.columns(2)

                    with prob_col1:
                        st.metric("Probabilidad NO", f"{probabilities.get('0', probabilities.get('no', 0))*100:.2f}%")

                    with prob_col2:
                        st.metric("Probabilidad SI", f"{probabilities.get('1', probabilities.get('yes', 0))*100:.2f}%")

                with st.expander("📤 Ver datos enviados a la API"):
                    st.json(payload)

                with st.expander("🔎 Ver respuesta completa de la API"):
                    st.json(result)

            else:
                st.error(f"Error en la predicción: {response.status_code}")
                st.json(response.json())

        except requests.exceptions.ConnectionError:
            st.error("❌ No se puede conectar a la API. Asegúrate de que esté ejecutándose.")

        except requests.exceptions.Timeout:
            st.error("❌ Tiempo de espera agotado. La API no respondió a tiempo.")

        except Exception as e:
            st.error(f"❌ Error inesperado: {str(e)}")

with tab2:
    st.header("📊 Información del Modelo")

    st.markdown("""
    ### Características del Modelo

    Este modelo de Machine Learning está diseñado para predecir si un cliente bancario
    suscribirá un depósito a plazo fijo basándose en:

    #### 📄 Variables de Entrada:

    **Datos Personales:**
    - Edad, Tipo de trabajo, Estado civil, Nivel educativo
    - Situación de crédito hipotecario, préstamos y default

    **Datos de Campaña:**
    - Tipo de contacto, Mes y día de la semana del contacto
    - Duración de la llamada, Número de contactos realizados
    - Resultado de campañas anteriores

    **Indicadores Económicos:**
    - Tasa de variación de empleo
    - Índice de precios al consumidor
    - Índice de confianza del consumidor
    - Tasa Euribor a 3 meses
    - Número de empleados

    #### 🎯 Salida del Modelo:
    - **Predicción**: `"yes"` o `"no"` (si el cliente suscribirá el depósito)
    - **Probabilidades**: Probabilidad para cada clase

    #### 🛠 Tecnologías Utilizadas:
    - **Backend**: FastAPI
    - **Frontend**: Streamlit
    - **Modelo**: Scikit-learn (Decision Tree)
    - **Preprocesamiento**: Pipeline de scikit-learn
    """)

    try:
        root_response = requests.get(API_URL, timeout=2)
        if root_response.status_code == 200:
            st.markdown("### 🌐 Endpoints Disponibles")
            st.json(root_response.json())
    except Exception:
        pass

    st.markdown("---")
    st.info("""
    💡 **Tip**: Para mejores predicciones, asegúrate de proporcionar datos precisos
    y completos del cliente. La duración de la llamada y el resultado de campañas
    anteriores son factores importantes en la predicción.
    """)