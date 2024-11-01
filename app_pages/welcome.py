import streamlit as st
st.write("# Welcome to LetiBot! 👋")

# Crear opciones en la barra lateral
st.sidebar.title("Opciones de la barra lateral")

# 1. Selectbox - Menú desplegable
opcion = st.sidebar.selectbox(
    "Selecciona una opción:",
    ("Opción 1", "Opción 2", "Opción 3")
)

# 2. Slider - Deslizador para seleccionar un número
valor_slider = st.sidebar.slider("Selecciona un valor", 0, 100, 50)

# 3. Text Input - Caja de texto
texto = st.sidebar.text_input("Introduce tu nombre", "Nombre")

# 4. Checkbox - Casilla de verificación
checkbox = st.sidebar.checkbox("Marcar esta casilla")

# 5. Radio Buttons - Botones de radio
radio = st.sidebar.radio(
    "Selecciona una opción de radio:",
    ("Primera opción", "Segunda opción", "Tercera opción")
)

# 6. SelectSlider - Deslizador con valores personalizados
valor_select_slider = st.sidebar.select_slider(
    "Selecciona un nivel:",
    options=["Bajo", "Medio", "Alto"]
)

# 7. Date Input - Seleccionar una fecha
fecha = st.sidebar.date_input("Selecciona una fecha")

# 8. Time Input - Seleccionar una hora
hora = st.sidebar.time_input("Selecciona una hora")

# 9. File Uploader - Subir un archivo
archivo = st.sidebar.file_uploader("Sube un archivo", type=["csv", "txt"])

# 10. Color Picker - Selector de color
color = st.sidebar.color_picker("Elige un color")

st.markdown(
    """
Que hace el letibot
"""
)