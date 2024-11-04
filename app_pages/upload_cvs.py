import streamlit as st
import langchain_mi_cv as lu
import pandas as pd
import json
from streamlit_navigation_bar import st_navbar
import time
import os
import index_functions as indxfunc
from streamlit_pills import pills
import app_pages. your_rag_cv


# Título de la aplicación

st.title("Submit the CV to be added to the index")
st.write("You can upload the PDF to the index, which will be processed automatically using a Retrieval-Augmented Generation (RAG) system.")

# Área de arrastrar y soltar para subir el archivo
uploaded_file = st.file_uploader("Drag and drop your file here or select a file", type=["pdf"])

# Botón para subir el archivo
if st.button("Upload File"):
    if uploaded_file is not None:
        #Upload file to index
        with st.spinner("Uploading... please wait."):
            file_uploded,cv_name= indxfunc.load_pdf_index(uploaded_file)
        st.success(f'The CV for {cv_name} has been successfully uploaded!', icon="✅")
        if not file_uploded:
            st.toast("There was an issue with the PDF. Please check the file and try again.")
        else:
            # st.switch_page("./app_pages/your_rag_cv.py")
            st.session_state.messages_mi_cv_rag = []
    else:
        st.toast("Please select a file and enter the applicant's name in the form to continue")

# Name of CV
# cv_name = st.text_input("Enter the CV Applicant's Name")


    