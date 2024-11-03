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

# Name of CV
cv_name = st.text_input("Enter the CV Applicant's Name")


# Botón para subir el archivo
if st.button("Upload File"):
    if uploaded_file is not None and cv_name is not None and cv_name!='':
        #Upload file to index
        file_uploded= indxfunc.load_pdf_index(uploaded_file, cv_name)
        if not file_uploded:
            st.toast("There was an issue with the PDF. Please check the file and try again.")
        else:
            st.switch_page("./app_pages/your_rag_cv.py")
            st.session_state.messages_mi_cv_rag = []
    else:
        st.toast("Please select a file and enter the applicant's name in the form to continue")

# st.title("File already uploaded")
# st.write("I don't need to upload a file")
# # Botón para subir el archivo
# if st.button("Continue"):
#     st.switch_page("./app_pages/your_rag_cv.py")
#     st.session_state.messages_mi_cv_rag = []
        

    