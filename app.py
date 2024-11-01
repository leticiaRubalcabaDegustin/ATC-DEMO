import streamlit as st

welcome = st.Page("./app_pages/welcome.py", title="Bienvenido", icon="👋")

nlp2sql = st.Page("./app_pages/nlp2sql.py", title="NLP2SQL", icon="🤖")
rag = st.Page("./app_pages/mi_cv.py", title="Mi CV", icon="📄")

bd = st.Page("./app_pages/bd.py", title="Añadir base de datos", icon="🔧")
index = st.Page("./app_pages/index.py", title="Gestionar índices", icon="🔍")


pg = st.navigation(
    {
       "Información": [welcome],
       "Bots": [nlp2sql, rag],
       "Ajustes": [bd, index] 
    }
    )

st.set_page_config(
    page_title="LetiBot",
    page_icon="🤖",
)

pg.run()