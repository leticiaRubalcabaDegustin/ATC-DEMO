import streamlit as st

welcome = st.Page("./app_pages/welcome.py", title="Welcome", icon="👋")

rag = st.Page("./app_pages/my_cv.py", title="My CV", icon="📄")

upload_files  = st.Page("./app_pages/upload_cvs.py", title="Upload your own CVs", icon="🔧")

your_own_rag = st.Page("./app_pages/your_rag_cv.py", title="Your own RAG with CVs", icon="🔍")


pg = st.navigation(
    {
       "Information": [welcome],
       "Leticia": [rag],
        "CV Bot": [upload_files, your_own_rag]
    }
    )

st.set_page_config(
    page_title="LetiBot",
    page_icon="🤖",
)

pg.run()