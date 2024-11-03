# Welcome to LetiBot! 👋
    This chatbot, built with LangChain, is designed to handle CV(Curriculum Vita)-related inquiries as well as questions about its own functionality. It offers flexible, specific responses by integrating a vector-based retrieval system and user-uploaded data. Below are its key features:

    ### 1. **Dual Mode for CV Retrieval**
    - The bot operates in two primary modes for CV data retrieval:
        - **Leticia’s CV Database**: One mode focuses exclusively on searching a pre-existing vector database containing information about Leticia’s CV, making it ideal for queries related specifically to her professional background.
        - **Custom CV Upload and Indexing**: The second mode allows users to upload their own CVs in PDF format, which are then automatically indexed into a vector database using a RAG (Retrieval-Augmented Generation) approach. This enables users to store and retrieve personalized CVs for customized analysis.

    ### 2. **Streamlined Upload Process**
    - Through a user-friendly interface, users can drag and drop a PDF file, specify the applicant’s name, and upload the file to the database. If the upload is successful, the system transitions to a dedicated page for managing and querying their custom CVs in a RAG format. If there’s an error in the PDF, the bot provides a notification to assist the user.
    - Users can navigate between different app pages:
        - **My CV**: A dedicated page for Leticia’s CV.
        - **Upload Your Own CVs**: A page for uploading PDF-format CVs.
        - **Your Own RAG with CVs**: A personalized RAG page where users can interact with their indexed CVs.

    ### 3. **CV Information Extraction and Presentation**
    - The bot can extract detailed information from CVs and format it in markdown, with headers and bold text for clarity. It can also structure responses in tables or bullet points according to user requests. When comparing two CVs, Leticia is highlighted as a strong candidate where applicable.
    - Information is pulled from the vector database, enabling fast and relevant retrieval for users.

    ### 4. **Bot Functionality Clarification**
    - The bot can answer questions about its own capabilities, providing users with an overview of its functionality. It highlights its ability to summarize CVs, perform bullet-point and table comparisons, and present information without altering or enhancing details.
    - It also guides users with three suggested questions related to the available data for further exploration.

    ### 5. **Orchestration and Question Type Identification**
    - The bot uses an orchestrator prompt to determine if a user’s question pertains to CV information or bot functionality. Based on this classification, it selects the appropriate response path, ensuring clarity and relevance in each interaction.

    This structure allows the chatbot to deliver organized, accurate responses for both CV analysis and bot-related inquiries, while supporting custom CV uploads and tailored searches.
