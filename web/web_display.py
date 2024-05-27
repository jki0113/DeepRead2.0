import streamlit as st

def display_version_updates():
    st.title("Version Update info")
    st.markdown("""---""")
    st.markdown("""
    ### 230804 - v1
    embedding - google universal-sentence-encoder 4  
    similarity - KNN  
    pdf2text module - PyPdf2
    """)
    st.markdown("""---""")
    st.markdown("""
    ### 230804 - v2
    embedding - google universal-sentence-encoder 4  
    similarity - KNN
    pdf2text module - PyMuPDF
    """)
    st.markdown("""---""")
    st.markdown("""
    ### 230809 - v3  
    embedding - OpenAI text-embedding-ada-002  
    similarity - FAISS  
    pdf2text module - PyMuPDF  
    CHUNK_SIZE = 350 words  
    OVERLAP_CHUNK_SIZE = 30 words  
    MAX_CANDIDATE = 5  
    CHAT_MODEL = "gpt-3.5-turbo-16k"  
    EMBEDDING_MODEL = "text-embedding-ada-002"  
    """)
    st.markdown("""---""")
    st.markdown("""
    ### 230811 - v4  
    embedding - OpenAI text-embedding-ada-002  
    similarity - FAISS  
    pdf2text module - PyMuPDF  
    CHUNK_SIZE = 350 words  
    OVERLAP_CHUNK_SIZE = 30 words  
    MAX_CANDIDATE = 5  
    CHAT_MODEL = "gpt-3.5-turbo-16k"  
    EMBEDDING_MODEL = "text-embedding-ada-002"
      
      
    *아이디만 입력하여 로그인 기능 추가  
    *로그인하면 과거 기록 보이게 수정(아이디 기준)  
    *기존pdf 불러올 때 대화 기록도 같이 나오도록 수정  
    *기존에서 pdf 불러오고 채팅하면 pdf 사라지는 이슈 수정  
    """)
    st.markdown("""---""")
    st.markdown("""
    ### 230811 - v5  
    embedding - OpenAI text-embedding-ada-002  
    similarity - FAISS  
    pdf2text module - PyMuPDF  
    CHUNK_SIZE = 350 words  
    OVERLAP_CHUNK_SIZE = 30 words  
    MAX_CANDIDATE = 5  
    CHAT_MODEL = "gpt-3.5-turbo-16k"  
    EMBEDDING_MODEL = "text-embedding-ada-002"

    *embedding vector db 추가  
    *word 서식지원 추가  
    *prompt v4로 변경  
    """)
    st.markdown("""---""")

def display_chatbot_prompt():
    st.title("Chatbot Prompt")
    st.markdown("""---""")
    st.markdown("""
        #### 230804-v1
        :red[{candidates}]  
        Instructions: Compose a comprehensive reply to the query using the search results given Cite each reference using [ Page Number] notation (every result has this number at the beginning).   
        Citation should be done at the end of each sentence. If the search results mention multiple subjects with the same name, create separate answers for each.  
        Only include information found in the results and don't add any additional information. Make sure the answer is correct and don't output false content.   
        If the text does not relate to the query, simply state 'Text Not Found in PDF'.  
        Ignore outlier search results which has nothing to do with the question. 
        Only answer what is asked.  
        The answer should be short and concise.  
        Answer step-by-step.  
        Query: :red[{input_text}]  
        Answer:""")
    st.markdown("""---""")
    st.markdown("""
        #### 230805-v2
        :red[{candidates}]  
        Query: :red[{input_text}]  
        Answer:
    """)
    st.markdown("""---""")
    st.markdown("""
        #### 230809-v3
        :red[{candidates}]  
        Based on the information about the above paper, please answer the following question.:  
        :red[{input_text}] 
    """)
    st.markdown("""---""")
    st.markdown("""
        #### 230824-v4
        You are the 'JournalLab-Chatbot' responding to inquiries about research papers.  

        \- Based on the information extracted from the papers, answer the questions.  
        \- For complex answers or when providing multiple pieces of information, structure your response in a list format for better readability, like:  
        · ...  
        · ...  
        · ...  
        \- If you cannot answer from the given information, inform the user that the information is not available and request a more specific question.  
        \- If a question is irrelevant to the research paper, respond that it's an unrelated question and therefore cannot be answered.  
        \- All answers to questions must be logically reasoned.  
        \- Ensure answers are clear, concise, and organized for easy comprehension by the user.  

        &#96;&#96;&#96;Extracted Information from Prompt:  
        :red[{candidates}]  
        &#96;&#96;&#96;Questions: :red[{input_text}]   
    """)
    st.markdown("""---""")

def display_sidebar_info():
    st.markdown("""
        *To reset, please refresh using :red[ctrl-shift-r].
        *There's a bug in MacOS using Google Chrome where the chat gets cut off if the last character is in Korean. When chatting in Korean, please end the last question with :red[English or symbols (!@#$%,.)].
    """)

def display_pdf_info():
    st.sidebar.write(f":red[**File Name:**] {st.session_state.file_name}")
    st.sidebar.write(f":red[**File Type:**] {st.session_state.extension}")
    st.sidebar.write(f":red[**File Path:**] {st.session_state.file_path}")