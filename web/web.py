import os
import sys
sys.path.insert(0, '/home/deepread2.0')

import web_session_manager
web_session_manager.init_session()
from log.logger_config import log, log_execution_time, logger, log
import web_display
import requests
import streamlit as st
st.set_page_config(layout='wide')
import web_display
import aiohttp

BACKEND_ROOT_PATH = 'http://127.0.0.1:8000'

st.title("DeepRead2.0")

# login session
if st.session_state.user_id is None:
    st.title("Log in")

    with st.form(key='id_form'):
        temp_id = st.text_input("Please enter your ID:")
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            if temp_id:
                st.session_state.user_id = temp_id
                st.success(f"Hello, {st.session_state.user_id}!")
                st.warning("please press enter one more time")
            else:
                st.warning("Please enter a valid ID.")
else:
    with st.sidebar:
        st.title("DeepRead2.0 PDF Chatbot")
        st.sidebar.markdown(f":red[**Log in with:**] {st.session_state.user_id}")

        languages = ["Korean", "English", "Japanese", "Chinese"]
        selected_language = st.selectbox("Choose a language for Chat:", languages)
        st.session_state.language = selected_language

        cols_row1 = st.columns(3)
        cols_row2 = st.columns(3)
        web_display.display_sidebar_info()

        # Retrieve the history file if the logged-in ID has one(Modification needed to allow execution only once through a session)
        response = requests.post(
            f'{BACKEND_ROOT_PATH}/storage/check_history',
            json={
                "user_id" : st.session_state.user_id
            }
        )
        st.session_state.history_files = response.json()['history_files']
        
    if "section" not in st.session_state:
        st.session_state.section = "Section 1"
    if cols_row1[0].button("Chatbot"):
        st.session_state.section = "Section 1"
    if cols_row1[1].button("Version"):
        st.session_state.section = "Section 2"
    if cols_row1[2].button("Prompt"):
        st.session_state.section = "Section 3"
    if cols_row2[0].button("dev"):
        st.session_state.section = "Section 4"
    if cols_row2[1].button("deepread1.0"):
        st.session_state.section = "Section 5"

    # chatbot session
    if st.session_state.section == "Section 1":
        st.title("Chat-bot")
        if st.session_state.history_files:
            st.sidebar.title(f"{st.session_state.user_id}'s Files")
            selected_file = st.sidebar.selectbox("Select a file:", st.session_state.history_files, index=0)
            if st.sidebar.button("open"):
                st.session_state.file_path = selected_file if selected_file else st.session_state.history_files[0]

                # When the button is pressed, it loads all the information from the history file.
                response = requests.post(
                    f'{BACKEND_ROOT_PATH}/storage/get_document_history', 
                    json={'history_path': st.session_state.file_path}
                )
                st.session_state.storage_path = response.json()['storage_path']
                st.session_state.file_name = response.json()['file_name']
                st.session_state.extension = response.json()['extension']
                st.session_state.document_info = response.json()['document_info']

                # displays the PDF temporarily upon request. This part will be removed later when we collaborate with the front-end team to retrieve the actual PDF
                response = requests.post(
                    f'{BACKEND_ROOT_PATH}/chat/tmp_get_full_text',
                    json={'file_path' : f'{st.session_state.file_path}'}
                )
                st.session_state.full_text = response.json()['full_text']
                
        keys = ["file_path", "storage_path", "file_name", "extension"]
        if all(st.session_state.get(key) is None for key in keys):      
            uploaded_pdf_file = st.file_uploader("Upload your research", type=['pdf', 'docx'])
            if uploaded_pdf_file:
                uploaded_pdf_file = (uploaded_pdf_file.name, uploaded_pdf_file, uploaded_pdf_file.type)

                # When a file is uploaded, it saves the PDF file to the user's folder and generates basic information
                response = requests.post(
                    f'{BACKEND_ROOT_PATH}/storage/save_uploaded_file', 
                    data={'user_id': st.session_state.user_id}, 
                    files={'uploaded_pdf_file': uploaded_pdf_file}
                )
                st.session_state.file_path = response.json()['file_path']
                st.session_state.storage_path = response.json()['storage_path']
                st.session_state.file_name = response.json()['file_name']
                st.session_state.extension = response.json()['extension']

                # displays the PDF temporarily upon request. This part will be removed later when we collaborate with the front-end team to retrieve the actual PDF
                response = requests.post(
                    f'{BACKEND_ROOT_PATH}/chat/tmp_get_full_text',
                    json={'file_path' : f'{st.session_state.file_path}'}
                )
                st.session_state.full_text = response.json()['full_text']
                st.markdown(st.session_state.full_text)

                # After extracting journal basic information based on full text, it is stored in 'paper_basic_info.json' at the 'storage_path'
                response = requests.post(
                    f'{BACKEND_ROOT_PATH}/chat/extract_paper_basic_info',
                    json={
                        'full_text' : st.session_state.full_text,
                        'storage_path' : st.session_state.storage_path
                    }
                )
###################################################################################################
                # response = requests.post(
                #     f'{BACKEND_ROOT_PATH}/analysis/extract_keywords',
                #     json = {
                #         'storage_path' : st.session_state.storage_path
                #     }
                # )
                # st.session_state.recommended_keyword = response.json()['recommended_keyword']

                # response = requests.post(
                #     f'{BACKEND_ROOT_PATH}/analysis/summarize',
                #     json = {
                #         'storage_path' : st.session_state.storage_path
                #     }
                # )
                # st.session_state.recommended_summarize = response.json()['recommended_summarize']

                # response = requests.post(
                #     f'{BACKEND_ROOT_PATH}/analysis/extract_titles',
                #     json = {
                #         'storage_path' : st.session_state.storage_path
                #     }
                # )
                # st.session_state.recommended_title = response.json()['recommended_title']

                # response = requests.post(
                #     f'{BACKEND_ROOT_PATH}/analysis/recommend-journal',
                #     json = {
                #         'storage_path' : st.session_state.storage_path
                #     }
                # )
                # st.session_state.recommended_journal = response.json()

                # response = requests.post(
                #     f'{BACKEND_ROOT_PATH}/analysis/recommend-references',
                #     json = {
                #         'storage_path' : st.session_state.storage_path
                #     }
                # )
                # st.session_state.recommended_reference = response.json()
###################################################################################################
                # dictionary containing chunks, embeddings, and page information is retrieved
                response = requests.post(
                    f'{BACKEND_ROOT_PATH}/chat/get_document_info',
                    json={
                        'file_path' : f'{st.session_state.file_path}',
                        'storage_path' : st.session_state.storage_path
                    }
                )
                st.session_state.document_info = response.json()['document_info']

        if all(key in st.session_state and st.session_state[key] is not None for key in ["full_text", "extension", "file_name", "file_path"]):
            web_display.display_pdf_info()
            st.markdown(st.session_state.full_text)
            st.session_state.chat_ready = True

        # Since file upload or loading is complete, the chat session is started
        if st.session_state.chat_ready:
            
            # Displaying the existing chat records.
            response = requests.post(
                f'{BACKEND_ROOT_PATH}/chat/show_chat',
                json={'chat_path' : f'{st.session_state.storage_path}'}
            )
            st.session_state.chat_log  = response.json()['chat_log']
            for message in st.session_state.chat_log:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            # Receive chat input (for screen, distinguish actual usage)."
            prompt = st.chat_input("write your question...")
            if prompt:
                # Display what has been entered
                with st.chat_message("user"):
                    st.markdown(prompt)
                # Initialize the place to receive responses.
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                response = requests.post(
                    f'{BACKEND_ROOT_PATH}/chat/get_total_prompt',
                    json={
                        'prompt' : f'{prompt}',
                        'storage_path' : f'{st.session_state.storage_path}',
                        'chat_log' : st.session_state.chat_log,
                        'language' : st.session_state.language
                    }
                )
            # Update the response
                message_placeholder.markdown(response.json()['total_prompt']['content'])
                st.session_state.chat_log.append(response.json()['total_prompt'])
    
    # 
    if st.session_state.section == "Section 2":
        st.markdown("Section 2")
        # 여기에서 deepread1.0의 모든 결과를 요약적으로 보여줘야 합니다.
    
    #
    if st.session_state.section == "Section 3":
        st.markdown("Section 3")
    
    # 
    if st.session_state.section == "Section 4":
        st.write(st.session_state)
    
    # DeepRead1.0 결과 반환
    if st.session_state.section == "Section 5":
        import json
        with open(st.session_state.storage_path + 'paper_basic_info.json', encoding='utf-8-sig') as json_file:
            json_data = json.load(json_file)
        st.json(json_data)

