import streamlit as st
from log.logger_config import log, log_execution_time, logger, log

def init_session():
    if not st.session_state.get('initialized', False):
        """
        Initializes the applications session state.
        After initialization, 'st.session_state.initialized' is set to 'True' to indicate that the session is initialized.

        Parameters:
            - None
        
        Returns:
            - None
        """
        if "user_id" not in st.session_state:
            st.session_state.user_id = None

        if "file_path" not in st.session_state:
            st.session_state.file_path = None

        if "storage_path" not in st.session_state:
            st.session_state.storage_path = None

        if "file_name" not in st.session_state:
            st.session_state.file_name = None

        if "extension" not in st.session_state:
            st.session_state.extension = None

        if "full_text" not in st.session_state:
            st.session_state.full_text = None
            
        if "section" not in st.session_state:
            st.session_state.section = "Section 1"

        if "chat_ready" not in st.session_state:
            st.session_state.chat_ready = False
        
        if "language" not in st.session_state:
            st.session_state.language = "Korean"
        
        if "recommended_keyword" not in st.session_state:
            st.session_state.recommended_keyword = "N/A"
        
        if "recommended_summarize" not in st.session_state:
            st.session_state.recommended_summarize = "N/A"
        
        if "recommended_title" not in st.session_state:
            st.session_state.recommended_title = "N/A"
        
        if "recommended_journal" not in st.session_state:
            st.session_state.recommended_title = "N/A"
        
        if "recommended_reference" not in st.session_state:
            st.session_state.recommended_title = "N/A"
        
    st.session_state.initialized = True
    return

    
def update_session(session_name, session_state):
    """
    Updates the session state with the given session name and value

    Parameters:
        - session_name (str): Name of the session variable to be updated
        - session_state: New value for the session variable
    
    Returns:
        - None
    """
    setattr(st.session_state, session_name, session_state)
    return

def delete_session(session_name):
    """
    Deletes the session state of the given session name

    Parameters:
        - session_name (str): Nmae of the session variable to be deleted
    
    Returns:
        - None
    """
    if session_name in st.session_state:
        del st.session_state[session_name]
    else:
        st.warning(f"{session_name} does not exist in the session state.")
