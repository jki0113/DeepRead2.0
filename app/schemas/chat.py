from fastapi import UploadFile, HTTPException, Body, Query, Path, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from datetime import datetime


# validate_chatbot_init
class request_validate_chatbot_init(BaseModel):
    user_id: str = "user_id"
    project_id: str = "project_id"

# gget_chatbot_init
class request_get_chatbot_init(BaseModel):
    project_id: str = "project_id"
    response_language : Optional[str] = "response_language"
class response_get_chatbot_init(BaseModel):
    project_id: str
    folder_name: str
    project_name: str
    project_file_path: str
    pdf_pages: int
    file_name: str
    created_at: datetime
    chat_log: List
    project_analysis_survey : Dict
    default_questions: List[str]

# 질문 답변에 대한 추가 질문 추천
class request_get_recommended_questions(BaseModel):
    response_language : Optional[str] = None
    user_chat: str
    assistant_chat: Union[List, str]
    function: str
class response_get_recommended_questions(BaseModel):
    question_list : List[str]

# 북마크가 표시된 프로젝트 조회
class request_get_projects_with_bookmarks(BaseModel):
    user_id : str
class response_get_projects_with_bookmarks(BaseModel):
    bookmark_project: List[Dict[str, Any]] = []

class request_get_chat_with_bookmarks(BaseModel):
    project_id: str
class response_get_chat_with_bookmarks(BaseModel):
    bookmark_chat: List[Dict[str, Any]] = []
    
# 북마크 상태 변경
class request_update_bookmark_status(BaseModel):
    chat_index : str

# 좋아요 상태 변경
class request_update_chat_like_status(BaseModel):
    chat_index : str

class response_bool(BaseModel):
    response : bool