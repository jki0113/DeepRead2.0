# tmp package
import os, sys
import openai
from sqlalchemy.orm import Session
from log.logger_config import logger, log_execution_time, log
from fastapi import APIRouter, UploadFile, Form, File, BackgroundTasks, WebSocket, Depends, HTTPException
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK
from typing import List, Optional, Dict, Union, Tuple, Set, Any

from app.queries import chat as query
from app.services import chat as services
from app.schemas import chat as schemas
from utils.async_gpt import *
from app.database.db_connection import get_async_db
openai.api_key = os.getenv('OPENAI_KEY')


router = APIRouter()


@router.post("/validate_chatbot_init", response_model=schemas.response_bool)
@log_execution_time
async def validate_chatbot_init(request: schemas.request_validate_chatbot_init, db: Session = Depends(get_async_db)) -> bool:
    success, ermsg, response = await query.validate_chatbot_init(db, user_id=request.user_id, project_id=request.project_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="챗봇 초기화에 실패했습니다.")
    if not response:
        raise HTTPException(status_code=404, detail="해당 프로젝트의 사용자를 찾을 수 없습니다.")
    
    return schemas.response_bool(response=response)


@router.get("/get_chatbot_init", response_model=schemas.response_get_chatbot_init)
@log_execution_time
async def get_chatbot_init(request: schemas.request_get_chatbot_init=Depends(), db: Session = Depends(get_async_db)):
    """
    Initializes chatbot settings based on a given project ID.  
  
    Parameters:  
        project_id: str  
        response_language: str  
  
    Returns:  
        project_id: str  
        folder_name: str  
        project_name: str  
        project_file_path: str  
        created_at : datetime
        file_name: str  
        recommended_questions: List
    """
    success, ermsg, response = await services.get_chatbot_init(db, request.project_id, request.response_language)
    if not success:
        raise HTTPException(status_code=500, detail="해당 프로젝트를 기반으로 챗봇 설정을 초기화하는데 실패했습니다.")

    return response


@router.websocket("/chat/{project_id}")
async def chat(websocket: WebSocket, project_id: str, db: Session = Depends(get_async_db)):
    logger.debug(f'WebSocket Open for {project_id}')
    """
    Websocket endpoint for a chat service in a specific project context. Handles real-time messaging and processing with OpenAI's model.  
  
    Parameters:  
        project_id: str  
        message parameters:  
            user_input: str  
            response_language: str  
            function: str  
  
    Returns:  
        None  
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            success, ermsg, response = await services.chat(db, websocket=websocket, data=data, project_id=project_id)
            if not success:
                logger.error(ermsg)
                raise HTTPException(status_code=500, detail="채팅 생성에 실패했습니다.")

    except WebSocketDisconnect:
        logger.warning(f"WebSocket connection has been disconnected for project ID: {project_id}.")

    except ConnectionClosedOK:
        logger.debug(f"WebSocket connection has been closed normally and as expected for project ID: {project_id}.")


@router.post('/get_recommended_questions', response_model=schemas.response_get_recommended_questions)
@log_execution_time
async def get_recommended_questions(request: schemas.request_get_recommended_questions):
    """
    Generates recommended follow-up questions based on the user's and assistant's previous chat messages.  
  
    Parameters:  
        user_chat: str  
        assistant_chat: str  
        response_language: str
  
    Returns:  
        question_list: List[str]  
    """
    success, ermsg, question_list = await services.get_recommended_questions(
        user_chat=request.user_chat, 
        assistant_chat=request.assistant_chat,
        function=request.function,
        response_language=request.response_language,
    )
    if not success:
        raise HTTPException(status_code=500, detail="이전 채팅 기반의 추천 질문을 조회하는데 실패했습니다.")

    return schemas.response_get_recommended_questions(question_list=question_list)


@router.get("/get_project_with_bookmark", response_model=schemas.response_get_projects_with_bookmarks)
@log_execution_time
async def get_projects_with_bookmarks(request: schemas.request_get_projects_with_bookmarks=Depends(), db: Session = Depends(get_async_db)):
    """
    Retrieves projects with bookmarks for a given user.  
  
    Parameters:  
        user_id: str  
  
    Returns:  
        bookmark_project: List[Dict[str, Any]]  
    """
    success, ermsg, bookmark_project = await services.get_project_with_bookmark(db, user_id=request.user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to retrieve bookmarked projects for the user.")
    return schemas.response_get_projects_with_bookmarks(bookmark_project=bookmark_project)


@router.get("/get_chat_with_bookmark", response_model=schemas.response_get_chat_with_bookmarks)
@log_execution_time
async def get_chat_with_bookmark(request: schemas.request_get_chat_with_bookmarks=Depends(), db: Session = Depends(get_async_db)):
    """
    Retrieves bookmarked chat messages for a given project.  
  
    Parameters:  
        project_id: str  
  
    Returns:  
        bookmark_chat: List[Dict[str, Any]]  
    """
    success, ermsg, bookmark_chat = await query.get_chat_with_bookmark(db, project_id=request.project_id)
    if not success:
        raise HTTPException(status_code=500, detail="해당 사용자의 북마크 프로젝트의 챗봇 대화 조회에 실패했습니다.")
    
    return schemas.response_get_chat_with_bookmarks(bookmark_chat=bookmark_chat)


@router.post("/update_bookmark_status", response_model=schemas.response_bool)
@log_execution_time
async def update_bookmark_status(request: schemas.request_update_bookmark_status, db: Session = Depends(get_async_db)):
    """
    Toggles the bookmark status of a specific chat message.  
  
    Parameters:  
        chat_index: str  
  
    Returns:  
        response: bool  
    """
    success, ermsg, response = await query.update_bookmark_status(db, chat_index=request.chat_index)
    if not success:
        raise HTTPException(status_code=500, detail="해당 챗의 북마크 상태 업데이트에 실패했습니다.")
    
    return schemas.response_bool(response=response)


@router.post("/update_chat_like_status", response_model=schemas.response_bool)
@log_execution_time
async def update_chat_like_status(request: schemas.request_update_chat_like_status, db: Session = Depends(get_async_db)):
    """
    Toggles the 'like' status of a specific chat message.
  
    Parameters:  
        chat_index: str  
  
    Returns:  
        response: bool  
    """
    success, ermsg, response = await query.update_chat_like_status(db, chat_index=request.chat_index)
    if not success:
        raise HTTPException(status_code=500, detail="해당 챗의 좋아요 상태 업데이트에 실패했습니다.")
    
    return schemas.response_bool(response=response)

