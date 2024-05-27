from datetime import datetime
from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, select, update
from log.logger_config import logger, log_execution_time, log

from app.database.models import *
from app.database.db_connection import create_db_connection


async def validate_chatbot_init(db: Session, user_id: str, project_id: str) -> Tuple[bool, any, any]:
    try:
        project = await db.execute(
            select(Project.user_id)
            .where(Project.project_id == project_id)
            .where(Project.del_yn == 'n')
        )
        project_result = project.scalar_one_or_none()

        if project_result and project_result == user_id:
            return True, None, True
        else:
            return True, None, False
    
    except Exception as e:
        logger.error(f"validate_chatbot_init error: {e}")
        await db.rollback()
        return False, e, None


async def chat_init(db: Session, project_id: str) -> Tuple[bool, any, any]:
    try:
        project_result = await db.execute(
            select(Project, Folder.folder_name).join(
                Folder, Project.folder_id == Folder.folder_id
            ).where(Project.project_id == project_id)
        )
        project_data_row = project_result.first()

        if project_data_row:
            project, folder_name = project_data_row
            project_data = {
                'project_id': project.project_id,
                'folder_name': folder_name,
                'project_name': project.project_name,
                'project_path': project.project_path,
                'created_at' : project.created_at,
            }
            return True, None, project_data
        else:
            return True, None, None

    except Exception as e:
        logger.error(f"chat_init error: {e}")
        await db.rollback()
        return False, e, None
    

async def get_project_chats(db: Session, project_id: str) -> Tuple[bool, any, any]:
    """
    주어진 프로젝트 ID에 해당하는 채팅 기록을 chat_index로 정렬하여 가져옵니다.

    Args:
        project_id (str): 프로젝트 ID

    Returns:
        List[dict]: 해당 프로젝트의 채팅 기록 리스트 (딕셔너리 형태)
    """
    try:
        result = await db.execute(
            select(UserChat)
            .where(UserChat.project_id == project_id)
            .order_by(UserChat.chat_index.asc())
        )
        chats = result.scalars().all()

        chat_dicts = [
            {
                "chat_index": chat.chat_index,
                "role": chat.role, 
                "content": chat.content, 
                "function": chat.function, 
                "bookmark": chat.bookmark, 
                "like": chat.like, 
                "created_at": chat.created_at
            } for chat in chats
        ]

        return True, None, chat_dicts
    
    except Exception as e:
        logger.error(f"get_project_chats error: {e}")
        await db.rollback()
        return False, e, None
    

async def get_project_analysis_survey(db: Session, project_id: str) -> Tuple[bool, any, any]:
    try:
        query = select(
            AnalysisSatisfaction, 
            Project.project_type
        ).join(
            Project, AnalysisSatisfaction.project_id == Project.project_id
        ).where(
            AnalysisSatisfaction.project_id == project_id
        )

        result = await db.execute(query)
        data_row = result.first()

        survey_data = {
            'project_type': None,
            'recommended_title_satis_yn': None,
            'recommended_keyword_satis_yn': None,
            'recommended_summarize_satis_yn': None,
            'recommended_potential_topics_satis_yn': None,
            'recommended_journal_satis_yn': None,
            'recommended_paper_satis_yn': None,
            'published_info_satis_yn': None,
            'comment': None,
            'created_at': None,
        }

        if data_row:
            analysis_survey, project_type = data_row
            survey_data.update({
                'project_type': project_type,
                'recommended_title_satis_yn': analysis_survey.recommended_title_satis_yn,
                'recommended_keyword_satis_yn': analysis_survey.recommended_keyword_satis_yn,
                'recommended_summarize_satis_yn': analysis_survey.recommended_summarize_satis_yn,
                'recommended_potential_topics_satis_yn': analysis_survey.recommended_potential_topics_satis_yn,
                'recommended_journal_satis_yn': analysis_survey.recommended_journal_satis_yn,
                'recommended_paper_satis_yn': analysis_survey.recommended_paper_satis_yn,
                'published_info_satis_yn': analysis_survey.published_info_satis_yn,
                'comment': analysis_survey.comment,
                'created_at': analysis_survey.created_at
            })

        else:
            project_result = await db.execute(
                select(Project.project_type).where(Project.project_id == project_id)
            )
            project_data_row = project_result.first()
            if project_data_row:
                project_type, = project_data_row
                survey_data['project_type'] = project_type

        return True, None, survey_data

    except Exception as e:
        logger.error(f"get_project_analysis_survey error: {e}")
        await db.rollback()
        return False, e, None
    
      
async def create_new_chat_index(db: Session, project_id: str) -> Tuple[bool, any, any, any]:
    try:
        max_index_result = await db.execute(
            select(func.max(UserChat.chat_index))
            .where(UserChat.project_id == project_id)
            .where(UserChat.chat_index.like(f"{project_id}_chat_%"))
        )
        max_index = max_index_result.scalar()

        if max_index:
            max_number = int(max_index.split('_')[-1]) + 1
            new_chat_index = f"{project_id}_chat_{max_number:03d}"
            new_chat_response_index = f"{project_id}_chat_{max_number + 1:03d}"
        else:
            new_chat_index = f"{project_id}_chat_001"
            new_chat_response_index = f"{project_id}_chat_002"

        return True, None, new_chat_index, new_chat_response_index
        
    except Exception as e:
        logger.error(f"create_new_chat_index error: {e}")
        await db.rollback()
        return False, e, None, None


async def update_user_chat(
    db: Session,
    project_id: str,
    chat_index: str,
    role: str,
    content: str,
    response_language: str,
    function: str
) -> Tuple[bool, any, any]:
    try:
        new_chat = UserChat(
            project_id=project_id,
            chat_index=chat_index,
            role=role,
            content=content,
            response_language=response_language,
            function=function,
        )
        db.add(new_chat)
        await db.commit()
        await db.refresh(new_chat)
        return True, None, new_chat
    
    except Exception as e:
        logger.error(f"update_user_chat error: {e}")
        await db.rollback()
        return False, e, None


async def get_project_with_bookmark(db: Session, user_id: str) -> Tuple[bool, any, any]:
    try:
        query = select(Project, Folder).join(Folder, Project.folder_id == Folder.folder_id).join(UserChat, Project.project_id == UserChat.project_id).where(UserChat.bookmark.isnot(None), Project.user_id == user_id, Folder.del_yn == 'n', Project.del_yn == 'n')
        result = await db.execute(query)

        projects_with_bookmarks = []
        for project, folder in result:
            project_info = {
                "folder_id": folder.folder_id,
                "folder_name": folder.folder_name,
                "project_id": project.project_id,
                "project_name": project.project_name,
                "project_path": project.project_path  # 추가된 부분
            }
            projects_with_bookmarks.append(project_info)

        return True, None, projects_with_bookmarks

    except Exception as e:
        logger.error(f"get_project_with_bookmark error: {e}")
        await db.rollback()
        return False, e, None


async def get_chat_with_bookmark(db: Session, project_id: str) -> Tuple[bool, any, any]:
    try:
        query = select(UserChat).join(Project, UserChat.project_id == Project.project_id).where(UserChat.bookmark.isnot(None), UserChat.project_id == project_id, Project.del_yn == 'n')
        result = await db.execute(query)

        chats_with_bookmarks = []
        for chat in result.scalars():
            chat_info = {
                "chat_index": chat.chat_index,
                "role": chat.role, 
                "content": chat.content, 
                "function": chat.function, 
                "bookmark": chat.bookmark, 
                "like": chat.like, 
                "created_at": chat.created_at
            }
            chats_with_bookmarks.append(chat_info)

        return True, None, chats_with_bookmarks

    except Exception as e:
        logger.error(f"get_chat_with_bookmark error: {e}")
        await db.rollback()
        return False, e, None
    

async def update_bookmark_status(db: Session, chat_index: str) -> Tuple[bool, any, any]:
    try:
        user_chat = await db.execute(
            select(UserChat.bookmark).where(UserChat.chat_index == chat_index)
        )
        user_chat_result = user_chat.scalar_one_or_none()

        new_bookmark_status = None if user_chat_result else datetime.now()

        await db.execute(
            update(UserChat).
            where(UserChat.chat_index == chat_index).
            values(bookmark=new_bookmark_status)
        )
        await db.commit()
        return True, None, True

    except Exception as e:
        logger.error(f"update_bookmark_status error: {e}")
        await db.rollback()
        return False, e, None


async def update_chat_like_status(db: Session, chat_index: str) -> Tuple[bool, any, any]:
    try:
        user_chat = await db.execute(
            select(UserChat.like).where(UserChat.chat_index == chat_index)
        )
        user_chat_result = user_chat.scalar_one_or_none()

        new_chat_like_status = None if user_chat_result else datetime.now()

        await db.execute(
            update(UserChat).
            where(UserChat.chat_index == chat_index).
            values(like=new_chat_like_status)
        )
        await db.commit()
        return True, None, True

    except Exception as e:
        logger.error(f"update_chat_like_status error: {e}")
        await db.rollback()
        return False, e, None
    

async def get_project_path_by_project_id(project_id: str) -> str:
    """
    주어진 프로젝트 ID에 해당하는 프로젝트의 파일 경로를 반환
    """
    async_session_factory = create_db_connection()
    async with async_session_factory() as session:
        try:
            project = await session.get(Project, project_id)
            if project:
                return project.project_path  # 프로젝트의 파일 경로 반환
            else:
                raise ValueError('해당 프로젝트가 존재하지 않습니다.')  # 프로젝트가 없는 경우 None 반환
        finally:
            await session.close()

async def update_project_recent_timestep(db: Session, project_id: str) -> Tuple[bool, any, any]:
    try:
        project = await db.execute(
            select(Project.recent_timestep).where(Project.project_id == project_id)
        )
        project_result = project.scalar_one_or_none()

        if project_result is None:
            logger.error(f"Project with ID {project_id} not found")
            return False, "Project not found", None

        await db.execute(
            update(Project).
            where(Project.project_id == project_id).
            values(recent_timestep=datetime.now())
        )
        await db.commit()
        return True, None, True

    except Exception as e:
        logger.error(f"update_project_recent_timestep error: {e}")
        await db.rollback()
        return False, e, None