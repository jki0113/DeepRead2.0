from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, select, update, or_
import sqlalchemy.exc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import *
from log.logger_config import logger, log_execution_time, log

from typing import List, Tuple
from datetime import datetime
import traceback
 
############################## USER ##############################
async def create_user(db: AsyncSession, user_id: str) -> Tuple[bool, any, any]:
    """ 새로운 사용자를 User 테이블에 등록 """
    try:
        new_user = User(user_id=user_id)
        db.add(new_user)
        await db.commit()
        return True, None, True

    except Exception as e:
        logger.error(f"create_user error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def get_user_is_in_database(db: AsyncSession, user_id: str) -> Tuple[bool, any, any]:
    """ 주어진 사용자 ID가 User 테이블에 존재하는지 확인 """
    try:
        query = select(User).where(User.user_id == user_id)
        result = await db.execute(query)
        user_exists = result.scalars().first() is not None
        return True, None, user_exists

    except Exception as e:
        logger.error(f"get_user_is_in_database error: {e}")
        await db.rollback()
        return False, str(e), None


async def check_user_completed_survey(db: AsyncSession, user_id: str, survey_id: str) -> Tuple[bool, any, any]:
    """ 사용자가 특정 설문조사를 완료했는지 확인 """
    try:
        query = select(SurveyAnswer).join(
            SurveyQuestions, 
            SurveyAnswer.question_id == SurveyQuestions.question_id
        ).where(
            SurveyQuestions.survey_id == survey_id, 
            SurveyAnswer.user_id == user_id
        )

        result = await db.execute(query)
        response = result.scalars().first() is not None # 결과가 하나라도 존재하면 True, 아니면 False
        return True, None, response
    
    except Exception as e:
        logger.error(f"check_user_completed_survey error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


########################## Folder CRUD ###########################
async def get_recent_projects(db: AsyncSession, user_id: str, limit: int, sort: str, order: str) -> Tuple[bool, any, any]:
    try:
        query = select(
            Project.project_id, 
            Project.project_name, 
            Project.project_type,
            Project.folder_id, 
            Folder.folder_name, 
            Project.created_at
        ).join(
            Folder, Project.folder_id == Folder.folder_id
        ).where(
            Project.user_id == user_id,
            Project.del_yn == 'n'  
        )

        order_function = asc if order == "asc" else desc
        if hasattr(Project, sort):
            query = query.order_by(order_function(getattr(Project, sort)))
        elif hasattr(Folder, sort):
            query = query.order_by(order_function(getattr(Folder, sort)))
        else:
            raise ValueError("Invalid sort parameter")

        query = query.limit(limit)

        result = await db.execute(query)
        projects = [
            {
                'project_id': project[0], 
                'project_name': project[1], 
                'project_type': project[2],
                'folder_id': project[3], 
                'folder_name': project[4], 
                'created_at': project[5]
            } 
            for project in result.fetchall()
        ]
        return True, None, projects

    except Exception as e:
        logger.error(f"get_recent_projects error : {e}\n{traceback.format_exc()}")
        return False, str(e), None


async def initialize_user_folders(
    db: AsyncSession, 
    user_id: str, 
    folder_id: str
) -> Tuple[bool, any, any]:
    try:
        """
        주어진 사용자 ID에 해당하는 폴더를 초기화합니다.
        """
        init_index = 0
        init_folder_name = "기본 폴더"
        new_folder = Folder(
            folder_id=folder_id,
            user_id=user_id,
            folder_index=init_index,  # 디폴트 폴더는 항상 0번으로 위치함
            folder_name=init_folder_name
        )
        db.add(new_folder)
        await db.commit()

        init_folder_list = [{
            'folder_id': folder_id,
            'folder_name': init_folder_name,
            'folder_index': init_index
        }]

        return True, None, init_folder_list 
    except Exception as e:
        logger.error(f"initialize_user_folders error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def get_project_list(db: AsyncSession, user_id: str, folder_id: str, sort: str = "created_at", order: str = "asc") -> Tuple[bool, any, any]:
    try:
        order_function = asc if order == "asc" else desc
        project_result = await db.execute(
            select(Project, Analysis.recommended_keyword).join(
                Analysis, Project.project_id == Analysis.project_id
            ).where(
                Project.user_id == user_id, 
                Project.folder_id == folder_id,
                Project.del_yn == 'n'
            ).order_by(order_function(getattr(Project, sort)))
        )
        projects = project_result.all()

        projects_as_dicts = []
        for project, recommended_keyword in projects:
            project_dict = {
                'project_id': project.project_id,
                'project_name': project.project_name,
                'project_path': project.project_path,
                'project_type': project.project_type,
                'folder_id': project.folder_id,
                'recent_timestep': project.recent_timestep,
                'recommended_keyword': recommended_keyword
            }
            projects_as_dicts.append(project_dict)
            
        return True, None, projects_as_dicts
    
    except Exception as e:
        logger.error(f"get_project_list error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


# storage 폴더 아이디를 만들어 주는 함수 내부적으로 스키마에 따로 뺄 것 
async def list_folders_by_user_id(db: AsyncSession, user_id: str, sort: str, order: str) -> Tuple[bool, any, any]:
    """
    주어진 사용자 ID에 해당하는 모든 'folders' 레코드를 조회하고, 주어진 정렬 기준에 따라 결과를 정렬
    """
    try:
        sort_mapping = {
            "folder_name": Folder.folder_name,
            "created_at": Folder.created_at,
            "updated_at": Folder.updated_at,
            "folder_index": Folder.folder_index
        }

        if sort not in sort_mapping:
            raise ValueError("Invalid sort parameter")

        order_by_field = sort_mapping[sort]
        order_function = asc if order == "asc" else desc

        query = (
            select(Folder)
            .where(Folder.user_id == user_id, Folder.del_yn == 'n')
            .order_by(order_function(order_by_field))
        )
        result = await db.execute(query)
        folder_info_list = [
            {
                'folder_id': folder.folder_id, 
                'folder_name': folder.folder_name, 
                'folder_index': folder.folder_index
            } 
            for folder in result.scalars().all()
        ]

        return True, None, folder_info_list

    except Exception as e:
        logger.error(f"list_folders_by_user_id error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


# 사용 안 함
# async def get_folder_id_by_name(db: Session, user_id: str, folder_name: str) -> str:
#     folder_result = await db.execute(
#         select(Folder.folder_id).where(Folder.user_id == user_id, Folder.folder_name == folder_name)
#     )
#     folder_id = folder_result.scalar()  
#     return folder_id


async def get_folder_name_by_folder_id(db: AsyncSession, folder_id: str) -> Tuple[bool, any, any]:
    try:
        folder_result = await db.execute(
            select(Folder.folder_name).where(Folder.folder_id == folder_id)
        )
        folder_name = folder_result.scalar()
        return True, None, folder_name
    
    except Exception as e:
        logger.error(f"get_folder_name_by_folder_id error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def get_folder_details(db: AsyncSession, folder_id: str) -> Tuple[bool, any, any]:
    """
    주어진 폴더 ID에 대한 폴더의 세부 정보를 딕셔너리 형태로 반환

    Args:
        folder_id (str): 폴더 ID

    Returns:
        dict: 폴더 세부 정보 (현재는 폴더 이름만 포함, 필요에 따라 확장 가능)
    """
    try:
        query = select(Folder).where(Folder.folder_id == folder_id)
        result = await db.execute(query)
        folder = result.scalar()

        if folder:
            folder_details = {
                "folder_id": folder.folder_id,
                "folder_name": folder.folder_name,
            }
            return True, None, folder_details
        else:
            return True, None, {}

    except Exception as e:
        logger.error(f"get_folder_details error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None
    

async def get_total_folders_and_projects(db: AsyncSession, user_id: str) -> Tuple[bool, any, any]:
    """ 주어진 사용자의 전체 프로젝트와 폴더의 개수를 카운트함 """
    try:
        projects_count = await db.execute(
            select(func.count()).where(Project.user_id == user_id)
        )
        projects_total = projects_count.scalar()

        folders_count = await db.execute(
            select(func.count()).where(Folder.user_id == user_id)
        )
        folders_total = folders_count.scalar()

        return True, None, folders_total, projects_total
    
    except Exception as e:
        logger.error(f"get_total_folders_and_projects error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None, None


async def create_folder(db: AsyncSession, user_id: str, folder_name: str, folder_id: str, folder_index: int) -> Tuple[bool, any, any]:
    try:
        new_folder = Folder(
            folder_id=folder_id,
            user_id=user_id,
            folder_index=folder_index,
            folder_name=folder_name
        )
        db.add(new_folder)
        await db.commit()
        return True, None, True

    except Exception as e:
        logger.error(f"create_folder error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def update_folder_name_by_folder_id(
    db: AsyncSession, 
    folder_id: str, 
    new_folder_name: str
) -> Tuple[bool, any, any]:
    try:
        folder = await db.get(Folder, folder_id)
        if folder:
            folder.folder_name = new_folder_name
            await db.commit()
            return True, None, True
        else:
            logger.error("Folder not found.")
            return True, "Folder not found.", False

    except Exception as e:
        logger.error(f"update_folder_name_by_folder_id error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


# 사용 안 함
# async def delete_folder_by_folder_id(db: Session, folder_id: str) -> bool:
#     try:
#         folder = await db.get(Folder, folder_id)
#         if folder:
#             folder.del_yn = 'y'
#             await db.commit()
#             return True
#         else:
#             logger.error("폴더 삭제에 실패하였습니다.")
#             return False

#     except Exception as e:
#         logger.error(f"delete_folder_by_folder_id error: {e}")
#         await db.rollback()
#         return {"error": str(e)}


async def update_folder_index_by_folder_id(db: AsyncSession, user_id: str, folder_id: str, new_folder_index: int) -> Tuple[bool, any, any]:
    try:
        result = await db.execute(select(Folder).where(Folder.user_id == user_id, Folder.folder_id == folder_id))
        folder = result.scalar()

        if folder:
            folder.folder_index = new_folder_index
            await db.commit()
            return True, None, True
        else:
            logger.error("해당 폴더를 찾을 수 없습니다.")
            return True, None, False

    except Exception as e:
        logger.error(f"update_folder_index_by_folder_id error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def check_folder_is_in_used(db: AsyncSession, user_id: str, folder_name: str) -> Tuple[bool, any, any]:
    """
    주어진 사용자 ID와 폴더 이름에 해당하는 폴더의 중복 여부를 확인

    Args:
        user_id (str): 사용자 ID
        folder_name (str): 폴더 이름

    Returns:
        str: 중복되는 폴더 이름 (없으면 빈 문자열 반환)
    """
    try:
        response = await db.execute(select(Folder).filter(Folder.user_id == user_id, Folder.folder_name == folder_name, Folder.del_yn == 'n'))
        response_folder = response.scalars().first()

        return True, None, response_folder.folder_name if response_folder else ''
    
    except Exception as e:
        logger.error(f"check_folder_is_in_used error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


######################### Project CRUD ###########################
async def create_project(
    db: AsyncSession,
    user_id: str,
    folder_id: str,
    project_id: str,
    project_name: str,
    project_type: str,
    file_path: str
) -> Tuple[bool, any, any]:
    """
    새로운 프로젝트를 생성

    Args:
        user_id (str): 사용자 ID
        folder_name (str): 폴더 이름
        project_name (str): 프로젝트 이름
        file_path (str): 파일 경로

    Returns:
        Project: 생성된 프로젝트 객체
    """
    try:
        new_project = Project(
            project_id=project_id, 
            user_id=user_id,
            folder_id=folder_id,
            project_name=project_name,
            project_path=file_path,
            project_type=project_type, 
            recent_timestep=func.now(),
            # recommended_keyword={},
        )
        db.add(new_project)
        await db.commit()

        return True, None, new_project

    except Exception as e:
        logger.error(f"create_project error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def init_project_analysis(db: AsyncSession, project_id: str) -> Tuple[bool, any, any]:
    """
    주어진 프로젝트 ID에 대해 analysis 테이블의 초기값을 설정합니다.
    """
    try:
        init_analysis = Analysis(project_id=project_id)
        db.add(init_analysis)
        await db.commit()
        return True, None, True
    except Exception as e:
        logger.error(f"init_project_analysis error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None



async def count_target_column_by_user_id(db:AsyncSession, user_id: str, target_column: str) -> Tuple[bool, any, any]:
    """
    주어진 사용자 ID에 해당하는 폴더 또는 프로젝트의 개수를 반환

    Args:
        user_id (str): 사용자 ID
        target_column (str): 대상 열 ('folder' 또는 'project')

    Returns:
        int: 해당하는 폴더 또는 프로젝트의 개수
    """
    try:
        if target_column == 'folder':
            count_query = select(func.count()).where(Folder.user_id == user_id)
        elif target_column == 'project':
            count_query = select(func.count()).where(Project.user_id == user_id)
        else:
            raise ValueError(f"{target_column}이 존재하지 않습니다.")

        result = await db.execute(count_query)
        return True, None, result.scalar()

    except Exception as e:
        logger.error(f"count_target_column_by_user_id error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def get_project_details(db: AsyncSession, project_id: str) -> Tuple[bool, any, any]:
    """
    주어진 프로젝트 ID에 대한 프로젝트 세부 정보를 반환

    Args:
        project_id (str): 프로젝트 ID

    Returns:
        dict: 프로젝트 세부 정보 (프로젝트 ID, 이름, 폴더 ID, 폴더 이름)
    """
    try:
        # Project와 Folder 테이블 조인
        query = select(
            Project.project_id,
            Project.project_name,
            Project.user_id,
            Folder.folder_id,
            Folder.folder_name
        ).join(Folder, Project.folder_id == Folder.folder_id).where(Project.project_id == project_id)
        
        result = await db.execute(query)
        project_details = result.first()

        if project_details:
            return True, None, {
                                    "project_id": project_details.project_id,
                                    "project_name": project_details.project_name,
                                    "folder_id": project_details.folder_id,
                                    "folder_name": project_details.folder_name,
                                    "user_id": project_details.user_id
                                }
        else:
            return True, None, {}
        
    except Exception as e:
        logger.error(f"get_project_details error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def get_project_name_by_project_id(db: AsyncSession, project_id: str) -> Tuple[bool, any, any]:
    try:
        project_result = await db.execute(
            select(Project.project_name).where(Project.project_id == project_id)
        )
        project_name = project_result.scalar()
        return True, None, project_name

    except Exception as e:
        logger.error(f"get_project_name_by_project_id error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def update_project_name_by_project_id(db: AsyncSession, project_id: str, new_project_name: str) -> Tuple[bool, any, any]:
    try:
        project = await db.get(Project, project_id)
        if project:
            project.project_name = new_project_name
            await db.commit()
            return True, None, True
        else:
            logger.error("해당 프로젝트를 찾을 수 없습니다.")
            return True, "해당 프로젝트를 찾을 수 없습니다.", False

    except Exception as e:
        logger.error(f"update_project_name_by_project_id error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None
    

async def delete_project_by_project_id(db: AsyncSession, project_id: str) -> Tuple[bool, any, any]:
    try:
        project = await db.get(Project, project_id)
        if project:
            project.del_yn = 'y'
            await db.commit()
            return True, None, True
        else:
            raise ValueError("해당 프로젝트가 존재하지 않습니다.")
        
    except Exception as e:
        logger.error(f"delete_project_by_project_id error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def delete_folder_and_projects_by_folder_id(
    db: AsyncSession, 
    folder_id: str
) -> Tuple[bool, any, any]:
    try:
        projects = await db.execute(
            select(Project).where(Project.folder_id == folder_id)
        )
        projects_to_update = projects.scalars().all()

        if projects_to_update:
            for project in projects_to_update:
                project.del_yn = 'y'

        await db.execute(
            update(Folder).where(Folder.folder_id == folder_id).values(del_yn='y')
        )
        await db.commit()
        return True, None, True

    except Exception as e:
        logger.error(f"delete_folder_and_projects_by_folder_id error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def move_project_folder(
    db: AsyncSession, 
    project_id: str, 
    new_folder_id: str, 
    new_project_name: str = None
) -> Tuple[bool, any, any]:
    """
    주어진 프로젝트 ID의 프로젝트를 새로운 폴더 ID로 이동

    Args:
        project_id (str): 이동할 프로젝트의 ID
        new_folder_id (str): 프로젝트를 이동할 새로운 폴더의 ID

    Returns:
        bool: 쿼리 실행 성공 여부 (True: 성공, False: 실패)
    """
    try:
        # Project 테이블에서 project_id에 해당하는 레코드의 folder_id 업데이트
        query = (
            update(Project)
            .where(Project.project_id == project_id)
            .values(folder_id=new_folder_id, project_name=new_project_name)
        )

        if new_project_name:
            query = query.values(project_name=new_project_name)

        await db.execute(query)
        await db.commit()
        return True, None, True

    except Exception as e:
        logger.error(f"move_project_folder error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def check_project_is_in_used(db: AsyncSession, user_id: str, folder_name: str, project_name: str) -> Tuple[bool, any, any]:
    """
    주어진 사용자 ID, 폴더 이름, 프로젝트 이름에 해당하는 프로젝트의 중복 여부를 확인

    Args:
        user_id (str): 사용자 ID
        folder_name (str): 폴더 이름
        project_name (str): 프로젝트 이름

    Returns:
        str: 중복되는 프로젝트 이름 (없으면 빈 문자열 반환)
    """
    try:
        response = await db.execute(select(Project).join(Folder, Project.folder_id == Folder.folder_id).filter(
            Folder.folder_name == folder_name, Project.user_id == user_id, Project.project_name == project_name, Project.del_yn == 'n'))
        response_project = response.scalars().first()
        return True, None, response_project.project_name if response_project else ''
    
    except Exception as e:
        logger.error(f"check_project_is_in_used error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def search_projects_by_keyword(db: AsyncSession, user_id: str, search_keyword: str) -> Tuple[bool, any, any]:
    """
    주어진 사용자 ID의 프로젝트 중에서 프로젝트 이름과 추천 키워드에 특정 키워드를 포함하는 프로젝트 정보 검색
    """
    try:
        query = select(
            Project.project_name, Folder.folder_name, Project.project_id, Project.folder_id,
            Project.created_at, Project.project_type, Project.project_path, Analysis.recommended_keyword
        ).join(Folder, Project.folder_id == Folder.folder_id
        ).join(Analysis, Project.project_id == Analysis.project_id
        ).where(
            Project.user_id == user_id,
            Folder.del_yn == 'n',
            Project.del_yn == 'n',
            or_(
                Project.project_name.contains(search_keyword),
                Analysis.recommended_keyword.contains(search_keyword)
            )
        )
        result = await db.execute(query)
        projects = []
        for row in result:
            projects.append({
                "project_name": row.project_name,
                "folder_name": row.folder_name,
                "project_id": row.project_id,
                "folder_id": row.folder_id,
                "created_at": row.created_at,
                "project_type": row.project_type,
                "project_path": row.project_path,
                'recommended_keyword': row.recommended_keyword
            })
        return True, None, projects
    
    except Exception as e:
        logger.error(f"search_projects_by_keyword error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None
