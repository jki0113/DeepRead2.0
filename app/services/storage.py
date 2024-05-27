import os
import tempfile
from datetime import datetime
from glob import glob
import json
from typing import Tuple

from fastapi import BackgroundTasks, UploadFile
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from app.queries import storage as query
from app.schemas import storage as schemas
from app.services.analysis import analyze_paper
from log.logger_config import log, log_execution_time, logger
from utils import chatbot, common, document_processor


############################## USER ##############################
async def get_user_status(
    db: AsyncSession, 
    user_id: str,
) -> Tuple[bool, any, any]:
    success, ermsg, response = await query.get_user_is_in_database(db, user_id=user_id)
    if not success:
        return False, ermsg, None
    # 등록되어있지 않은 유저로 처음 서비스를 이용하는 유저입니다.
    if not response:
        success, ermsg, response = await query.create_user(db, user_id=user_id)
        if not success:
            return False, ermsg, None

        # 유저의 상태 정보 초기화
        return True, None, schemas.response_get_user_status(
            user_info={
                'total_folder': 0,
                'total_project': 0,
                'tutorial_status': 'to_be_completed', # 듀토리얼 확인 여부
                'satisfaction_survey_2_status': 'not_required', # 2회 이상 만족도 조사 여부
                'satisfaction_survey_7_status': 'not_required', # 7회 이상 만족도 조사 여부
            }
        )
    else:
        success, ermsg, user_info = await get_user_info(db, user_id=user_id)
        if not success:
            return False, ermsg, None
        return True, None, schemas.response_get_user_status(user_info=user_info)


async def get_user_info(
    db: AsyncSession, 
    user_id:str
) -> Tuple[bool, any, any]:
    success, ermsg, total_folder, total_project = await query.get_total_folders_and_projects(db, user_id=user_id)
    if not success:
        return False, ermsg, None
    
    # 듀토리얼
    success, ermsg, tutorial_done = await query.check_user_completed_survey(db, user_id=user_id, survey_id=os.getenv('SURVEY_TUTORIAL'))
    if not success:
        return False, ermsg, None
    tutorial_status = "to_be_completed" if not tutorial_done else "completed"
    
    # 설문조사 3회 사용 기준
    success, ermsg, satisfaction_survey_v1_done =await query.check_user_completed_survey(db, user_id=user_id, survey_id=os.getenv('SURVEY_SATISFACTION_V1'))
    if not success:
        return False, ermsg, None
    satisfaction_survey_v1_status = "to_be_completed" if not satisfaction_survey_v1_done and total_project >= 3 else \
                                "completed" if satisfaction_survey_v1_done and total_project >= 3 else \
                                "not_required"
    
    # 설문조사 7회 사용 기준
    success, ermsg, satisfaction_survey_v2_done = await query.check_user_completed_survey(db, user_id=user_id, survey_id=os.getenv('SURVEY_SATISFACTION_V2'))
    if not success:
        return False, ermsg, None
    satisfaction_survey_v2_status = "to_be_completed" if not satisfaction_survey_v2_done and total_project >= 7 else \
                                "completed" if satisfaction_survey_v2_done else \
                                "not_required"

    user_info = {
        'total_folder': total_folder,
        'total_project': total_project,
        'tutorial_status': tutorial_status,
        'satisfaction_survey_v1_status': satisfaction_survey_v1_status,
        'satisfaction_survey_v2_status': satisfaction_survey_v2_status,
    }

    return True, None, user_info


########################## Folder CRUD ###########################
async def get_or_initialize_user_folders(
    db: AsyncSession, 
    user_id: str,
    sort: str,
    order: str,
) -> Tuple[bool, any, any]:
    """
    주어진 사용자 ID의 폴더 레코드를 조회하거나 초기화
    """
    success, ermsg, folder_list = await query.list_folders_by_user_id(
        db=db, 
        user_id=user_id, 
        sort=sort, 
        order=order
    )
    if not success:
        return False, ermsg, None

    if not folder_list:
        success, ermsg, initialized_folder_id = await create_unique_id(
            db=db, 
            user_id=user_id, 
            target_column='folder'
        )
        if not success:
            return False, ermsg, None
        
        success, ermsg, folder_list = await query.initialize_user_folders(\
            db=db, 
            user_id=user_id, 
            folder_id=initialized_folder_id
        )
        if not success:
            return False, ermsg, None
        
    return True, None, folder_list


async def create_folder(
    db: AsyncSession, 
    user_id: str,
    folder_name: str,
) -> Tuple[bool, any, any]:
    """유저 아이디와 입력한 생성할 폴더 이름을 입력 받아 폴더 생성"""

    # 폴더 이름 중복시 유니크 폴더 생성
    success, ermsg, folder_name_to_use = await get_unique_folder_name(db, user_id, folder_name)
    if not success:
        return False, ermsg, None
    
    # 유니크한 폴더 아이디 생성
    success, ermsg, folder_id = await create_unique_id(db, user_id=user_id, target_column="folder")
    if not success:
        return False, ermsg, None
    
    # 폴더의 순서를 항당하기 위해 인덱스 추출
    success, ermsg, folder_index = await query.count_target_column_by_user_id(db, user_id=user_id, target_column="folder")
    if not success:
        return False, ermsg, None
    
    success, ermsg, response = await query.create_folder(
        db,
        user_id = user_id,
        folder_name = folder_name_to_use,
        folder_id = folder_id,
        folder_index= folder_index
    )
    if not success:
        return False, ermsg, None
    
    return True, None, (folder_id, folder_name_to_use)


async def update_folder_name(
    db: AsyncSession, 
    user_id: str,
    folder_id: str,
    new_folder_name: str,
) -> Tuple[bool, any, any]:
    """폴더 이름을 업데이트"""
    # 수정하려는 폴더명이 현재 폴더명과 같을 경우 수정하지 않음
    success, ermsg, current_folder_name = await query.get_folder_name_by_folder_id(
        db=db, 
        folder_id=folder_id
    )
    if not success:
        return False, ermsg, None
    if new_folder_name == current_folder_name:
        logger.warning("new_folder_name is the same as the current_folder_name.")
        return True, None, True
    
    # 중복처리
    success, ermsg, folder_name_to_use = await get_unique_folder_name(
        db=db, 
        user_id=user_id, 
        folder_name=new_folder_name,
    )
    if not success:
        return False, ermsg, None

    success, ermsg, response = await query.update_folder_name_by_folder_id(
        db=db, 
        folder_id=folder_id, 
        new_folder_name=folder_name_to_use
    )
    if not success:
        return False, ermsg, None
    if not response:
        return True, ermsg, False
    
    return True, None, True


async def delete_folder(
    db: AsyncSession, 
    folder_id: str,
) -> Tuple[bool, any, any]:
    """ 폴더 아이디로 폴더 삭제: 하위 프로젝트 일괄 삭제 """
    success, ermsg, response = await query.delete_folder_and_projects_by_folder_id(
        db=db, 
        folder_id=folder_id
    )
    if not success:
        return False, ermsg, None
    return True, None, response


async def move_folder_index(
    db: AsyncSession, 
    user_id: str,
    folder_id: str,
    move_method: str,
) -> Tuple[bool, any, any]:
    success, ermsg, folder_index_list = await query.list_folders_by_user_id(
        db=db, 
        user_id=user_id, 
        sort='folder_index', 
        order='asc'
    )
    
    current_index = next((index for index, folder in enumerate(folder_index_list) if folder['folder_id'] == folder_id), None)
    if move_method == "up" and current_index > 0:
        new_index = current_index - 1
    elif move_method == "down" and current_index < len(folder_index_list) - 1:
        new_index = current_index + 1
    # 위치 변경 불가능한 상태(이미 맨위 or 맨아래)로 현재 위치 반환
    else:
        return True, None, folder_index_list  
    
    # 폴더 인덱스 값 교환
    folder_index_list[current_index]['folder_index'], folder_index_list[new_index]['folder_index'] = folder_index_list[new_index]['folder_index'], folder_index_list[current_index]['folder_index']

    # 프론트에 보내줄 폴더 순서 값 교환
    folder_index_list[current_index], folder_index_list[new_index] = folder_index_list[new_index], folder_index_list[current_index]

    success, ermsg, response = await query.update_folder_index_by_folder_id(db, user_id, folder_index_list[current_index]['folder_id'], folder_index_list[current_index]['folder_index'])
    if not success:
        return False, ermsg, None
    success, ermsg, response = await query.update_folder_index_by_folder_id(db, user_id, folder_index_list[new_index]['folder_id'], folder_index_list[new_index]['folder_index'])
    if not success:
        return False, ermsg, None

    return True, None, folder_index_list


async def get_unique_folder_name(db: AsyncSession, user_id: str, folder_name: str) -> Tuple[bool, any, any]:
    success, ermsg, response = await check_name_is_in_used(db, user_id=user_id, folder_name=folder_name, check_column="folder_name")
    if not success:
        return False, ermsg, None
    
    if response:
        success, ermsg, modified_name = await get_unique_name(db, user_id=user_id, folder_name=folder_name, check_column="folder_name")
        if not success:
            return False, ermsg, None
        return True, None, modified_name
    
    return True, None, folder_name


######################### Project CRUD ###########################
async def get_recent_projects(
    db: AsyncSession, 
    user_id: str,
    limit: str,
    sort: str,
    order: str,
) -> Tuple[bool, any, any]:
    success, ermsg, recent_project_list = await query.get_recent_projects(
        db,
        user_id = user_id,
        limit = limit,
        sort = sort,
        order = order,
    )

    if not success:
        return False, ermsg, None

    return True, None, schemas.response_get_recent_projects(recent_project_list=recent_project_list)


async def get_project_list(
    db: AsyncSession, 
    user_id: str,
    folder_id: str,
    sort: str,
    order: str,
) -> Tuple[bool, any, any]:
    success, ermsg, project_list = await query.get_project_list(
        db,
        user_id=user_id,
        folder_id=folder_id,
        sort=sort,
        order=order,
    )

    filtered_project_list = []
    for project in project_list:
        pdf_file_paths = glob(os.path.join(project['project_path'], '*.pdf'))
        
        if pdf_file_paths:
            pdf_file_path = pdf_file_paths[0]
            project['file_name'] = os.path.basename(pdf_file_path)
            project['file_path'] = pdf_file_path.replace(os.getenv('STORAGE_PATH_SYSTEM'), os.getenv('STORAGE_PATH_URL'))
            filtered_project_list.append(project)
        else:
            logger.info(f"{project['project_id']} 경로가 존재하지 않아 프로젝트를 삭제하였습니다.")
            success, ermsg, response = await query.delete_project_by_project_id(db, project_id=project['project_id'])

            if not success:
                return False, ermsg, None

    return True, None, filtered_project_list


async def create_project(
    db: AsyncSession,
    background_tasks: BackgroundTasks,
    user_id: str,
    folder_id: str,
    project_name: str,
    project_file: UploadFile,
    project_type: str
) -> Tuple[bool, any, any]:
    success, ermsg, folder_name = await query.get_folder_name_by_folder_id(db, folder_id=folder_id)
    if not success:
        return False, ermsg, None
    
    success, ermsg, project_name_to_use = await get_unique_project_name(db, user_id, folder_name, project_name)
    if not success:
        return False, ermsg, None
    
    success, ermsg, project_id = await create_unique_id(db, user_id=user_id, target_column='project')
    if not success:
        return False, ermsg, None

    project_path = await common.save_projects(user_id = user_id, project_id = project_id, project_file = project_file)

    success, ermsg, project = await query.create_project(
                        db,
                        user_id=user_id,
                        folder_id=folder_id,
                        project_id=project_id,
                        project_name=project_name_to_use,
                        project_type=project_type,
                        file_path=os.path.dirname(project_path),
                    )
    if not success:
        return False, ermsg, None
    
    success, ermsg, init_analysis = await query.init_project_analysis(db, project_id=project_id)
    if not success:
        return False, ermsg, None
    if init_analysis:
        background_tasks.add_task(chatbot.get_document_metadata_info, os.path.dirname(project_path))
        background_tasks.add_task(analyze_paper, db, user_id, project_id, os.path.dirname(project_path))
        return True, None, project_id
    else:
        return True, ermsg, False


async def update_project_name(
    db: AsyncSession, 
    user_id: str,
    folder_name: str,
    project_id: str,
    new_project_name: str,
) -> Tuple[bool, any, any]:
    """프로젝트의 이름 수정"""
    # 수정하려는 프로젝트명이 현재 프로젝트명과 같을 경우 수정하지 않고 반환
    success, ermsg, current_project_name = await query.get_project_name_by_project_id(db, project_id=project_id)
    if not success:
        return False, ermsg, None
    if new_project_name == current_project_name:
        return True, None, True

    success, ermsg, project_name_to_use = await get_unique_project_name(db, user_id, folder_name, new_project_name)
    if not success:
        return False, ermsg, None

    success, ermsg, response = await query.update_project_name_by_project_id(db, project_id=project_id, new_project_name=project_name_to_use)
    if not success:
        return False, ermsg, None
    if not response:
        return False, ermsg, False

    return True, None, True
    

async def delete_project(
    db: AsyncSession, 
    project_id: str,
) -> Tuple[bool, any, any]:
    """프로젝트 삭제"""
    success, ermsg, response = await query.delete_project_by_project_id(db, project_id=project_id)
    
    if not success:
        return False, ermsg, None
    
    return True, None, response


async def move_project_folder(
    db: AsyncSession, 
    target_project_id: str,
    target_folder_id: str,
) -> Tuple[bool, any, any]:
    success, ermsg, project_info = await query.get_project_details(db, target_project_id)
    if not success:
        return False, ermsg, None

    success, ermsg, folder_info = await query.get_folder_details(db, target_folder_id)
    if not success:
        return False, ermsg, None

    user_id =project_info['user_id']
    project_name = project_info['project_name']
    new_project_name = project_name # 새로운 프로젝트 이름 초기화
    folder_id_before = project_info['folder_id']
    folder_id_after = folder_info['folder_id']
    folder_name_before = project_info['folder_name']
    folder_name_after = folder_info['folder_name']

    if folder_id_before == folder_id_after:
        return True, None, True
        
    success, ermsg, project_name_is_in_used = await check_name_is_in_used(
        db,
        user_id=user_id, 
        folder_name=folder_name_after,
        project_name=project_name,
        check_column='project_name'
    )
    if not success:
        return False, ermsg, None

    # 중복되는 파일이 있는 경우 새로운 new_project_name 변수 업데이트
    if project_name_is_in_used:
        success, ermsg, new_project_name = await get_unique_name(
            db,
            user_id=user_id,
            folder_name=folder_name_after,
            project_name=project_name,
            check_column='project_name'
        )
        if not success:
            return False, ermsg, None


    success, ermsg, response = await query.move_project_folder(
        db, 
        project_id=target_project_id, 
        new_folder_id=folder_id_after,
        new_project_name=new_project_name,
    )
    if not success:
            return False, ermsg, None
    
    return True, None, response


async def search_projects_by_keyword(
    db: AsyncSession, 
    user_id: str,
    search_keyword: str,
) -> Tuple[bool, any, any]:
    success, ermsg, search_result_list = await query.search_projects_by_keyword(
        db, 
        user_id=user_id, 
        search_keyword=search_keyword
    )
    if not success:
            return False, ermsg, None
    
    for search_result in search_result_list:
        pdf_file_paths = glob(os.path.join(search_result['project_path'], '*.pdf'))

        if pdf_file_paths:
            file_name, extension = os.path.splitext(os.path.basename(pdf_file_paths[0]))
            search_result['file_name'] = f"{file_name}{extension}"
        else:
            search_result['file_name'] = "Untitled.pdf"  # 적절한 기본값 할당

    return True, None, search_result_list


async def get_unique_project_name(db: AsyncSession, user_id: str, folder_name: str, project_name: str) -> Tuple[bool, any, any]:
    success, ermsg, response = await check_name_is_in_used(db, user_id=user_id, folder_name=folder_name, project_name=project_name, check_column="project_name")
    if not success:
        return False, ermsg, None
    
    if response:
        success, ermsg, modified_name = await get_unique_name(db, user_id=user_id, folder_name=folder_name, project_name=project_name, check_column="project_name")
        if not success:
            return False, ermsg, None
        return True, None, modified_name
    
    return True, None, project_name



###################################################################
# 고유한 ID 생성
async def create_unique_id(db: AsyncSession, user_id: str, target_column: str) -> Tuple[bool, any, any]:
    """
    고유한 ID를 생성
    """
    success, ermsg, count = await query.count_target_column_by_user_id(db, user_id, target_column)
    
    if not success:
        return False, ermsg, None
    
    new_index = count + 1
    prefix = "PRJ" if target_column == "project" else "FD"
    random_id = await common.generate_random_string(length=16)
    id_str = f"{prefix}_{datetime.now().strftime('%Y%m%d')}_{random_id}_{str(new_index).zfill(3)}"
    return True, None, id_str


async def check_name_is_in_used(db: AsyncSession, user_id: str, check_column: str, folder_name: str = '', project_name: str = '') -> Tuple[bool, any, any]:
    """폴더/프로젝트 이름 중복 체크"""
    if folder_name.strip() == '':
        folder_name = '기본 폴더'

    if check_column == "project_name":
        success, ermsg, is_in_project_name = await query.check_project_is_in_used(db, user_id, folder_name, project_name)
        if not success:
            return False, ermsg, None
        return True, None, True if is_in_project_name else False
    
    elif check_column == "folder_name":
        success, ermsg, is_in_folder_name = await query.check_folder_is_in_used(db, user_id, folder_name)
        if not success:
            return False, ermsg, None
        return True, None, True if is_in_folder_name else False
    
    else:
        return False, "Invalid check_column value. Must be 'project_name' or 'folder_name'", None


# 폴더 혹은 프로젝트 고유 이름 생성
async def get_unique_name(db: AsyncSession, user_id: str, check_column: str, count: int = 1, folder_name: str='', project_name: str='') -> Tuple[bool, any, any]:
    """폴더 or 프로젝트 이름이 중복되는 경우 유니크한 이름을 생성"""
    # 파일 이름인 경우 확장자와 분리해서 파일 이름만 해야함
    if check_column == "project_name":
        modified_project_name = f"{project_name}({count})"
        success, ermsg, name_exists = await check_name_is_in_used(
            db,
            user_id=user_id,
            folder_name=folder_name,
            project_name=modified_project_name,
            check_column=check_column
        )
        if not success:
            return False, ermsg, None
        
        if name_exists:
            return await get_unique_name(db, user_id=user_id, folder_name=folder_name, project_name=project_name, check_column=check_column, count=count + 1)

        return True, None, modified_project_name

    elif check_column == "folder_name":
        modified_folder_name = f"{folder_name} ({count})"

        success, ermsg, name_exists = await check_name_is_in_used(
            db,
            user_id=user_id,
            folder_name=modified_folder_name,
            check_column=check_column
        )
        if not success:
            return False, ermsg, None
        
        if name_exists:
            return await get_unique_name(db, user_id=user_id, folder_name=folder_name, check_column=check_column, count=count + 1)
            
        return True, None, modified_folder_name

    else:
        return False, "Invalid check_column value. Must be 'project_name' or 'folder_name'", None

async def validate_document_structure(project_file: UploadFile) -> Tuple[bool, any, any]:
    # 확장자 검증
    project_name, extension = os.path.splitext(project_file.filename)
    if extension.lower() not in ['.pdf', '.docx', '.txt']:
        return False, "Unsupported file extension", None
    
    # 파일 크기 검증
    if project_file.size == 0:
        return False, "File size is 0 bytes", None

    # 파일 텍스트 검증
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        content = await project_file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    await project_file.seek(0)
    if extension == '.pdf':
        text = await document_processor.extract_text_from_pdf(file_path=temp_file_path)
    elif extension == '.docx':
        text = await document_processor.extract_text_from_docx(file_path=temp_file_path)
    elif extension == '.txt':
        text = await document_processor.extract_text_from_txt(file_path=temp_file_path)
    os.remove(temp_file_path)

    if len(text) < 2000:
        return False, 'Text extracted from the uploaded file is too short for analysis', None
    
    return True, None, project_name