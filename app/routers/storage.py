import os
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, constr, validator
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db_connection import get_async_db
from app.schemas import storage as schemas
from app.services import chat, storage as services
from utils import common
from log.logger_config import log, log_execution_time, logger

router = APIRouter()


############################## USER ##############################
@router.post('/get_user_status', response_model=schemas.response_get_user_status)
@log_execution_time
async def get_user_status(
    request: schemas.request_get_user_status,
    db: AsyncSession = Depends(get_async_db)
):
    """
    유저의 DeepRead 상태 정보를 검색

    - Args:
        - **user_id (str)**: 유저 넥스트비즈 아이코드

    - Returns:
        - **user_info (Dict[str, Any])**
            - **total_folder (int)**: 유저의 총 폴더 갯수
            - **total_project (int)**: 유저의 총 프로젝트 갯수
            - **tutorial_status (str)**: 듀토리얼 확인 여부
            - **satisfaction_survey_2_status (str)**: 2회 만족도 조사 확인 여부
            - **satisfaction_survey_7_status (str)**: 7회 만족도 조사 확인 여부
    """
    # status 값은 to_be_completed, not_required, completed 3가지로 구분
    success, ermsg, user_info = await services.get_user_status(
        db = db, 
        user_id = request.user_id
    )

    if not success or not user_info:
        raise HTTPException(status_code=500, detail=ermsg)

    return user_info


########################## Folder CRUD ###########################
@router.post("/create_folder", response_model=schemas.response_create_folder)
@log_execution_time
async def create_folder(
    request: schemas.request_create_folder, 
    db: AsyncSession = Depends(get_async_db)
):
    """
    유저 폴더 생성

    - Args:
        - **user_id (str)**: 유저 넥스트비즈 아이코드
        - **folder_name (str)**: 유저가 입력한 폴더 이름

    - Returns:
        - **folder_id (str)**: 생성된 폴더의 아이디
        - **folder_name (str)**: 생성된 폴더의 이름
    """
    folder_name = request.folder_name.strip() or 'Untitled'
    folder_name = common.unicode_normalization(folder_name)
    if len(folder_name) > 50:
        raise HTTPException(status_code=400, detail='Folder Name is too long')
    
    success, ermsg, folder_info = await services.create_folder(
        db=db, 
        user_id=request.user_id, 
        folder_name=folder_name
    )
    if not success or not folder_info:
        raise HTTPException(status_code=500, detail=ermsg)
    
    new_folder_id, folder_name = folder_info[0], folder_info[1]


    return schemas.response_create_folder(folder_id=new_folder_id, folder_name=folder_name)


@router.get("/list_folders_by_user_id", response_model=schemas.response_get_user_folders)
@log_execution_time
async def list_folders_by_user_id(
    request: schemas.request_get_user_folders = Depends(), 
    db: AsyncSession = Depends(get_async_db)
): 
    """
    유저 폴더 조회(처음 사용하는 경우 폴더 초기화)

    - Args:
        - **user_id (str)**: 유저 넥스트비즈 아이코드
        - **sort (str)**: 정렬 기준
        - **order (str)**: 정렬 순서 ['ASC', 'DESC']

    - Returns:
        - **folder_list (List[dcit])**: 폴더 리스트
    """
    success, ermsg, folder_list = await services.get_or_initialize_user_folders(
        db=db, 
        user_id=request.user_id,
        sort=request.sort,
        order=request.order,
    )
    
    if not success:
        raise HTTPException(status_code=500, detail=f"{ermsg}")
    
    return schemas.response_get_user_folders(folder_list=folder_list)


@router.post("/update_folder_name", response_model=schemas.response_bool)
@log_execution_time
async def update_folder_name(request: schemas.request_update_folder_name, db: AsyncSession = Depends(get_async_db)):
    """
    기존 폴더의 이름을 업데이트합니다. 새로운 이름이 이미 사용 중인 경우, 고유한 이름을 생성합니다.
    
    - Args:
        - **user_id (str)**: 폴더 소유 유저의 넥스트비즈 아이코드
        - **folder_id (str)**: 이름을 변경할 폴더의 아이디
        - **new_folder_name (str)**: 폴더에 새로 지정할 이름

    - Returns:
        - **response (bool)**: 폴더 이름 변경 성공 여부. 성공시 True, 실패시 False 반환
    """
    new_folder_name = request.new_folder_name.strip() or 'Untitled'
    new_folder_name = common.unicode_normalization(new_folder_name)
    if len(new_folder_name) > 50:
        raise HTTPException(status_code=400, detail='Folder Name is too long')
    
    success, ermsg, response = await services.update_folder_name(
        db=db, 
        user_id = request.user_id,
        folder_id = request.folder_id,
        new_folder_name = new_folder_name,
    )

    if not success:
        raise HTTPException(status_code=500, detail=f"{ermsg}")
    if not response:
        raise HTTPException(status_code=404, detail=ermsg)
    
    return schemas.response_bool(response=response)


@router.post("/delete_folder", response_model=schemas.response_bool)
@log_execution_time
async def delete_folder(request: schemas.request_delete_folder, db: AsyncSession = Depends(get_async_db)):
    """
    제공된 폴더 ID에 기반하여 폴더를 삭제
    
    - Args:
        - **folder_id (str)**: 삭제할 폴더의 아이디
    
    - Returns:
        - **response (bool)**: 폴더 삭제 성공 여부
    """
    success, ermsg, response = await services.delete_folder(
        db=db, 
        folder_id=request.folder_id
    )

    if not success or not response:
        raise HTTPException(status_code=500, detail=ermsg)

    return schemas.response_bool(response=response)

    
@router.post('/move_folder_index', response_model=schemas.response_move_folder_index)
@log_execution_time
async def move_folder_index(request: schemas.request_move_folder_index, db: AsyncSession = Depends(get_async_db)):
    """
    지정된 이동 방법에 따라 폴더의 인덱스를 변경
    
    - Args:
        - **user_id (str)**: 사용자 식별자
        - **folder_id (str)**: 이동할 폴더의 식별자
        - **move_method (Literal['up', 'down'])**: 폴더를 위나 아래로 이동하는 방법
    
    - Returns:
        - **folder_list (List[Dict])**: 새롭게 정렬된 폴더 목록
    """
    success, ermsg, new_folder_index = await services.move_folder_index(
        db=db, 
        user_id=request.user_id,
        folder_id=request.folder_id,
        move_method=request.move_method,
    )
        
    if not success:
        raise HTTPException(status_code=500, detail=ermsg)

    return schemas.response_move_folder_index(folder_list=new_folder_index)


######################### Project CRUD ###########################
@router.get('/get_recent_projects', response_model=schemas.response_get_recent_projects)
@log_execution_time
async def get_recent_porjects(request: schemas.request_get_recent_projects = Depends(), db: AsyncSession = Depends(get_async_db)):
    """
    사용자의 최근 프로젝트 목록을 조회합니다.

    - Args:
        - **user_id (str)**: 사용자 식별자입니다. 조회할 사용자의 고유 ID.
        - **limit (int)**: 반환할 프로젝트의 최대 개수 default > 3
        - **sort (Optional[str])**: 프로젝트 목록을 정렬할 기준 default > created_at
        - **order (Optional[str])**: 정렬 순서 ASC or DESC

    - Returns:
        - **recent_project_list (List[Dict])**: 사용자의 최근 프로젝트 정보를 담은 리스트
    """

    success, ermsg, recent_project_list = await services.get_recent_projects(
        db=db, 
        user_id=request.user_id,
        limit=request.limit,
        sort=request.sort,
        order=request.order,
    )
    
    if not success:
        raise HTTPException(status_code=500, detail=ermsg)
    
    return recent_project_list


@router.get("/list_projects_by_folder", response_model=schemas.response_list_projects_by_folder)
@log_execution_time
async def get_projects_by_folder_id(
    request: schemas.request_list_projects_by_folder=Depends(), 
    db: AsyncSession = Depends(get_async_db)
):
    """
    특정 폴더에 속한 프로젝트 목록을 사용자별로 검색하여 반환합니다.

    해당 함수는 사용자 ID와 폴더 ID를 기반으로 해당 폴더에 있는 프로젝트들의 목록을 검색합니다.
    검색 결과는 선택적으로 정렬 기준과 순서에 따라 정렬될 수 있습니다.

    - Args:
        - **user_id (str)**: 프로젝트 목록을 검색할 사용자의 식별자
        - **folder_id (str)**: 프로젝트 목록을 검색할 폴더의 식별자
        - **sort (Optional[str])**: 프로젝트 목록을 정렬할 기준 default > created_at
        - **order (Optional[str])**: 정렬 순서 ASC or DESC

    - Returns:
        - **project_list (List[Dict[str, Any]])**: 검색된 프로젝트들의 목록
    """
    success, ermsg, project_list = await services.get_project_list(
        db=db, 
        user_id=request.user_id,
        folder_id=request.folder_id,
        sort=request.sort,
        order=request.order,
    )

    if not success:
        raise HTTPException(status_code=500, detail=ermsg)

    return schemas.response_list_projects_by_folder(project_list = project_list)


@router.post("/create_project", response_model=schemas.response_create_project)
@log_execution_time
async def create_project(
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user_id: str = Form(...),
    folder_id: str = Form(...),
    project_file: UploadFile = File(...),
    project_type: str = Form('draft'),
    db: AsyncSession = Depends(get_async_db),
):
    """
    folder_id에 새로운 프로젝트를 생성 함

    - Args:
        - **user_id (str)**: 프로젝트를 생성할 사용자의 식별자
        - **folder_id (str)**: 프로젝트를 생성할 폴더의 식별자
        - **project_file (UploadFile)**: 생성할 프로젝트의 파일
        - **project_type (str)**: 프로젝트의 유형 default > draft

    - Returns:
        - **project_id (str)**: 생성된 프로젝트의 식별자
    """
    success, ermsg, project_name = await services.validate_document_structure(project_file=project_file)
    if not success:
        logger.error(ermsg)
        raise HTTPException(status_code=400, detail=ermsg)
    
    project_name = project_name.strip() or 'Untitled'
    project_name = common.unicode_normalization(project_name)
    if len(project_name) > 50:
        raise HTTPException(status_code=400, detail='Project Name is too long')

    success, ermsg, project_id = await services.create_project(
                        db,
                        background_tasks=background_tasks,
                        user_id=user_id,
                        folder_id=folder_id,
                        project_name=project_name,
                        project_file=project_file,
                        project_type=project_type,
                    )
    if not success or not project_id:
        raise HTTPException(status_code=500, detail=ermsg)

    return schemas.response_create_project(project_id=project_id)


@router.post("/update_project_name", response_model=schemas.response_bool)
@log_execution_time
async def update_project_name(
    request: schemas.request_update_project_name, 
    db: AsyncSession = Depends(get_async_db)
):
    """
    기존 프로젝트의 이름을 업데이트
    
    - Args:
        - **user_id (str)**: 프로젝트 명을 업데이트할 사용자 식별자
        - **folder_name (str)**: 프로젝트가 속한 폴더의 이름
        - **project_id (str)**: 명칭을 업데이트할 프로젝트의 식별자
        - **new_project_name (str)**: 프로젝트에 부여할 새로운 명칭

    - Returns:
        - **response (bool)**: 프로젝트 명 업데이트의 성공 여부
    """

    new_project_name = request.new_project_name.strip() or 'Untitled'
    new_project_name = common.unicode_normalization(new_project_name)
    if len(new_project_name) > 50:
        raise HTTPException(status_code=400, detail='Project Name is too long')
    
    success, ermsg, response = await services.update_project_name(
        db=db, 
        user_id=request.user_id,
        folder_name=request.folder_name,
        project_id=request.project_id,
        new_project_name=new_project_name,
    )
    
    if not success:
        raise HTTPException(status_code=500, detail=ermsg)
    
    return schemas.response_bool(response=response)


@router.post("/delete_project", response_model=schemas.response_bool)
@log_execution_time
async def delete_project(request: schemas.request_delete_project, db: AsyncSession = Depends(get_async_db)):
    """
    주어진 프로젝트 ID로 프로젝트 삭제

    - Args:
        - **project_id (str)**: 삭제할 프로젝트의 식별자

    - Returns:
        - **response (bool)**: 프로젝트 삭제 성공 여부
    """
    success, ermsg, response = await services.delete_project(
        db=db, 
        project_id=request.project_id,
    )

    if not success:
        raise HTTPException(status_code=500, detail=ermsg)
    
    return schemas.response_bool(response=response)


@router.post('/move_project_folder', response_model=schemas.response_bool)
@log_execution_time
async def move_project_folder(request: schemas.request_move_project_folder, db: AsyncSession = Depends(get_async_db)):
    """
    프로젝트를 다른 폴더로 이동

    - Args:
        - **target_project_id (str)**: 이동할 프로젝트의 ID
        - **target_folder_id (str)**: 목표 폴더의 ID

    - Returns:
        - **response (bool)**: 프로젝트 이동 성공 여부
    """

    success, ermsg, response = await services.move_project_folder(
        db=db, 
        target_project_id=request.target_project_id,
        target_folder_id=request.target_folder_id,
    )

    if not success:
        raise HTTPException(status_code=500, detail=ermsg)
    
    return schemas.response_bool(response=response)


@router.post('/search_projects_by_keyword', response_model=schemas.response_search_projects_by_keyword)
@log_execution_time
async def search_projects(
    request: schemas.request_search_projects_by_keyword, 
    db: AsyncSession = Depends(get_async_db)
):
    """
    키워드를 사용해 프로젝트 검색

    - Args:
        - **user_id (str)**: 검색을 요청한 사용자의 ID
        - **search_keyword (str)**: 검색할 키워드

    - Returns:
        - **searched_project_list**: 키워드 검색 결과에 해당하는 프로젝트 목록
    """

    success, ermsg, search_result = await services.search_projects_by_keyword(
        db=db, 
        user_id=request.user_id,
        search_keyword=request.search_keyword,
    )

    if not success:
        raise HTTPException(status_code=500, detail=ermsg)
    
    return schemas.response_search_projects_by_keyword(
            searched_project_list = search_result
        )
