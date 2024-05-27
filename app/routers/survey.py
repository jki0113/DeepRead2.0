import os
from pydantic import BaseModel, validator, constr
from log.logger_config import logger, log_execution_time, log
from fastapi import APIRouter, UploadFile, Form, File, Depends, BackgroundTasks, HTTPException
from typing import List, Optional, Dict, Union, Tuple, Set, Any

from app.services import survey as services
from app.schemas import survey as schemas

from sqlalchemy.orm import Session
from app.database.db_connection import get_async_db


router = APIRouter()

@router.post("/update_tutorial_survey", response_model=schemas.response_bool)
@log_execution_time
async def update_tutorial_status(request: schemas.request_update_tutorial_survey, db: Session = Depends(get_async_db)):
    """
    유저아이디를 기반으로 듀토리얼 상태를 완료로 변경(completed, to_be_completed, not_required)
    """
    success, ermsg, response = await services.update_tutorial_status(
        db, request.user_id, request.tutorial_status
    )

    if not success:
        raise HTTPException(status_code=500, detail="듀토리얼 상태 업데이트에 실패했습니다.")
 
    return schemas.response_bool(response=response)


@router.post("/update_satisfaction_survey", response_model=schemas.response_bool)
@log_execution_time
async def check_user_is_in_first(request: schemas.request_update_satisfaction_survey, db: Session = Depends(get_async_db)):
    """
    유저아이디를 기반으로 만족도 조사 상태를 완료로 변경(completed, to_be_completed, not_required)
    """
    success, ermsg, response = await services.update_satisfaction_survey(
        db,
        request.satisfaction_survey_round,
        request.user_id,
        request.satisfaction_answer,
        request.satisfaction_comment
    )

    if not success:
        raise HTTPException(status_code=500, detail="만족도 조사 상태 업데이트에 실패했습니다.")
    if not response:
        raise HTTPException(status_code=404, detail="만족도 조사가 존재하지 않습니다.")
    
    return schemas.response_bool(response=response)


# 나중에 get 으로 바꿀 것 이름 list_folders_by_user_id
@router.post("/update_analysis_survey", response_model=schemas.response_bool)
@log_execution_time
async def update_analysis_survey(request: schemas.request_update_analysis_survey, db: Session = Depends(get_async_db)): 
    """
    분석결과 설문조사
    project_id와 project_type을 입력 받아
    각각의 분석결과에 대한 만족도 저장
    """
    # 일단 프로젝트 아이디로 프로젝트 타입을 찾아서 일치하는지 확인해야 함(필요 없으면 나중에 삭제 해도 될 듯)
    success, ermsg, project_type = await services.get_project_type(db, project_id = request.project_id)
    if not success:
        raise HTTPException(status_code=500, detail="만족도 조사 정보 업데이트에 실패했습니다.")
    if request.project_type != project_type:
        raise HTTPException(status_code=400, detail="프로젝트 타입이 일치하지 않습니다.")

    # 프로젝트 타입 확정되면 필요없는 부분은 제거?

    # 나중에 프로젝트 아이디 조회해서 없는 아이디면 에러 나게 해야함?
    if request.project_type == 'draft':
        success, ermsg, response = await services.survey_analysis_draft(
            db,
            request.project_id,
            request.recommended_title_satis_yn,
            request.recommended_keyword_satis_yn,
            request.recommended_summarize_satis_yn,
            request.recommended_journal_satis_yn,
            request.recommended_paper_satis_yn,
            request.comment
        )
        if not success:
            raise HTTPException(status_code=500, detail="draft 프로젝트의 만족도 조사 정보 업데이트에 실패했습니다.")
    
    elif request.project_type == 'published':
        success, ermsg, response = await services.survey_analysis_published(
            db,
            request.project_id,
            request.recommended_keyword_satis_yn,
            request.recommended_summarize_satis_yn,
            request.recommended_paper_satis_yn,
            request.recommended_potential_topics_satis_yn,
            request.published_info_satis_yn,
            request.comment
        )
        if not success:
            raise HTTPException(status_code=500, detail="published 프로젝트의 만족도 조사 정보 업데이트에 실패했습니다.")
    
    return schemas.response_bool(response=response)

