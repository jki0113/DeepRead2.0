import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, constr, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.database.db_connection import get_async_db, get_async_db_read_committed
from app.schemas import analysis as schemas
from app.services import analysis as services
from log.logger_config import log, log_execution_time, logger


router = APIRouter()

@router.get("/get_document_info", response_model=schemas.response_get_basic_info)
@log_execution_time
async def get_basic_info(
    request: schemas.request_get_basic_info = Depends(), 
    db: AsyncSession = Depends(get_async_db_read_committed)
):
    """
    프로젝트의 문서 기본 정보를 검색

    - Args:
        - **project_id (str)**: 프로젝트 ID
        - **response_language (Optional[Literal['Korean', 'English', 'Chinese', 'Japanese']])**: 응답 언어 (기본값: None)

    - Returns:
        - **title (str)**: 문서의 제목
        - **author (str)**: 문서의 저자
        - **abstract (str)**: 문서의 요약
    """
    success, ermsg, document_basic_info = await services.get_basic_info(
        db=db, 
        project_id=request.project_id, 
    )
    if not success:
        raise HTTPException(status_code=500, detail=ermsg)
    return schemas.response_get_basic_info(
                title=document_basic_info["title"],
                author=document_basic_info["author"],
                abstract=document_basic_info["abstract"],
            )


@router.get("/get_recommended_journal", response_model=schemas.response_get_recommended_journal)
@log_execution_time
async def get_recommended_journals(
    request: schemas.request_get_recommended_journal = Depends(), 
    db: AsyncSession = Depends(get_async_db_read_committed)
):
    """
    프로젝트의 저널 추천 정보를 검색

    - Args:
        - **project_id (str)**: 프로젝트 ID
        - **response_language (Optional[Literal['Korean', 'English', 'Chinese', 'Japanese']])**: 응답 언어 (기본값: None)

    - Returns:
        - **recommended_journal (List[Dict])**: 추천된 저널 정보 리스트
    """
    success, ermsg, recommended_journals = await services.get_recommended_journals(
        db=db, 
        project_id=request.project_id, 
    )
    if not success:
        raise HTTPException(status_code=500, detail=ermsg)

    return schemas.response_get_recommended_journal(
        recommended_journal=recommended_journals["recommended_journals"],
    )


@router.get("/get_related_paper", response_model=schemas.response_get_related_paper)
@log_execution_time
async def get_related_papers(
    request: schemas.request_get_related_paper = Depends(), 
    db: AsyncSession = Depends(get_async_db_read_committed)
):
    """
    프로젝트의 관련 레퍼런스 정보를 검색

    - Args:
        - **project_id (str)**: 프로젝트 ID
        - **response_language (Optional[Literal['Korean', 'English', 'Chinese', 'Japanese']])**: 응답 언어 (기본값: None)

    - Returns:
        - **recommended_journal (List[Dict])**: 추천된 저널 정보 리스트
    """

    success, ermsg, recommended_papers = await services.get_recommended_papers(
        db=db, 
        project_id=request.project_id, 
    )

    if not success:
        raise HTTPException(status_code=500, detail=ermsg)

    return schemas.response_get_related_paper(
        recommended_paper=recommended_papers["recommended_papers"],
    )


@router.get("/get_analysis_info", response_model=schemas.response_get_analysis_info)
@log_execution_time
async def get_analysis_info(
    request: schemas.request_get_analysis_info = Depends(), 
    db: AsyncSession = Depends(get_async_db_read_committed)
):
    """
    프로젝트의 분석 정보를 검색(제목추천, 키워드 추천, 요약, 추가연구 주제)
  
    - Args:
        - **project_id (str)**: 프로젝트 ID
        - **response_language (Optional[Literal['Korean', 'English', 'Chinese', 'Japanese']])**: 응답 언어 (기본값: None)

    - Returns:
        - **recommended_title (List[str])**: 추천된 저널 정보 리스트
        - **recommended_keyword (List[str])**: 추천된 저널 정보 리스트
        - **recommended_summarize (Dict[str, str])**: 추천된 저널 정보 리스트
        - **recommended_potential_topics (List[str])**: 추천된 저널 정보 리스트
    """
    success, ermsg, analysis_info = await services.get_analysis_info_(
        db=db, 
        project_id=request.project_id, 
        # response_language=request.response_language,
    )
    if not success:
        raise HTTPException(status_code=500, detail=ermsg)

    success, ermsg, analysis_detail_info = await services.get_analysis_detail_info(
        analysis_info=analysis_info,
        response_language = request.response_language
    )
    if not success:
        raise HTTPException(status_code=500, detail="문서의 상세 번역 조회에 실패했습니다.")
    
    return schemas.response_get_analysis_info(
        recommended_title=analysis_info['recommended_title'],
        recommended_keyword=analysis_info['recommended_keyword'],
        recommended_summarize=analysis_info['recommended_summarize'],
        recommended_potential_topics=analysis_info['recommended_potential_topics'],
    )


@router.get("/get_published_info", response_model=schemas.response_get_published_info)
@log_execution_time
async def get_related_papers(
    request: schemas.request_get_published_info = Depends(), 
    db: AsyncSession = Depends(get_async_db_read_committed)
):
    """
    프로젝트의 출판 정보를 검색

    - Args:
        - **project_id (str)**: 프로젝트 ID
        - **response_language (Optional[Literal['Korean', 'English', 'Chinese', 'Japanese']])**: 응답 언어 (기본값: None)

    - Returns:
        - **published_info (Dict)**: 논문 출판 정보
    """
    
    success, ermsg, published_info = await services.get_published_info(
        db=db,
        project_id=request.project_id, 
    )
    if not success:
        raise HTTPException(status_code=500, detail=ermsg)
    
    ### 제목, 저자 같은 기본 정보는 초기에 뽑은 정보로 덮어 씌움 ###
    success, ermsg, basic_info = await services.get_basic_info(
        db=db,
        project_id=request.project_id, 
    )
    if not success:
        raise HTTPException(status_code=500, detail=ermsg)
    published_info['published_info']['paper_title'] = basic_info['title']
    published_info['published_info']['paper_author'] = basic_info['author']
    published_info['published_info']['paper_abstract'] = basic_info['abstract']

    return schemas.response_get_published_info(
        published_info = published_info['published_info']
    )
