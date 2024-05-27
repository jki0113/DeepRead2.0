from datetime import datetime
import json
import traceback
from typing import List, Tuple

from sqlalchemy import asc, desc, func, select, update
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db_connection import create_db_connection  # 임시방편
from app.database.models import *
from log.logger_config import log, log_execution_time, logger


async def get_basic_info(db: AsyncSession, project_id: str) -> Tuple[bool, any, any]:
    try:
        query = select(
            Analysis
        ).where(Analysis.project_id == project_id)
        
        result = await db.execute(query)
        analysis_data = result.scalar_one_or_none()
        
        if analysis_data:
            data = {
                "title": analysis_data.title,
                "author": analysis_data.author,
                "abstract": analysis_data.abstract
            }
            return True, None, data
        else:
            return True, None, None
    except Exception as e:
        logger.error(f"get_basic_info error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None
    

async def get_recommended_journals(db: AsyncSession, project_id: str) -> Tuple[bool, any, any]:
    try:
        query = select(
            Analysis
        ).where(Analysis.project_id == project_id)
        
        result = await db.execute(query)
        analysis_data = result.scalar_one_or_none()
        
        if analysis_data:
            data = {
                "recommended_journals": analysis_data.recommended_journal,
            }
            return True, None, data
        else:
            return True, None, None
    except Exception as e:
        logger.error(f"get_recommended_journals error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None
    

async def get_recommended_papers(db: AsyncSession, project_id: str) -> Tuple[bool, any, any]:
    try:
        query = select(
            Analysis
        ).where(Analysis.project_id == project_id)
        
        result = await db.execute(query)
        analysis_data = result.scalar_one_or_none()
        
        if analysis_data:
            data = {
                "recommended_papers": analysis_data.recommended_paper,
            }
            return True, None, data
        else:
            return True, None, None
    except Exception as e:
        logger.error(f"get_related_papers error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def get_analysis_info_(db: AsyncSession, project_id: str) -> Tuple[bool, any, any]:
    try:
        query = select(
            Analysis
        ).where(Analysis.project_id == project_id)
        
        result = await db.execute(query)
        analysis_data = result.scalar_one_or_none()
        
        if analysis_data:
            data = {
                "recommended_title": analysis_data.recommended_title,
                "recommended_keyword": analysis_data.recommended_keyword,
                "recommended_summarize": analysis_data.recommended_summarize,
                "recommended_potential_topics": analysis_data.recommended_potential_topics,
            }
            return True, None, data
        else:
            return True, None, None
    except Exception as e:
        logger.error(f"get_related_papers error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def get_published_info(db: AsyncSession, project_id: str) -> Tuple[bool, any, any]:
    try:
        query = select(
            Analysis
        ).where(Analysis.project_id == project_id)
        
        result = await db.execute(query)
        analysis_data = result.scalar_one_or_none()
        
        if analysis_data:
            data = {
                "published_info": analysis_data.published_info,
            }
            return True, None, data
        else:
            return True, None, None
    except Exception as e:
        logger.error(f"get_published_info error: {e}\n{traceback.format_exc()}")
        await db.rollback()
        return False, str(e), None


async def get_analysis_info(db: AsyncSession, project_id: str, columns: List[str]) -> Tuple[bool, any, any]:
    """플젝 아이디와 컬럼을 입력 받아 입력 받은 컬럼의 분석 결과 추출"""
    async_session_factory = create_db_connection()
    async with async_session_factory() as session_:
        try:
            selected_columns = [getattr(Analysis, column) for column in columns]
            query = select(*selected_columns).where(Analysis.project_id == project_id)

            # result = await db.execute(query)
            result = await session_.execute(query) # 임시방편
            analysis_data = result.fetchall()

            formatted_data = [{column: getattr(row, column) for column in columns} for row in analysis_data]

            return True, None, formatted_data
        
        except Exception as e:
            logger.error(f"get_analysis_info error: {e}")
            await db.rollback()
            return False, e, None
        
        finally:
            await session_.close() #임시방편


async def update_analysis(db: AsyncSession, user_id: str, project_id: str, analysis_data: dict):
    """
    주어진 프로젝트 ID에 해당하는 분석 데이터를 추가하거나 업데이트

    Args:
        user_id (str): 사용자 ID
        project_id (str): 프로젝트 ID
        analysis_data (dict): 분석 데이터

    Returns:
        Analysis: 데이터베이스에 저장된 분석 데이터 객체
    """
    try:
        existing_analysis = await db.get(Analysis, project_id)
        if existing_analysis is None:
            new_analysis = Analysis(
                project_id=project_id,
                **analysis_data
            )
            db.add(new_analysis)
        else:
            for key, value in analysis_data.items():
                setattr(existing_analysis, key, value)
        await db.commit()
        return existing_analysis if existing_analysis else new_analysis
    
    except Exception as e:
        logger.error(f"update_analysis error: {e}")
        await db.rollback()
        return e