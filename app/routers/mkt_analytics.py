import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, constr, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from sqlalchemy import func, distinct

from app.database.models import *
from app.database.db_connection import get_async_db, get_async_db_read_committed
from app.schemas import analysis as schemas
from app.services import analysis as services
from log.logger_config import log, log_execution_time, logger

router = APIRouter()

def iterfile(file_like):  # 제너레이터 함수 정의
    try:
        yield from file_like
    finally:
        file_like.close()

@router.get("/download_analytics")
@log_execution_time
async def download_analytics(db: AsyncSession = Depends(get_async_db_read_committed)):
    try:
        # Extracting project data
        project_result = await db.execute(select(Project.project_id, Project.project_type, Project.created_at, Project.del_yn))
        project_data = project_result.fetchall()
        project_df = pd.DataFrame(project_data, columns=project_result.keys()) if project_data else pd.DataFrame(columns=[col.key for col in Project.__table__.columns])

        # Extracting chat data
        chat_result = await db.execute(select(UserChat.chat_index, UserChat.role, UserChat.function, UserChat.response_language, UserChat.created_at))
        chat_data = chat_result.fetchall()
        chat_df = pd.DataFrame(chat_data, columns=chat_result.keys()) if chat_data else pd.DataFrame(columns=[col.key for col in UserChat.__table__.columns])

        # Extracting survey response data
        survey_response_result = await db.execute(
            select(
                SurveyQuestions.question,
                SurveyAnswer.answer,
                func.count().label('response_count')
            )
            .join(SurveyAnswer, SurveyQuestions.question_id == SurveyAnswer.question_id)
            .group_by(SurveyQuestions.question, SurveyAnswer.answer)
        )
        survey_response_data = survey_response_result.fetchall()
        survey_response_df = pd.DataFrame(survey_response_data, columns=survey_response_result.keys()) if survey_response_data else pd.DataFrame(columns=['question', 'answer', 'response_count'])

        # Extracting survey participant count data
        participant_count_result = await db.execute(
            select(
                SurveyList.survey_title,  
                SurveyList.survey_description,  
                func.count(distinct(SurveyAnswer.user_id)).label('participant_count')
            )
            .join(SurveyQuestions, SurveyList.survey_id == SurveyQuestions.survey_id)
            .join(SurveyAnswer, SurveyQuestions.question_id == SurveyAnswer.question_id)
            .group_by(SurveyList.survey_title, SurveyList.survey_description)
        )
        participant_count_data = participant_count_result.fetchall()
        participant_count_df = pd.DataFrame(participant_count_data, columns=participant_count_result.keys()) if participant_count_data else pd.DataFrame(columns=['survey_title', 'survey_description', 'participant_count'])

        # Extracting analysis satisfaction data
        analysis_satisfaction_result = await db.execute(
            select(
                AnalysisSatisfaction.project_id,
                AnalysisSatisfaction.recommended_title_satis_yn,
                AnalysisSatisfaction.recommended_keyword_satis_yn,
                AnalysisSatisfaction.recommended_summarize_satis_yn,
                AnalysisSatisfaction.recommended_potential_topics_satis_yn,
                AnalysisSatisfaction.recommended_journal_satis_yn,
                AnalysisSatisfaction.recommended_paper_satis_yn,
                AnalysisSatisfaction.published_info_satis_yn,
                AnalysisSatisfaction.comment,
                AnalysisSatisfaction.created_at
            )
        )
        analysis_satisfaction_data = analysis_satisfaction_result.fetchall()
        analysis_satisfaction_df = pd.DataFrame(analysis_satisfaction_data, columns=analysis_satisfaction_result.keys()) if analysis_satisfaction_data else pd.DataFrame(columns=[col.key for col in AnalysisSatisfaction.__table__.columns])

        # Saving DataFrames to an Excel file
        b_io = BytesIO()
        with pd.ExcelWriter(b_io, engine='openpyxl') as writer:
            project_df.to_excel(writer, sheet_name='Projects', index=False)
            chat_df.to_excel(writer, sheet_name='User Chats', index=False)
            survey_response_df.to_excel(writer, sheet_name='Survey Responses', index=False)
            participant_count_df.to_excel(writer, sheet_name='Participant Counts', index=False)
            analysis_satisfaction_df.to_excel(writer, sheet_name='Analysis Satisfaction', index=False)

        b_io.seek(0)
        response = StreamingResponse(iterfile(b_io), media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response.headers["Content-Disposition"] = "attachment; filename=analytics_data.xlsx"
        return response

    except Exception as e:
        logger.error(f"An error occurred: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))