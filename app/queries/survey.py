import sys
import os
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, select

from app.database.models import *
from log.logger_config import logger, log_execution_time, log

from typing import List, Tuple


async def create_survey_analysis(
    db: Session,
    project_id: str,
    recommended_keyword_satis_yn: str,
    recommended_summarize_satis_yn: str,
    recommended_paper_satis_yn: str,
    recommended_title_satis_yn: Optional[str] = None,
    recommended_potential_topics_satis_yn: Optional[str] = None,
    recommended_journal_satis_yn: Optional[str] = None,
    published_info_satis_yn: Optional[str] = None,
    comment: Optional[str] = None
) -> Tuple[bool, any, any]:
    """ 설문조사 """
    new_survey = AnalysisSatisfaction(
        project_id=project_id, 
        recommended_title_satis_yn=recommended_title_satis_yn,
        recommended_keyword_satis_yn=recommended_keyword_satis_yn,
        recommended_summarize_satis_yn=recommended_summarize_satis_yn,
        recommended_potential_topics_satis_yn=recommended_potential_topics_satis_yn,
        recommended_journal_satis_yn=recommended_journal_satis_yn,
        recommended_paper_satis_yn=recommended_paper_satis_yn,
        published_info_satis_yn=published_info_satis_yn,
        comment=comment
    )
    try:
        db.add(new_survey)
        await db.commit()
        return True, None, True
        
    except Exception as e:
        logger.error(f"create_survey_analysis error: {e}")
        await db.rollback()
        return False, e, None


async def update_survey_answers(db: Session, survey_entries: list) -> Tuple[bool, any, any]:
    """ 여러 SurveyAnswer 레코드를 한 번에 업데이트 """
    try:
        for entry in survey_entries:
            new_survey_answer = SurveyAnswer(
                answer_id=entry['answer_id'],
                question_id=entry['question_id'],
                answer=entry['answer'],
                user_id=entry['user_id']
            )
            await db.merge(new_survey_answer)
        await db.commit()
        return True, None, True

    except Exception as e:
        logger.error(f"update_survey_answers error: {e}")
        await db.rollback()
        return False, e, None


async def get_project_type(db: Session, project_id: str) -> Tuple[bool, any, any]:
    try:
        query = select(Project.project_type).where(Project.project_id == project_id)
        result = await db.execute(query)
        data_row = result.first()
        if data_row:
            project_type = data_row[0]
            return True, None, project_type
        else:
            return True, None, None

    except Exception as e:
        logger.error(f"get_project_type error: {e}")
        await db.rollback()
        return False, e, None
