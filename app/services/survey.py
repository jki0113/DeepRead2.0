import os
from sqlalchemy.orm import Session
from typing import Tuple, Optional

from app.schemas import survey as schemas
from app.queries import survey as query


async def survey_analysis_draft(
    db: Session,
    project_id: str,
    recommended_title_satis_yn: Optional[str],
    recommended_keyword_satis_yn: Optional[str],
    recommended_summarize_satis_yn: Optional[str],
    recommended_journal_satis_yn: Optional[str],
    recommended_paper_satis_yn: Optional[str],
    comment: Optional[str]
) -> Tuple[bool, any, any]:
    success, ermsg, response = await query.create_survey_analysis(
        db,
        project_id=project_id,
        recommended_title_satis_yn=recommended_title_satis_yn,
        recommended_keyword_satis_yn=recommended_keyword_satis_yn,
        recommended_summarize_satis_yn=recommended_summarize_satis_yn,
        recommended_journal_satis_yn=recommended_journal_satis_yn,
        recommended_paper_satis_yn=recommended_paper_satis_yn,
        comment=comment
    )
    if not success:
        return False, ermsg, None
    
    return True, None, response


async def survey_analysis_published(
    db: Session,
    project_id: str,
    recommended_keyword_satis_yn: Optional[str],
    recommended_summarize_satis_yn: Optional[str],
    recommended_paper_satis_yn: Optional[str],
    recommended_potential_topics_satis_yn: Optional[str],
    published_info_satis_yn: Optional[str],
    comment: Optional[str]
) -> Tuple[bool, any, any]:
    success, ermsg, response = await query.create_survey_analysis(
        db,
        project_id=project_id,
        recommended_keyword_satis_yn=recommended_keyword_satis_yn,
        recommended_summarize_satis_yn=recommended_summarize_satis_yn,
        recommended_paper_satis_yn=recommended_paper_satis_yn,
        recommended_potential_topics_satis_yn=recommended_potential_topics_satis_yn,
        published_info_satis_yn=published_info_satis_yn,
        comment=comment
    )
    if not success:
        return False, ermsg, None
    
    return True, None, response


async def update_tutorial_status(db: Session, user_id: str, tutorial_status: str) -> Tuple[bool, any, any]:
    survey_entries = [{
        'answer_id': os.getenv('SURVEY_TUTORIAL_Q1').replace('SQ_', f'SA_{user_id}_'),
        'question_id': os.getenv('SURVEY_TUTORIAL_Q1'),
        'answer': tutorial_status,
        'user_id': user_id
    }]
    
    success, ermsg, response = await query.update_survey_answers(db, survey_entries)
    if not success:
        return False, ermsg, None
    
    return True, None, response


async def update_satisfaction_survey(
    db: Session,
    satisfaction_survey_round: int,
    user_id: str,
    satisfaction_answer: str,
    satisfaction_comment: str
) -> Tuple[bool, any, any]:
    if satisfaction_survey_round == 3:
        survey_q1 = os.getenv('SURVEY_SATISFACTION_V1_Q1')
        survey_q2 = os.getenv('SURVEY_SATISFACTION_V1_Q2')
    elif satisfaction_survey_round == 7:
        survey_q1 = os.getenv('SURVEY_SATISFACTION_V2_Q1')
        survey_q2 = os.getenv('SURVEY_SATISFACTION_V2_Q2')
    else:
        return True, "만족도 조사가 존재하지 않습니다.", False

    survey_entries = [
        {
            'answer_id': survey_q1.replace('SQ_', f'SA_{user_id}_'),
            'question_id': survey_q1,
            'answer': satisfaction_answer,
            'user_id': user_id
        },
        {
            'answer_id': survey_q2.replace('SQ_', f'SA_{user_id}_'),
            'question_id': survey_q2,
            'answer': satisfaction_comment[:2500],
            'user_id': user_id
        }
    ]

    success, ermsg, response = await query.update_survey_answers(db, survey_entries)
    if not success:
        return False, ermsg, None
    
    return True, None, response


async def get_project_type(db: Session, project_id: str) -> Tuple[bool, any, any]:
    success, ermsg, project_type = await query.get_project_type(db, project_id=project_id)
    if not success:
        return False, ermsg, None
    
    return True, None, project_type