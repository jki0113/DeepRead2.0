from fastapi import UploadFile, HTTPException, Body, Query, Path, Header, Form, File
from pydantic import BaseModel, validator, constr
from typing import List, Optional, Dict, Any, Literal


class request_update_analysis_survey(BaseModel):
    project_id: str
    project_type: Literal['draft', 'published']
    recommended_title_satis_yn: Optional[Literal['y', 'n', None]] = None
    recommended_keyword_satis_yn: Optional[Literal['y', 'n', None]] = None
    recommended_summarize_satis_yn: Optional[Literal['y', 'n', None]] = None
    recommended_potential_topics_satis_yn: Optional[Literal['y', 'n', None]] = None
    recommended_journal_satis_yn: Optional[Literal['y', 'n', None]] = None
    recommended_paper_satis_yn: Optional[Literal['y', 'n', None]] = None
    published_info_satis_yn: Optional[Literal['y', 'n', None]] = None
    comment: Optional[str] = None

class request_update_tutorial_survey(BaseModel):
    user_id: str
    tutorial_status: str

class request_update_satisfaction_survey(BaseModel):
    satisfaction_survey_round: Literal[3, 7]
    user_id: str
    satisfaction_answer: str
    satisfaction_comment: str

class response_bool(BaseModel):
    response: bool