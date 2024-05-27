from fastapi import UploadFile, HTTPException, Body, Query, Path, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal


class request_get_basic_info(BaseModel):
    project_id: str
    response_language: Optional[Literal['Korean', 'English', 'Chinese', 'Japanese']] = None
class response_get_basic_info(BaseModel):
    title: str
    author: str
    abstract: str

class request_get_recommended_journal(BaseModel):
    project_id: str
    response_language: Optional[Literal['Korean', 'English', 'Chinese', 'Japanese']] = None
class response_get_recommended_journal(BaseModel):
    recommended_journal: List[Dict]

class request_get_related_paper(BaseModel):
    project_id: str
    response_language: Optional[Literal['Korean', 'English', 'Chinese', 'Japanese']] = None
class response_get_related_paper(BaseModel):
    recommended_paper: List[Dict]

class request_get_analysis_info(BaseModel):
    project_id: str
    response_language: Optional[Literal['Korean', 'English', 'Chinese', 'Japanese']] = None
class response_get_analysis_info(BaseModel):
    recommended_title : List[str]
    recommended_keyword : List[str]
    recommended_summarize : Dict[str, str]
    recommended_potential_topics : List[str]

class request_get_published_info(BaseModel):
    project_id: str
    response_language: Optional[Literal['Korean', 'English', 'Chinese', 'Japanese']] = None
class response_get_published_info(BaseModel):
    published_info : Dict