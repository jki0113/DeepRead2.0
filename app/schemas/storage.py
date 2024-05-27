from typing import Any, Dict, List, Literal, Optional

from fastapi import Body, File, Form, HTTPException, Header, Path, Query, UploadFile
from pydantic import BaseModel, constr, validator, Field


############################## USER ##############################
class request_get_user_status(BaseModel):
    user_id : str
class response_get_user_status(BaseModel):
    user_info : Dict


########################## Folder CRUD ###########################
class request_get_user_folders(BaseModel):
    user_id: str
    sort: Optional[str] = 'folder_index'
    order: Optional[str] = 'asc'
class response_get_user_folders(BaseModel):
    folder_list: List[Dict]

class request_create_folder(BaseModel):
    user_id: str
    folder_name: str = '기본 폴더'

    @validator('folder_name')
    def strip_and_default(cls, v):
        """공백을 제거하고 빈 문자열인 경우 'Untitled'으로 대체"""
        if not v.strip():
            return 'Untitled'
        return v.strip()
class response_create_folder(BaseModel):
    folder_id: str
    folder_name: str

class request_update_folder_name(BaseModel):
    user_id : str = 'user_id'
    folder_id : str = 'folder_id'
    new_folder_name : str = 'new_folder_name'

    @validator('new_folder_name', pre=True, always=True)
    def strip_folder_name(cls, v):
        """공백을 제거하고 빈 문자열인 경우 'Untitled'으로 대체"""
        if not v.strip():
            return 'Untitled'
        return v.strip()

class request_delete_folder(BaseModel):
    folder_id : str

class request_move_folder_index(BaseModel):
    user_id: str = 'user_id'
    folder_id: str = 'folder_id'
    move_method: Literal['up', 'down']
class response_move_folder_index(BaseModel):
    folder_list: List[Dict]


######################### Project CRUD ###########################
class request_get_recent_projects(BaseModel):
    user_id : str = 'user_id'
    limit : int = '3'
    sort: Optional[str] = 'created_at'
    order: Optional[str] = 'desc'
class response_get_recent_projects(BaseModel):
    recent_project_list: List[Dict]

class request_list_projects_by_folder(BaseModel):
    user_id: str
    folder_id: str
    sort: Optional[str] = 'created_at'
    order: Optional[str] = 'asc'
class response_list_projects_by_folder(BaseModel):
    project_list: List[Dict[str, Any]] = []

class response_create_project(BaseModel):
    project_id: str

class request_update_project_name(BaseModel):
    user_id : str = 'user_id'
    folder_name : str = 'folder_name'
    project_id : str = 'project_id'
    new_project_name: str = 'new_project_name'

class request_delete_project(BaseModel):
    project_id: str

class request_move_project_folder(BaseModel):
    target_project_id: str = 'project_id'
    target_folder_id: str = 'folder_id'
class response_move_project_folder(BaseModel):
    moved_folder_id: str = 'folder_id'
    moved_folder_name: str = 'folder_name'
    
class request_search_projects_by_keyword(BaseModel):
    user_id: str = 'user_id'
    search_keyword: str = 'keyword'
class response_search_projects_by_keyword(BaseModel):
    searched_project_list: List[Dict]



class response_bool(BaseModel):
    response : bool

class tmp_schema(BaseModel):
    tmp: str

