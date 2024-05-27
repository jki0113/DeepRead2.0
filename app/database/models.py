from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, create_engine, VARCHAR, Enum, ForeignKey, TEXT, create_engine, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
load_dotenv('.env.dev')
import sys
sys.path.insert(0, os.getenv('SYS_ROOT_PATH'))


Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    user_id = Column(VARCHAR(60), primary_key=True, comment="유저 고유 식별 아이디")

class Folder(Base):
    __tablename__ = 'folders'
    folder_id = Column(VARCHAR(60), primary_key=True, comment="폴더 고유 번호")
    folder_index = Column(Integer, nullable=False, comment="폴더 순서")
    folder_name = Column(VARCHAR(100), unique=False, comment="폴더 이름")
    user_id = Column(VARCHAR(60), ForeignKey('users.user_id'), nullable=False, comment="유저 고유 식별 아이디")
    created_at = Column(DateTime, nullable=False, default=func.now(), comment="레코드 생성 시각")
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now(), comment="레코드 마지막 업데이트 시각")
    del_yn = Column(VARCHAR(3), nullable=False, default='n', comment="파일 삭제 여부(y:삭제 o, n: 삭제 x)")

class Project(Base):
    __tablename__ = 'projects'
    project_id = Column(VARCHAR(60), primary_key=True, comment="프로젝트 고유 번호")
    project_name = Column(VARCHAR(600), nullable=False, comment="프로젝트 이름")
    user_id = Column(VARCHAR(60), ForeignKey('users.user_id'), nullable=False, comment="유저 고유 식별 아이디")
    project_path = Column(VARCHAR(1000), nullable=False, comment="storage 내 유저 폴더 경로")
    project_type = Column(Enum('draft', 'published'), nullable=False, comment="작성 중인 논문/게재 된 논문 구분")
    folder_id = Column(VARCHAR(60), ForeignKey('folders.folder_id'), nullable=False, comment="폴더 고유 번호")
    recent_timestep = Column(DateTime, nullable=False, default=func.now()) # 최근 사용일
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    del_yn = Column(VARCHAR(3), nullable=False, default='n', comment="파일 삭제 여부(y:삭제 o, n: 삭제 x)")

class UserChat(Base):
    __tablename__ = 'user_chats'
    # id = Column(Integer, primary_key=True)
    chat_index = Column(VARCHAR(70), primary_key=True, comment="프로젝트의 채팅 인덱스")
    project_id = Column(VARCHAR(60), ForeignKey('projects.project_id'), comment="프로젝트 고유 번호")
    role = Column(VARCHAR(40), nullable=False, comment="챗봇 역할")
    content = Column(JSON, nullable=False, comment="챗봇 내용")
    function = Column(VARCHAR(30), nullable=False, comment="기능")
    response_language = Column(VARCHAR(40), nullable=False, comment="챗봇 응답 언어")
    bookmark = Column(DateTime, comment="북마크 설정 날짜")
    like = Column(DateTime, comment="좋아요 여부")
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

class Analysis(Base):
    __tablename__ = 'analyses'
    project_id = Column(VARCHAR(60), ForeignKey('projects.project_id'), primary_key=True, comment="프로젝트 고유 번호")
    title = Column(VARCHAR(1024), nullable=True, comment='논문 제목')
    author = Column(VARCHAR(2048), nullable=True, comment='논문 저자')
    abstract = Column(TEXT, nullable=True, comment='논문 초록')
    recommended_title = Column(JSON, nullable=True, comment="추출 키워드")
    recommended_keyword = Column(JSON, nullable=True, comment="추출 키워드")
    recommended_summarize = Column(JSON, nullable=True, comment="추출 요약")
    recommended_potential_topics = Column(JSON, nullable=True, comment="추가 연구 제안")
    recommended_journal = Column(JSON, nullable=True, comment="추천 투고 저널")
    recommended_paper = Column(JSON, nullable=True, comment="추천 관련 레퍼런스")
    published_info = Column(JSON, nullable=True, comment="출판 정보")
    created_at = Column(DateTime, nullable=False, default=func.now())

class AnalysisSatisfaction(Base):
    __tablename__ = 'analysis_surveys'
    project_id = Column(VARCHAR(60), ForeignKey('projects.project_id'), primary_key=True, comment="프로젝트 고유 번호")
    recommended_title_satis_yn = Column(VARCHAR(3), nullable=True, comment="추출 키워드 만족도")
    recommended_keyword_satis_yn = Column(VARCHAR(3), nullable=True, comment="추출 키워드 만족도")
    recommended_summarize_satis_yn = Column(VARCHAR(3), nullable=True, comment="추출 요약 만족도")
    recommended_potential_topics_satis_yn = Column(VARCHAR(3), nullable=True, comment="추가 연구 제안 만족도")
    recommended_journal_satis_yn = Column(VARCHAR(3), nullable=True, comment="추천 투고 저널 만족도")
    recommended_paper_satis_yn = Column(VARCHAR(3), nullable=True, comment="추천 관련 레퍼런스 만족도")
    published_info_satis_yn = Column(VARCHAR(3), nullable=True, comment="출판 정보 만족도")
    comment = Column(TEXT, nullable=True, comment="추가 코멘트")
    created_at = Column(DateTime, nullable=False, default=func.now())

class SurveyList(Base):
    __tablename__ = 'survey_list'
    survey_id = Column(VARCHAR(60), primary_key=True)
    survey_title = Column(VARCHAR(600), nullable=False)
    survey_description = Column(VARCHAR(600), nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())


class SurveyQuestions(Base):
    __tablename__ = 'survey_questions'
    question_id = Column(VARCHAR(60), primary_key=True)
    survey_id = Column(VARCHAR(60), ForeignKey('survey_list.survey_id'), nullable=False)
    question = Column(VARCHAR(1000), nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())

class SurveyAnswer(Base):
    __tablename__ = 'survey_answers'
    answer_id = Column(VARCHAR(60), primary_key=True)
    question_id = Column(VARCHAR(60), ForeignKey('survey_questions.question_id'), nullable=False)
    answer = Column(VARCHAR(6000))
    user_id = Column(VARCHAR(60), ForeignKey('users.user_id'), nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())
