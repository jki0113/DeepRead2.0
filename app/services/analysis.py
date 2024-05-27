import os, sys
import asyncio
import json
from glob import glob
from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from log.logger_config import logger, log_execution_time, log

from utils.async_gpt import *
from utils import translator, common, paper_analysis, chatbot, price_logger
from app.queries import analysis as query
from app.schemas import analysis as schemas

MAX_ATTEMPTS_DEFAULT = 60
LATENCY_DEFAULT = 3

async def analyze_paper(db: AsyncSession, user_id: str, project_id: str, storage_path):
    pdf_path = glob(f"{storage_path}/*.pdf")[0]
    full_text, _ = chatbot.get_full_text(pdf_path, 'PyMuPDFLoader')
    preprocessed_text = chatbot.preprocess_text(full_text, 5000)

    # 기본 정보 추출
    result_dict = await paper_analysis.get_paper_basic_info(preprocessed_text)
    await query.update_analysis(db, user_id, project_id, result_dict)
    
    abstract = preprocessed_text[:4000] if result_dict['abstract'] == 'N/A' else result_dict['abstract']

    # 제목, 요약, 연구주제, 키워드
    extract_title_keywords_summarization = await paper_analysis.extract_title_keywords_summarization(abstract)
    result_dict.update({
        'recommended_title': extract_title_keywords_summarization[0],
        'recommended_keyword': extract_title_keywords_summarization[1],
        'recommended_summarize': extract_title_keywords_summarization[2],
        'recommended_potential_topics': extract_title_keywords_summarization[3],

        'recommended_title_price': extract_title_keywords_summarization[4][0], # for pirce logger
        'recommended_keyword_price': extract_title_keywords_summarization[4][1], # for pirce logger
        'recommended_summarize_price': extract_title_keywords_summarization[4][2], # for pirce logger
        'recommended_potential_topics_price': extract_title_keywords_summarization[4][3], # for pirce logger
    })
    await query.update_analysis(db, user_id, project_id, result_dict)

    # 저널추천
    # result_dict['recommended_journal'] = await paper_analysis.journal_recommender(result_dict['title'] + '\n' + abstract) # for pirce logger
    result_dict['recommended_journal'], result_dict['recommended_journal_price'] = await paper_analysis.journal_recommender(result_dict['title'] + '\n' + abstract) # for pirce logger
    await query.update_analysis(db, user_id, project_id, result_dict)
    
    # 관련 레퍼런스 추천
    result_dict['recommended_paper'] = await paper_analysis.paper_recommender(result_dict['recommended_keyword'][:5])
    await query.update_analysis(db, user_id, project_id, result_dict)

    # 출판정보
    published_info = await paper_analysis.get_published_information_semantic_scholar(title = result_dict['title'])
    result_dict['published_info'] = published_info
    await query.update_analysis(db, user_id, project_id, result_dict)

    # 240516 price logger 중지 
    # price_logger.add_analysis_log(
    #     mode = os.getenv("MODE"),
    #     project_id = project_id,
    #     file_name = os.path.basename(pdf_path),
    #     character= len(full_text),
    #     paper_basic_info_price=result_dict['paper_basic_info_price'],
    #     recommended_title_price=result_dict['recommended_title_price'],
    #     recommended_keyword_price=result_dict['recommended_keyword_price'],
    #     recommended_summarize_price=result_dict['recommended_summarize_price'],
    #     recommended_potential_topics_price=result_dict['recommended_potential_topics_price'],
    #     recommended_journal_price=result_dict['recommended_journal_price'],
    # )


async def get_basic_info(
    db: AsyncSession,
    project_id: str,
    max_attempts: int = MAX_ATTEMPTS_DEFAULT,
    latency: int = LATENCY_DEFAULT,
) -> Tuple[bool, any, any]:
    """논문의 기본정보인 제목, 저자, 초록을 디비에서 불러옵니다."""
    attempt_count = 0
    while attempt_count < max_attempts:
        try:
            success, ermsg, document_basic_info = await query.get_basic_info(
                db=db, 
                project_id=project_id, 
            )
            if not success:
                return success, ermsg, None
            
            # 문서 정보에서 None 값이 포함된 경우 에러 발생
            if any(value is None for value in document_basic_info.values()):
                raise ValueError("Document info contains None value")
            
            return True, None, document_basic_info

        except Exception as e:
            attempt_count += 1
            await asyncio.sleep(latency)

    # MAX ATTEMPT가 끝난 후 None 값이 여전히 존재할 경우 N/A로 대체
    logger.warning("Failed to retrieve basic info after maximum attempts. Returning default N/A values.")
    for key in document_basic_info:
        if document_basic_info[key] is None:
            document_basic_info[key] = "N/A"
    return True, None, document_basic_info


async def get_recommended_journals(
    db: AsyncSession,
    project_id: str,
    max_attempts: int = MAX_ATTEMPTS_DEFAULT,
    latency: int = LATENCY_DEFAULT,
) -> Tuple[bool, any, any]:
    """저널 추천 정보를 데이터베이스에서 불러옵니다."""
    attempt_count = 0
    while attempt_count < max_attempts:
        try:
            success, ermsg, recommended_journals = await query.get_recommended_journals(
                db=db, 
                project_id=project_id
            )
            if not success:
                return success, ermsg, None
            
            # 데이터가 None이거나 리스트가 비어 있는 경우 재시도
            if recommended_journals is None or not recommended_journals["recommended_journals"]:
                raise ValueError("Recommended journals list is empty or not ready yet")

            # 이미지 URL 경로 변환 로직
            for item in recommended_journals["recommended_journals"]:
                if 'image_url' in item:
                    item['image_url'] = item['image_url'].replace(os.getenv('IMG_PATH_SYSTEM'), os.getenv('IMG_PATH_URL'))

            return True, None, recommended_journals

        except Exception as e:
            attempt_count += 1
            await asyncio.sleep(latency)

    # 모든 시도가 실패한 후의 처리 >>> 분석 파트에서 어떠한 이유로 에러가 나 DB에 Null 값 있는 경우
    logger.warning("Failed to retrieve recommended journals after maximum attempts. Returning default N/A values.")
    default_journal = {
        "JCI": "N/A", "SJR": "N/A", "link": "N/A", "score": "N/A",
        "title": "N/A", "H index": "N/A", "image_url": "N/A",
        "Impact Factor": "N/A", "Journal Index": "N/A", "Citations Count": "N/A",
        "Issues Per Year": "N/A", "Topic Relevance": "N/A", "Cite Score (2022)": "N/A",
        "Frequency Preference": "N/A", "Open Access % (Gold)": "N/A"
    }
    return True, None, {"recommended_journals": [default_journal]}


async def get_recommended_papers(
    db: AsyncSession,
    project_id: str,
    max_attempts: int = MAX_ATTEMPTS_DEFAULT,
    latency: int = LATENCY_DEFAULT,
) -> Tuple[bool, any, any]:
    """관련 논문 레퍼런스 정보를 데이터베이스에서 불러옵니다."""
    attempt_count = 0
    while attempt_count < max_attempts:
        try:
            success, ermsg, recommended_papers = await query.get_recommended_papers(
                db=db, 
                project_id=project_id
            )
            if not success:
                return success, ermsg, None
            
            if recommended_papers is None or not recommended_papers["recommended_papers"]:
                raise ValueError("Recommended papers list is empty or not ready yet")

            return True, None, recommended_papers

        except Exception as e:
            attempt_count += 1
            await asyncio.sleep(latency)

    # 모든 시도가 실패한 후의 처리 >>> 분석 파트에서 어떠한 이유로 에러가 나 DB에 Null 값 있는 경우
    logger.warning("Failed to retrieve recommended papers after maximum attempts. Returning default N/A values.")
    default_paper = {
        "Links": "N/A", "Years": "N/A", "Titles": "N/A", "Authors": "N/A",
        "PDF Links": "N/A", "Citation_Count": "N/A"
    }
    return True, None, {"recommended_papers": [default_paper]}

async def get_analysis_info_(
    db: AsyncSession,
    project_id: str,
    max_attempts: int = MAX_ATTEMPTS_DEFAULT,
    latency: int = LATENCY_DEFAULT,
) -> Tuple[bool, any, any]:
    """관련 논문 레퍼런스 정보를 데이터베이스에서 불러옵니다."""
    attempt_count = 0
    while attempt_count < max_attempts:
        try:
            success, ermsg, analysis_info = await query.get_analysis_info_(
                db=db, 
                project_id=project_id
            )
            if not success:
                return success, ermsg, None
            
            # 문서 정보에서 None 값이 포함된 경우 에러 발생
            if any(value is None for value in analysis_info.values()):
                raise ValueError("Document info contains None value")
            return True, None, analysis_info

        except Exception as e:
            attempt_count += 1
            await asyncio.sleep(latency)

    logger.warning("Failed to retrieve analysis info after maximum attempts. Returning default N/A values.")
    # None 값을 "N/A"로 대체하는 로직
    for key in ["recommended_title", "recommended_keyword", "recommended_potential_topics"]:
        if analysis_info.get(key) is None:
            analysis_info[key] = ["N/A"]

    if analysis_info.get("recommended_summarize") is None:
        analysis_info["recommended_summarize"] = {
            "topic": "N/A", 
            "method": "N/A", 
            "conclusion": "N/A"
        }
    else:
        # recommended_summarize가 None이 아닐 때, 내부 키 값들 한번 더 확인
        for sub_key in ["topic", "method", "conclusion"]:
            if analysis_info["recommended_summarize"].get(sub_key) is None:
                analysis_info["recommended_summarize"][sub_key] = "N/A"

    return True, None, analysis_info
        

async def get_published_info(
    db: AsyncSession,
    project_id: str,
    max_attempts: int = MAX_ATTEMPTS_DEFAULT,
    latency: int = LATENCY_DEFAULT,
) -> Tuple[bool, any, any]:
    """논문의 출판정보를 디비에서 불러옵니다."""
    attempt_count = 0
    while attempt_count < max_attempts:
        try:
            success, ermsg, published_info = await query.get_published_info(
                db=db, 
                project_id=project_id
            )
            if not success:
                return success, ermsg, None
            
            if published_info is None or not published_info["published_info"]:
                raise ValueError("Published info is empty or not ready yet")

            return True, None, published_info

        except Exception as e:
            attempt_count += 1
            await asyncio.sleep(latency)

    # 모든 시도가 실패한 후의 처리 >>> 분석 파트에서 어떠한 이유로 에러가 나 DB에 Null 값 있는 경우
    logger.warning("Failed to retrieve recommended papers after maximum attempts. Returning default N/A values.")
    default_paper = {
        "Links": "N/A", "Years": "N/A", "Titles": "N/A", "Authors": "N/A",
        "PDF Links": "N/A", "Citation_Count": "N/A"
    }
    return True, None, {"recommended_papers": [default_paper]}


async def get_analysis_detail_info(analysis_info: dict, response_language: str) -> Tuple[bool, any, any]:
    recommended_title_lang = await common.check_text_lang(str(analysis_info["recommended_title"]))
    recommended_summarize_lang = await common.check_text_lang(str(analysis_info["recommended_summarize"]))
    recommended_potential_topics_lang = await common.check_text_lang(str(analysis_info["recommended_potential_topics"]))

    if not recommended_title_lang == response_language:
        logger.info(f'제목 추천 번역을 시작합니다.')
        # 제목 추천 번역
        analysis_info["recommended_title"] = await translator.translate_deepl(analysis_info["recommended_title"], response_language)
    
    if not recommended_summarize_lang == response_language:
        logger.info(f'요약 번역을 시작합니다.')
        # 요약 번역
        translated_summarize = await translator.translate_deepl([
            analysis_info['recommended_summarize']['topic'], 
            analysis_info['recommended_summarize']['method'],
            analysis_info['recommended_summarize']['conclusion'],
        ], response_language)
        analysis_info['recommended_summarize']['topic'] = translated_summarize[0]
        analysis_info['recommended_summarize']['method'] = translated_summarize[1]
        analysis_info['recommended_summarize']['conclusion'] = translated_summarize[2]

    if not recommended_potential_topics_lang == response_language:
        logger.info(f'추가 연구주제 추천 번역을 시작합니다.')
        # 추가 연구주제 번역
        analysis_info["recommended_potential_topics"] = await translator.translate_deepl(analysis_info["recommended_potential_topics"], response_language)

    return True, None, analysis_info