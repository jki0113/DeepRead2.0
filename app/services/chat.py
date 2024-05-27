import os, sys
import json
import openai
from glob import glob
from fastapi import HTTPException, WebSocket
from typing import Tuple, Optional

from log.logger_config import logger, log_execution_time, log
from sqlalchemy.orm import Session

from app.queries import chat as query
from app.schemas import chat as schemas
from utils.async_gpt import *
from utils import prompt_engineering, chatbot, price_logger

OPENAI_KEY = openai.api_key = os.getenv("OPENAI_KEY")


async def chatbot_init(db: Session, project_id: str) -> Tuple[bool, any, any]:
    success, ermsg, chatbot_init_info = await query.chat_init(db, project_id=project_id)
    if not success:
        return False, ermsg, None

    pdf_file_path = glob(os.path.join(chatbot_init_info['project_path'], '*.pdf'))[0]
    chatbot_init_info['file_name'] = os.path.basename(pdf_file_path)
    preprocessed_text, pages = chatbot.get_full_text(pdf_file_path, 'PyMuPDFLoader')
    chatbot_init_info['pdf_page'] = len(pages)
    chatbot_init_info['project_path'] = pdf_file_path.replace(os.getenv('STORAGE_PATH_SYSTEM'), os.getenv('STORAGE_PATH_URL'))

    return True, None, chatbot_init_info


async def get_chatbot_init(
    db: Session,
    project_id: str,
    response_language: Optional[str]
) -> Tuple[bool, any, any]:
    success, ermsg, run_chatbot_init = await chatbot_init(db, project_id=project_id)
    if not success:
        return False, ermsg, None

    success, ermsg, chat_log_list = await query.get_project_chats(db, project_id=project_id)
    if not success:
        return False, ermsg, None

    default_questions = await chatbot.list_default_questions(response_language)

    success, ermsg, project_analysis_survey = await query.get_project_analysis_survey(db, project_id=[project_id])
    if not success:
        return False, ermsg, None
    
    return True, None, schemas.response_get_chatbot_init(
                            project_id=run_chatbot_init['project_id'],
                            folder_name=run_chatbot_init['folder_name'],
                            project_name=run_chatbot_init['project_name'],
                            project_file_path=run_chatbot_init['project_path'],
                            pdf_pages=run_chatbot_init['pdf_page'],
                            file_name=run_chatbot_init['file_name'],
                            created_at=run_chatbot_init['created_at'],
                            chat_log = chat_log_list,
                            project_analysis_survey=project_analysis_survey,
                            default_questions = default_questions,
                        )


async def generate_chat_prompt(
    db: Session,
    project_id: str,
    user_input:str,
    candidate_list: str,
    response_language: str
) -> Tuple[bool, any, any]:
    # 채팅 로그를 가져옵니다.
    success, ermsg, chat_log = await query.get_project_chats(db=db, project_id=project_id)
    if not success:
        return False, ermsg, None

    # role 과 content만 기록에서 발라냄 
    # 일단 펑션으로 실행한 기능들도 딕셔너리 형태이지만 다 str로 바꿔줘야함
    chat_log = [{'role': item['role'], 'content': str(item['content'])} for item in chat_log]
    
    chat_log.pop() # 유저 질문은 제거
    # 새로운 사용자 입력을 채팅 로그에 추가
    chat_subset = {"role": 'user', "content": prompt_engineering.make_prompt_chatbot(user_input, candidate_list, response_language)} 
    chat_log.append(chat_subset)

    # 채팅 토큰을 관리하고 GPT 대화 프롬프트를 만듭니다.
    truncated_chat = chatbot.chat_tokens_manager(chat_log)
    return True, None, truncated_chat


async def chat(db: Session, websocket: WebSocket, data: dict, project_id: str) -> Tuple[bool, any, any]:
    logger.critical('total_price 초기화 -> .0') # for price logger
    total_price = .0 # for price logger
 
    user_input = data["user_input"]
    response_language = data["response_language"] 
    function = data['function']
    success, ermsg, input_chat_index, output_chat_index = await query.create_new_chat_index(db, project_id=project_id)
    if not success:
        return False, ermsg, None
    
    if function=='chat':
        chat_classification = chatbot.chat_classification(user_input)
        if '챗봇' in chat_classification:
            candidate_list = await chatbot.FAISS_recommender(user_input, project_id)
        elif '기본정보' in chat_classification:
            candidate_list = chatbot.paper_basic_info_recommender(input.storage_path + "paper_basic_info.json", '기본정보')

        await update_user_chat(
            db,
            chat_index=input_chat_index,
            project_id=project_id, 
            content=user_input, 
            response_language=response_language,
            role='user',
            function=function
        )

        success, ermsg, messages = await generate_chat_prompt(
                db=db,
                project_id=project_id,
                candidate_list=candidate_list,
                response_language=response_language,
                user_input=user_input,
            )
        if not success:
            return False, ermsg, None
            
        total_price += sum([num_tokens_from_string(text=i['content']) for i in messages]) * 0.00001 # for price logger
        
        res = await asyncio_gpt_for_chat(request_json_list=[
            {
                "model": 'gpt-4-1106-preview',
                "messages": messages,
                "temperature": 0.3,
                "stream": True,
            }
        ])
        request, response, session = res["results"][0]
        chat_response = await chatbot.process_chat_stream(response_stream=response, websocket=websocket, function=function, output_chat_index=output_chat_index)

        total_price += num_tokens_from_string(text=chat_response) * 0.00003 # for price logger
        await close_session_gpt_for_chat(session)

    elif function == 'content_summary':
        user_input_with_function, user_input_for_db = await chatbot.get_chatbot_function_summary_prompt(user_input=user_input, response_language=response_language)
        total_price += num_tokens_from_string(text=user_input_with_function) * 0.00001 # for price logger
        user_input_with_function = [{'role': 'user', 'content': f'{user_input_with_function}'}]

        await update_user_chat(
            db,
            chat_index=input_chat_index,
            project_id=project_id, 
            content=user_input_for_db, 
            response_language=response_language,
            role='user',
            function=function,
        )

        res = await asyncio_gpt_for_chat(request_json_list=[
            {
                "model": 'gpt-4-1106-preview',
                "messages": user_input_with_function,
                "temperature": .0,
                "stream": True,
            }
        ])
        request, response, session = res["results"][0]
        chat_response = await chatbot.process_chat_stream(response_stream=response, websocket=websocket, function=function, output_chat_index=output_chat_index)
        total_price += num_tokens_from_string(text=chat_response) * 0.00003 # for price logger
        await close_session_gpt_for_chat(session)

    elif function == 'explain_term':
        logger.debug(f"chatbot function > user_input > {user_input}")
        user_input_with_function = "드래그한 부분에 대한 용어를 검색해줘:\n\n" + user_input.replace('\n', '')
        await update_user_chat(
            db,
            chat_index=input_chat_index,
            project_id=project_id, 
            content=user_input_with_function, 
            response_language=response_language,
            role='user',
            function=function,
        )

        # query_list = await chatbot.extract_term(user_input) # for price logger
        query_list, query_list_price = await chatbot.extract_term(user_input) # for price logger
        total_price += query_list_price # for price logger

        logger.debug(f"chatbot function > query_list > {query_list}")

        chat_response = await chatbot.process_explain_term(query_list['result'], project_id, response_language)
        total_price += num_tokens_from_string(text=str(chat_response)) * 0.00003 # for price logger

        logger.debug(f"chatbot function > explain_term result > {json.dumps(chat_response, indent=4, ensure_ascii=False)}")
        await websocket.send_json(
            {
                "chat_index": output_chat_index,
                "content": chat_response,
                "function": function,
            }
        )

    elif function == 'web_search':
        total_price += 1/50 # for price logger (웹 검색은 월 api 고정 지출 비용금액의 1회 분을 산정)
        user_input_with_function = "드래그한 부분에 대해 웹 검색해줘:\n\n" + user_input.replace('\n', '')
        
        await update_user_chat(
            db,
            chat_index=input_chat_index,
            project_id=project_id, 
            content=user_input_with_function, 
            response_language=response_language,
            role='user',
            function=function,
        )    

        if len(user_input.split()) <= 5:
            search_query = user_input
        else:
            search_query = await chatbot.get_search_query(user_input)
            search_query = search_query['result']
        chat_response = await chatbot.search_from_serp_api(search_query)
        await websocket.send_json(
            {
                "chat_index": output_chat_index,
                "content": chat_response,
                "function": function,
            }
        )

    elif function == 'translate':
        """출발어와 도착어가 동일한 경우 결과를 그대로 반환하는 로직이 필요합니다."""
        user_input_with_function, user_input_for_db = await chatbot.get_chatbot_function_translate_prompt(user_input=user_input, response_language=response_language)
        total_price += num_tokens_from_string(text=user_input_with_function) * 0.00001 # for price logger
        user_input_with_function = [{'role': 'user', 'content': f'{user_input_with_function}'}]
        
        await update_user_chat(
            db,
            chat_index=input_chat_index,
            project_id=project_id, 
            content=user_input, 
            response_language=response_language,
            role='user',
            function=function,
        )
        request_json = [
            {
                "model": 'gpt-4-1106-preview',
                "messages": user_input_with_function,
                "temperature": .0,
                "stream": True,
            }
        ]
        res = await asyncio_gpt_for_chat(request_json_list=request_json)
        request, response, session = res["results"][0]
        chat_response = await chatbot.process_chat_stream(response_stream=response, websocket=websocket, function=function, output_chat_index=output_chat_index)
        total_price += num_tokens_from_string(text=chat_response) * 0.00003 # for price logger
        await close_session_gpt_for_chat(session)
        
    await update_user_chat(
        db,
        chat_index=output_chat_index,
        project_id=project_id, 
        content=chat_response, 
        response_language=response_language,
        role='assistant',
        function=function,
    )
    # 채팅이 다 끝나면 project의 timestep을 업데이트 해줘야 함
    await query.update_project_recent_timestep(
        db,
        project_id=project_id
    )

    # 240516 price logger 중지
    # price_logger.add_chat_log(
    #     mode=os.getenv("MODE"),
    #     function=function,
    #     response_language=response_language,
    #     chat_price=total_price,
    # )
    return True, None, True


async def update_user_chat(
    db: Session,
    chat_index: str,
    project_id: str,
    content: str,
    response_language: str,
    role:str,
    function:str
):
    success, ermsg, new_chat = await query.update_user_chat(
                                        db,
                                        chat_index=chat_index,
                                        project_id=project_id,
                                        content=content,
                                        role=role,
                                        response_language=response_language,
                                        function=function,
                                    )
    if not success:
        raise HTTPException(status_code=500, detail="채팅 업데이트에 실패했습니다.")
    
    return new_chat


async def get_recommended_questions(user_chat:str, assistant_chat: str, function: str, response_language: str) -> Tuple[bool, any, any]:
    if function == 'explain_term':
        preprocessed_assistant_chat = 'heres the web search result'
        for idx, term in enumerate(assistant_chat):
            if idx == 3:
                break
            preprocessed_assistant_chat += '\n\n'
            preprocessed_assistant_chat += f"\nterm: {term.get('term', 'Not Found')}"
            preprocessed_assistant_chat += f"\nmethod: {term.get('method', 'Not Found')}"
            assistant_chat = preprocessed_assistant_chat

    elif function == 'web_search':
        preprocessed_assistant_chat = 'heres the web search result'
        for i in assistant_chat:
            preprocessed_assistant_chat += '\n\n'
            preprocessed_assistant_chat += f"\ntitle: {i.get('title', 'Not Found')}"
            preprocessed_assistant_chat += f"\nsource: {i.get('source', 'Not Found')}"
            preprocessed_assistant_chat += f"\nsnippet: {i.get('snippet', 'Not Found')}"
            assistant_chat = preprocessed_assistant_chat
            
    prompt_list = [prompt_engineering.make_prompt_recommended_questions(user_chat, assistant_chat, response_language)]
    result = [None] * len(prompt_list)
    result_list = await process_api_requests_from_prompt_list(
            prompt_list=prompt_list,
            model='gpt-3.5-turbo-1106',
            temperature=.3,
            api_key=OPENAI_KEY,
            request_url="https://api.openai.com/v1/chat/completions",
            json_mode=True,
            desc="recommended questions",
            timeout=30
    )
    price = .0 # for price logger
    for gpt_result in result_list:
        if gpt_result is not None:
            result[gpt_result[0]] = gpt_result[1][1]['choices'][0]['message']['content']
            price += (gpt_result[1][1]['usage']['prompt_tokens'] * 0.01 / 1000) + (gpt_result[1][1]['usage']['completion_tokens'] * 0.03 / 1000) # for price logger
    result_dict = prompt_engineering.parse_json_string(result[0], ['result'])
    reuslt = result_dict['result']

    # price_logger.add_recommended_questions_log(os.getenv("MODE"), price) # 240516 price logger 중지
    # if result == ['unk']:
    #     result = await chatbot.list_default_questions(response_language)
    
    return True, None, reuslt


async def get_project_with_bookmark(db: Session, user_id: str) -> Tuple[bool, any, any]:
    success, ermsg, bookmark_project = await query.get_project_with_bookmark(db, user_id)
    if not success:
        return False, ermsg, None

    for item in bookmark_project:
        pdf_file_path = glob(os.path.join(item['project_path'], '*.pdf'))[0]
        file_name = os.path.basename(pdf_file_path)
        item['file_name'] = os.path.basename(pdf_file_path)
        item['file_path'] = pdf_file_path.replace(os.getenv('STORAGE_PATH_SYSTEM'), os.getenv('STORAGE_PATH_URL'))
        del item['project_path']
        del item['file_path']

    return True, None, bookmark_project

