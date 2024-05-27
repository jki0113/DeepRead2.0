import os
import openai
import tiktoken
from tqdm import tqdm
import faiss
import requests
from glob import glob
import numpy as np
from ftlangdetect import detect
from serpapi import GoogleSearch
from log.logger_config import logger, log_execution_time, log
from sqlalchemy.orm import Session
import json

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader # pdf에서 구조 정보를 뽑을 수 있지만 살짝 애매함
from langchain.document_loaders import OnlinePDFLoader # 걍 + 느림
from langchain.document_loaders import PyPDFium2Loader # 걍
from langchain.document_loaders import PDFMinerLoader # 걍
from langchain.document_loaders import PDFMinerPDFasHTMLLoader # HTML 형식으로 가져옴
from langchain.document_loaders import PyMuPDFLoader # 제일 빠름
from langchain.document_loaders import PyPDFDirectoryLoader # 애는 디렉토리를 선언해줘야 함 ~.pdf 까지 경로 입력하는 것 아님
from langchain.document_loaders import PDFPlumberLoader # 애는 못쓰겠다
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter

from app.services import analysis
from app.queries import chat as query
from utils.async_gpt import *
from utils import prompt_engineering, paper_analysis, price_logger

# OPENAI Settings
OPENAI_KEY = openai.api_key = os.getenv("OPENAI_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
MAX_TOKEN = 20000
CHUNK_SIZE = 3000
OVERLAP_CHUNK_SIZE = 200
DATABASE = f"embedding_vector_{CHUNK_SIZE}_{OVERLAP_CHUNK_SIZE}"
MAX_CANDIDATE = 3
CHAT_MODEL = "gpt-4-1106-preview"
CHAT_TOKENIZER = tiktoken.get_encoding("cl100k_base")
EMBEDDING_MODEL = "text-embedding-3-large"
HOST = os.getenv("HOST")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
DB = os.getenv("DB")
CHARSET = os.getenv("CHARSET")


async def list_default_questions(response_language:str):
    question_list_ko = [
        '논문에서 사용된 연구 방법론은 무엇인가요?',# 1. 연구 방법론
        '논문의 연구 결론은 무엇이며, 그 의미는 무엇인가요?',# 2. 연구 결론
        '논문의 연구 한계점은 무엇이라고 생각하시나요?',# 3. 연구 한계점
        '논문이 어떤 논문인지 간략하게 설명해 주세요.',# 4. 논문 소개
        '논문의 주제는 무엇인가요?',# 5. 논문 주제
        '논문의 목적은 무엇인가요?',# 6. 논문 목적
        '논문에서 주요 다루고 있는 분야는 어떤 분야인가요?',# 7. 주요 분야
        '논문을 통해서 얻을 수 있는 인사이트는 무엇인가요?',# 8. 인사이트
        '논문을 기반으로 연구를 한다면 어떤 주제를 기반으로 연구할 수 있나요?',# 9. 기반 연구
        '이 논문을 더 발전시키기 위해 어떤 추론이나 가설을 제안할 수 있나요?',# 10. 발전 방향
    ]
    question_list_en = [
        "What research methodology was used in the paper?",
        "What are the conclusions of the research in the paper, and what do they mean?",
        "What do you think are the limitations of the research in the paper?",
        "Can you briefly describe what the paper is about?",
        "What is the topic of the paper?",
        "What is the purpose of the paper?",
        "What are the main fields addressed in the paper?",
        "What insights can be gained from the paper?",
        "If research is to be based on this paper, what topics could be explored?",
        "What inferences or hypotheses can be proposed to further develop the paper?",
    ]

    if response_language.lower() == "korean":
        return question_list_ko
    elif response_language.lower() == "english":
        return question_list_en
    else:
        return question_list_ko
    

def get_full_text(doc_path, parser):
    loaders_mapping = {
        "PyPDFLoader": PyPDFLoader,
        "UnstructuredPDFLoader": UnstructuredPDFLoader,
        "OnlinePDFLoader": OnlinePDFLoader,
        "PyPDFium2Loader": PyPDFium2Loader,
        "PDFMinerLoader": PDFMinerLoader,
        "PDFMinerPDFasHTMLLoader": PDFMinerPDFasHTMLLoader,
        "PyMuPDFLoader": PyMuPDFLoader,
        "PyPDFDirectoryLoader": PyPDFDirectoryLoader,
        "PDFPlumberLoader": PDFPlumberLoader
    }
    loader = loaders_mapping[parser](doc_path)
    pages = loader.load()
    total_text = '\n'.join([text.page_content for text in pages])
    split_total_text = [i.strip() for i in total_text.split('\n')]
    preprocessed_text = '\n'.join(split_total_text)
    return preprocessed_text, pages


def get_recursive_chunk(load_pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_CHUNK_SIZE)
    text = text_splitter.split_documents(load_pages)

    document_info = {}
    for item in text:
        content = item.page_content
        page = item.metadata['page']
        if content in document_info:
            document_info[content]["page"].append(page)
        else:
            document_info[content] = {"page": [page+1]}
    return document_info


async def get_text_embedding(document_info):
    embedding_list = []

    for content, info in tqdm(document_info.items(), desc="get_text_embedding"):
        embedding_list.append(content)

    if embedding_list:
        total_price = .0 # for price logger
        result_list = await process_api_requests_from_prompt_list(
                prompt_list=embedding_list,
                model='text-embedding-3-large',
                api_key=OPENAI_KEY,
                request_url = "https://api.openai.com/v1/embeddings",
                desc="embedding",
                timeout=5
        )
        for result in tqdm(result_list, desc="Parsing Embedding result"):
            if result is None:
                continue
            else:
                try:
                    content = result[1][0]['input']
                    embedding = result[1][1]['data'][0]['embedding']
                    total_price += result[1][1]['usage']['total_tokens'] * 0.00000013 # for price logger
                except:
                    raise Exception
                
                document_info[content]['embedding'] = embedding

    # return document_info
    return document_info, {'total_price' : total_price, 'chunk_length': len(embedding_list)} # for price logger


def save_document_info(storage_path, document_info):
    document_info_json_path = os.path.join(storage_path, 'document_info.json')
    
    with open(document_info_json_path, 'w', encoding='utf-8-sig') as f:
        json.dump(document_info, f, ensure_ascii=False, indent=4)


def openai_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response['data'][0]['embedding']


def chat_classification(input_chat):
    return '챗봇'
    prompt_list = [prompt_engineering.make_prompt_chat_classification(input_chat)]
    result = [None] * len(prompt_list)

    result_list = asyncio.run(
        process_api_requests_from_prompt_list(
            prompt_list = prompt_list,
            model = 'gpt-3.5-turbo',
            temperature = .0,
            api_key = OPENAI_KEY,
            request_url = "https://api.openai.com/v1/chat/completions",
            desc='chat classification',
            timeout=15
        )
    )
    for gpt_result in result_list:
        if gpt_result is None:
            continue
        else:
            result[gpt_result[0]] = gpt_result[1][1]['choices'][0]['message']['content']
    return result[-1]


async def FAISS_recommender(user_input, project_id):
    storage_path = await query.get_project_path_by_project_id(project_id)
    try:
        with open(f"{storage_path}/document_info.json", 'r', encoding='utf-8-sig') as f:
        # with open(f"/home/deepread2.0/storage/kyungill/PRJ_20231212_kyungill_001/document_info.json", 'r', encoding='utf-8-sig') as f:
            documents = json.load(f)
    except FileNotFoundError:
        storage_path_url = storage_path.replace(os.getenv('STORAGE_PATH_SYSTEM'), os.getenv('STORAGE_PATH_URL')) + '/document_info.json'
        response = requests.get(storage_path_url)
        if response.status_code == 200:
            decoded_content = response.content.decode('utf-8-sig')
            documents = json.loads(decoded_content)
        else:
            raise Exception("원격 파일을 가져오는 데 실패했습니다.")
    question_embedding = openai_embedding(user_input)
    dimension = len(question_embedding)
    index = faiss.IndexFlatL2(dimension)
    
    embeddings = []
    mapping = {}
    for chunk, info in documents.items():
        page, embedding = info.get('page', -1), info.get('embedding', [])
        embeddings.append(embedding)
        mapping[len(embeddings) - 1] = {'chunk': chunk, 'page': page}
    
    embeddings = np.array(embeddings)
    index.add(embeddings)
    
    distances, indices = index.search(np.array([question_embedding]), 30)
        
    results = []

    for idx in [idx for idx in indices[0] if idx != -1]:
        document_info = mapping[idx] 
        pages_str = ", ".join(map(str, document_info['page']))

        lines = document_info['chunk'].split('\n')
        short_lines_count = sum(1 for line in lines if len(line) < 20)

        if short_lines_count < 15:
            results.append({'chunk': document_info['chunk']})
        if len(results) == 3:
            break
    return results


def paper_basic_info_recommender(paper_basic_info_path, question_type):
    # with open(input.storage_path + "paper_basic_info.json", 'r') as f:
    with open(paper_basic_info_path, 'r', encoding='utf-8-sig') as f:
        paper_basic_info = json.load(f)

    info_strings = []
    if '기본정보' in question_type:
        chosen_key = ['title', 'author', 'abstract']
        for key in chosen_key:
            if key in paper_basic_info:
                info_strings.append(f"{key}: {paper_basic_info[key]}")
    
    elif '요약' in question_type:
        chosen_key = ['recommended_summarize']
        for key in chosen_key:
            if key in paper_basic_info:
                info_strings.append(f"{key}: {paper_basic_info[key]}")
    else:
        pass

    return "\n".join(info_strings)


# 최소한 입력한 채팅은 들어갈 수 있도록 수정 필요
def chat_tokens_manager(messages_for_history):
    max_token = MAX_TOKEN
    num_tokens = 0
    truncated_chat = []
    for i in reversed(messages_for_history):
        len_token = len(CHAT_TOKENIZER.encode(str(i['content'])))
        num_tokens += len_token
        if num_tokens > max_token:
            break
        truncated_chat.append(i)
    return list(reversed(truncated_chat))


def preprocess_text(full_text, max_char):
    full_text_split = full_text.split('\n')
    preprocessed_list = [
        line for line in full_text_split 
        if len(line.strip()) > 1 
        and '···' not in line 
        and '...' not in line 
        and '…' not in line
    ]

    preprocessed_list = [re.sub(' +', ' ', line) for line in preprocessed_list]
    detect_lang = detect(text=full_text.replace('\n', ' '), low_memory=False)['lang']
    preprocessed_text = ' '.join(preprocessed_list)
    # "abstract" 단어를 대소문자 구분 없이 찾기
    # abstract_match = re.search(r'abstract', preprocessed_text, re.IGNORECASE)
    # if abstract_match:
    #     # "abstract" 단어가 있는 위치부터 텍스트 업데이트
    #     start_index = abstract_match.start()
    #     preprocessed_text = preprocessed_text[start_index:]
    if detect_lang == 'en':
        return preprocessed_text[:max_char]
    else:
        return preprocessed_text[:max_char+1000]


async def get_chatbot_function_summary_prompt(user_input:str, response_language:str) -> str:
    user_input_with_function = prompt_engineering.content_summary(user_input, response_language)
    processed_user_input = user_input.replace('\n', '')
    user_input_for_db = f"""드래그한 부분을 요약해줘:\n\n{processed_user_input}"""
    return user_input_with_function, user_input_for_db


async def get_chatbot_function_translate_prompt(user_input:str, response_language:str) -> str:
    user_input_with_function = prompt_engineering.translate(user_input, response_language)
    processed_user_input = user_input.replace('\n', '')
    user_input_for_db = f"""드래그한 부분을 번역해줘:\n\n{processed_user_input}"""
    return user_input_with_function, user_input_for_db


async def process_chat_stream(response_stream, websocket, function, output_chat_index):
    chat_stream_result = ''
    async for chunk in response_stream:
        chunk_str = chunk.decode("utf-8")
        if chunk_str.strip():
            chunk_str = chunk_str.replace("data: ", "").strip()
            if chunk_str == "[DONE]":
                break
            else:
                chunk_data = json.loads(chunk_str)
                if ("delta" in chunk_data["choices"][0]
                    and "content" in chunk_data["choices"][0]["delta"]):
                    chat_stream = chunk_data["choices"][0]["delta"]["content"]
                    chat_stream_result += chat_stream
                    response = {
                        'chat_index': output_chat_index,
                        'content': chat_stream,
                        'function': function,
                    }
                    await websocket.send_json(response)
    response = {
        'chat_index': output_chat_index,
        'content': '<eos>',
        'function': function,
    }
    await websocket.send_json(response)
    return chat_stream_result


async def get_search_query(user_input):
    prompt_list = [prompt_engineering.make_prompt_chat_web_search_query(user_input, '_')]
    result = [None] * len(prompt_list)
    result_list = await process_api_requests_from_prompt_list(
            prompt_list=prompt_list,
            model='gpt-4-1106-preview',
            temperature=.0,
            api_key=OPENAI_KEY,
            request_url="https://api.openai.com/v1/chat/completions",
            json_mode=True,
            desc="extract web search query",
            timeout=30
    )

    for gpt_result in result_list:
        if gpt_result is not None:
            result[gpt_result[0]] = gpt_result[1][1]['choices'][0]['message']['content']
    result_dict = prompt_engineering.parse_json_string(result[0], ['result'])
    return result_dict


async def search_from_serp_api(query):
    params = {
        'engine' : 'google',
        'q' : query,
        'api_key' : SERPAPI_KEY
    }
    search = GoogleSearch(params).get_dict()
    saerch_result = [
        {
            'title': entry['title'],
            'link': entry['link'],
            'favicon': entry.get('favicon', None),  # 값 없는 경우 None
            'snippet': entry['snippet'], 
            'source' : entry.get('source', None)
        }
        for entry in search['organic_results']
    ]
    return saerch_result


async def extract_term(chat_input):
    prompt_list = [prompt_engineering.make_prompt_chat_explain_term(chat_input, '_')]
    async_gpt_result = await process_api_requests_from_prompt_list(
            prompt_list=prompt_list,
            model = "gpt-4-1106-preview",
            temperature = .0,
            api_key = OPENAI_KEY,
            request_url = "https://api.openai.com/v1/chat/completions",
            json_mode = True,
            desc = "search query",
            timeout=40
    )
    result = [None] * len(prompt_list)
    price_list = [None] * len(prompt_list)
    for gpt_result in async_gpt_result:
        if gpt_result is None:
            continue
        else:
            result[gpt_result[0]] = gpt_result[1][1]['choices'][0]['message']['content']
            price_list[gpt_result[0]] = (gpt_result[1][1]['usage']['prompt_tokens'] * 0.01 / 1000) + (gpt_result[1][1]['usage']['completion_tokens'] * 0.03 / 1000)
    result_dict = prompt_engineering.parse_json_string(result[0], ['result'])
    # return result_dict
    return result_dict, sum(price_list) # for price logger


async def process_explain_term(query_list, project_id, response_language):
    information_list = [await FAISS_recommender(term, project_id) for term in query_list]
    prompt_list = [prompt_engineering.make_prompt_for_term_search(term, ' ', response_language) for term in zip(query_list, information_list)]
    async_gpt_result = await process_api_requests_from_prompt_list(
            prompt_list=prompt_list,
            model = "gpt-4-1106-preview",
            temperature = .0,
            api_key = OPENAI_KEY,
            request_url = "https://api.openai.com/v1/chat/completions",
            json_mode = True,
            desc = "search query",
            timeout=60  
    )
    result = [None] * len(prompt_list)
    for gpt_result in async_gpt_result:
        if gpt_result is None:
            continue
        else:
            result[gpt_result[0]] = gpt_result[1][1]['choices'][0]['message']['content']
    result_dict = [prompt_engineering.parse_json_string(term, ['term', 'explanation', 'method']) for term in result]
    
    # origin_term을 보장하기 위해 term의 키 값을 query_list로 대체 
    for origin_term, result in zip(query_list, result_dict):
        result['term'] = origin_term
        result['origin_term'] = origin_term

    return result_dict


async def get_document_metadata_info(storage_path: str):
    # pdf 경로를 입력 받아 전체 텍스트 추출
    pdf_path = glob(f"{storage_path}/*.pdf")[0]
    full_text, pages = get_full_text(pdf_path, 'PyMuPDFLoader')
    
    # 청크 분할
    document_info = get_recursive_chunk(pages)
    
    # 임베딩을 유저의 storage_path에 문서로 저장
    # document_info = await get_text_embedding(document_info)
    document_info, price_info = await get_text_embedding(document_info) # for price logger
    save_document_info(storage_path, document_info)

    # 240516 price logger 중지
    # price_logger.add_embed_log(
    #     mode = os.getenv("MODE"),
    #     project_id = os.path.basename(storage_path),
    #     file_name = os.path.basename(pdf_path),
    #     character= len(full_text),
    #     chunks = price_info['chunk_length'],
    #     embed_price = price_info['total_price']
    # )