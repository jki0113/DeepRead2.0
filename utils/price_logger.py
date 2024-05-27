import pandas as pd
from datetime import datetime
import os
import portalocker

def add_embed_log(mode, project_id, file_name, character, chunks, embed_price):
    file_path = '/home/price_log/embedding_log.xlsx'
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    data = {
        'mode': mode,
        'project_id': project_id,
        'file_name': file_name,
        'character': character,
        'chunks': chunks,
        'embed_price': embed_price,
        'created_at': current_time
    }
    
    new_row = pd.DataFrame([data])
    with portalocker.Lock(file_path, 'a+', timeout=5) as locked_file:
        try:
            df = pd.read_excel(file_path)
        except FileNotFoundError:
            df = pd.DataFrame()
        except ValueError:
            df = pd.DataFrame()  # 파일이 비어있으면 빈 DataFrame 생성

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(file_path, index=False)


def add_analysis_log(mode, project_id, file_name, character, paper_basic_info_price,
                     recommended_title_price, recommended_keyword_price, recommended_summarize_price,
                     recommended_potential_topics_price, recommended_journal_price):
    file_path = '/home/price_log/analysis_log.xlsx'
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {
        'mode': mode,
        'project_id': project_id,
        'file_name': file_name,
        'character': character,
        'paper_basic_info_price': paper_basic_info_price,
        'recommended_title_price': recommended_title_price,
        'recommended_keyword_price': recommended_keyword_price,
        'recommended_summarize_price': recommended_summarize_price,
        'recommended_potential_topics_price': recommended_potential_topics_price,
        'recommended_journal_price': recommended_journal_price,
        'created_at': current_time
    }
    new_row = pd.DataFrame([data])
    with portalocker.Lock(file_path, 'a+', timeout=5) as locked_file:
        try:
            df = pd.read_excel(file_path)
        except FileNotFoundError:
            df = pd.DataFrame()  # 파일이 없으면 빈 데이터프레임 생성
        except ValueError:
            df = pd.DataFrame()  # 파일이 비어있으면 빈 DataFrame 생성
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(file_path, index=False)


def add_recommended_questions_log(mode, recommended_questions_price):
    file_path = '/home/price_log/recommended_questions_log.xlsx'
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    data = {
        'mode': mode,
        'recommended_questions_price': recommended_questions_price,
        'created_at': current_time
    }
    
    new_row = pd.DataFrame([data])
    with portalocker.Lock(file_path, 'a+', timeout=5) as locked_file:
        try:
            df = pd.read_excel(file_path)
        except FileNotFoundError:
            df = pd.DataFrame()
        except ValueError:
            df = pd.DataFrame()  # 파일이 비어있으면 빈 DataFrame 생성

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(file_path, index=False)

def add_chat_log(mode, function, response_language, chat_price):
    file_path = '/home/price_log/chat_log.xlsx'
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    data = {
        'mode' : mode,
        'function' : function,
        'response_language': response_language,
        'chat_price': chat_price,
        'created_at': current_time,
    }

    new_row = pd.DataFrame([data])
    with portalocker.Lock(file_path, 'a+', timeout=5) as locked_file:
        try:
            df = pd.read_excel(file_path)
        except FileNotFoundError:
            df = pd.DataFrame()
        except ValueError:
            df = pd.DataFrame()

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(file_path, index=False)