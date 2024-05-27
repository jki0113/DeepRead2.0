import os, sys
import asyncio, nest_asyncio
import time
import torch
import faiss
import openai
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import regex as re
from semanticscholar import SemanticScholar
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import concurrent.futures

from openai.embeddings_utils import cosine_similarity
from log.logger_config import logger, log_execution_time, log

from utils.async_gpt import *
from utils import prompt_engineering
from utils import chatbot


# journal_recommender - Put weights on values to be computed
primary_weight = 0.5
secondary_weight = 0.3
tertiary_weight = 0.2
weights_dic = {
    'cosine_similarity': primary_weight,
    'IF 2022 - compute':secondary_weight, 
    '2022 Cite Score - compute':secondary_weight,
    'H index - compute':secondary_weight, 
    'SJR - compute': secondary_weight,
    'Issues Per Year - compute': secondary_weight,
    'Citations - compute': secondary_weight,
    'JCI - compute': secondary_weight,
    'Frequency Preference':tertiary_weight,
    'percentageOAGold - compute': tertiary_weight,
}


def openai_embedding(text):
    response = openai.Embedding.create(
        input = text,
        model = "text-embedding-3-large"
    )
    return response['data'][0]['embedding']


def cal_score(scores):
    return -sum([s * w for s, w in zip(scores, weights_dic.values())])


async def sts_journal_recommender(input_paper, journal_df, journal_emb):
    paper_text = f'{input_paper}'
    embed_query = np.array(openai_embedding(paper_text))
    
    similarities = [cosine_similarity(i, embed_query) for i in journal_emb]
    similarities = torch.FloatTensor(similarities)
    top_values, top_indices = torch.topk(similarities, k=5)
    
    journal_df['cosine_similarity'] = similarities
    
    scores_df = journal_df.loc[top_indices, [column_name for column_name in weights_dic.keys()]]
    scores_df['Similarity'] = top_values

    # standardize scale of score values
    top_scores = StandardScaler().fit_transform(scores_df.values)
    top_scores = [cal_score(s) for s in top_scores]

    top_n = [n['idx'] for n in sorted([{'idx': idx, 'score': scr} for idx, scr in zip(top_indices, top_scores)], key=lambda x: x['score'])]
    journal_df.loc[top_n]
    
    data = []

    for idx, sim, score in zip(top_indices, top_values, top_scores):

        journal = journal_df.iloc[idx.item()]
        _title = journal['title']
        _link = journal['link']
        _hindex = journal['H index']
        _if2022 = journal['IF 2022']
        _2022citescore = journal['2022 Cite Score']
        _freqpr = journal['Frequency Preference']
        _sjr = journal['SJR']
        _issuesperyr = journal['Issues Per Year']
        _citations = journal['Citations']
        _jci = journal['JCI']
        _openaccess = journal['percentageOAGold']
        _journalindex = journal['Journal Index']
        _cossim = sim.item()
        
        # Append original values from original columns to display if data is None / Not Available
        data.append({'title': journal['title'],
                    'link': journal['link'],
                    'Topic Relevance': _cossim, #cosine similarity
                    'Impact Factor': journal['IF 2022'], #IF 2022
                    'Cite Score (2022)': journal['2022 Cite Score'],
                    'H index': journal['H index'],
                    'SJR': journal['SJR'],
                    'Issues Per Year': journal['Issues Per Year'],
                    'Citations Count': journal['Citations'],
                    'JCI': journal['JCI'],
                    'Frequency Preference': journal['Frequency'],
                    'Open Access % (Gold)': journal['percentageOAGold'],
                    'Journal Index': journal['Journal Index'],
                    'score': score
        })

    return data


async def sts_journal_recommender_v2(input_paper, journal_df, journal_emb):
    """
    기존 Ralph가 작성했던 코드에서 async_gpt로 임베딩을 변경, faiss로 유사도 검색 방법 변경하였습니다.
    openai_embedding 함수는 더이상 사용하지 않습니다.
    """
    total_price = .0 # for price logger
    request_embedding = await process_api_requests_from_prompt_list(
            prompt_list=[input_paper],
            # model='text-embedding-3-large',
            model='text-embedding-ada-002',
            api_key=OPENAI_KEY,
            request_url = "https://api.openai.com/v1/embeddings",
            desc="embedding",
            timeout=5
    )
    for result in tqdm(request_embedding, desc="Parsing Embedding result"):
        if result is None:
            continue
        else:
            try:
                content = result[1][0]['input']
                embedding = np.array(result[1][1]['data'][0]['embedding'])
                total_price += result[1][1]['usage']['total_tokens'] * 0.00000010
            except Exception as E:
                raise E

    # FAISS를 사용한 유사도 검색
    dimension = embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(journal_emb).astype('float32'))
    
    # 유사한 임베딩 5개 검색
    distances, indices = index.search(embedding.reshape(1, -1).astype('float32'), 5)
    
    # 유사도 계산 (옵션, 거리를 유사도로 변환하는 방법에 따라 다름)
    top_values = 1 / (1 + distances[0])  # 예시: 거리를 유사도로 변환
    top_indices = indices[0]

    # 선택된 인덱스에 대해 유사도 값을 데이터프레임에 추가
    for i, idx in enumerate(top_indices):
        journal_df.at[idx, 'cosine_similarity'] = top_values[i]

    # 필요한 점수 데이터프레임 생성
    scores_df = journal_df.loc[top_indices, [column_name for column_name in weights_dic.keys()]]
    scores_df['Similarity'] = top_values
    
    # standardize scale of score values
    top_scores = StandardScaler().fit_transform(scores_df.values)
    top_scores = [cal_score(s) for s in top_scores]

    top_n = [n['idx'] for n in sorted([{'idx': idx, 'score': scr} for idx, scr in zip(top_indices, top_scores)], key=lambda x: x['score'])]
    journal_df.loc[top_n]
    
    data = []

    for idx, sim, score in zip(top_indices, top_values, top_scores):

        journal = journal_df.iloc[idx.item()]
        _title = journal['title']
        _link = journal['link']
        _hindex = journal['H index']
        _if2022 = journal['IF 2022']
        _2022citescore = journal['2022 Cite Score']
        _freqpr = journal['Frequency Preference']
        _sjr = journal['SJR']
        _issuesperyr = journal['Issues Per Year']
        _citations = journal['Citations']
        _jci = journal['JCI']
        _openaccess = journal['percentageOAGold']
        _journalindex = journal['Journal Index']
        _cossim = sim.item()
        
        # Append original values from original columns to display if data is None / Not Available
        data.append({'title': journal['title'],
                    'link': journal.get('link_journal', journal['link']),
                    'Topic Relevance': _cossim, #cosine similarity
                    'Impact Factor': journal['IF 2022'], #IF 2022
                    'Cite Score (2022)': journal['2022 Cite Score'],
                    'H index': journal['H index'],
                    'SJR': journal['SJR'],
                    'Issues Per Year': journal['Issues Per Year'],
                    'Citations Count': journal['Citations'],
                    'JCI': journal['JCI'],
                    'Frequency Preference': journal['Frequency'],
                    'Open Access % (Gold)': journal['percentageOAGold'],
                    'Journal Index': journal['Journal Index'],
                    'score': score
        })

    # return data
    return data, total_price # for price logger


async def likert_scale(df, column_name):
    # Define the Likert scale ranges and corresponding values
    likert_ranges = [(0.01, 0.1999), (0.20, 0.3999), (0.40, 0.5999), (0.60, 0.7999), (0.80, 1.0)]
    likert_values = [1, 2, 3, 4, 5]

    # Create a new column to store the Likert scale values
    likert_column = []

    # Iterate through the specified column and assign Likert values
    for value in df[column_name]:
        for i, (start, end) in enumerate(likert_ranges):
            if start <= value <= end:
                likert_column.append(likert_values[i])
                break
    # else:
    #     likert_column.append(None)  # Handle values outside the specified ranges

    df[column_name] = likert_column

    return df


async def journal_recommender(input_paper):
    """
    기존의 Ralph코드에서 매번 저널 정보를 전처리 하는 과정의 시간이 너무 많이 발생하여 변경했습니다.
    """
    # journal_df = pd.read_parquet('/home/deepread2.0/assets/journal_data/journal_embedding_openai_v4.parquet')
    # journal_df['new_embedding'] = journal_df['new_embedding'].apply(lambda x: json.loads(x))
    # journal_emb = journal_df['new_embedding'].to_list()

    # #Prepare Journal Index
    # journal_df['JI Category'] = journal_df['Category & Journal Quartiles'].apply(lambda x: x.split('; ') if ';' in str(x) else str(x))
    # journal_df['JI Category'] = journal_df['JI Category'].apply(lambda x: ',\n'.join(x) if isinstance(x, list) else x)
    # journal_df['Journal Index'] = journal_df['Core Collection'] + ':\n' + journal_df['JI Category']
    # journal_df['Journal Index'] = journal_df['Journal Index'].apply(lambda x: x.replace(':\n0', '') if '0' in x else x)

    # # Create copy of series data to be filled with 0 to avoid displaying 0 instead of 'Nan'
    # journal_df['IF 2022 - compute'] = journal_df['IF 2022'].copy().fillna(0)
    # journal_df['2022 Cite Score - compute'] = journal_df['2022 Cite Score'].copy().fillna(0)
    # journal_df['H index - compute'] = journal_df['H index'].copy().fillna(0)
    # journal_df['SJR - compute'] = journal_df['SJR'].copy().fillna(0)
    # journal_df['Issues Per Year - compute'] = journal_df['Issues Per Year'].copy().fillna(0)
    # journal_df['Citations - compute'] = journal_df['Citations'].copy().fillna(0)
    # journal_df['JCI - compute'] = journal_df['JCI'].copy().fillna(0)
    # journal_df['percentageOAGold - compute'] = journal_df['percentageOAGold'].copy().fillna(0)

    # # Assign corresponding weight of frequency preference
    # freq_prefer_dict = {1 : ['Monthly', 'Bi-monthly', 'Semi-monthly'], 
    #                     0.8 : ['Weekly', 'Fortnightly', 'Quarterly', 'Continuous publication', 'Article-by-article'],
    #                     'dislike' : ['Annual', 'Semi-annual', 'Tri-annual', 'Irregular'],
    #                     }
    # freq_prefer = ['dislike'] * journal_df.shape[0]


    # for i in tqdm(range(journal_df.shape[0])):
    #     row_freq = journal_df.iloc[i]['Frequency']

    #     for freq in freq_prefer_dict.keys():
    #         if row_freq in freq_prefer_dict[freq]:
    #             freq_prefer[i] = freq
    #             break

    # journal_df['Frequency Preference'] = freq_prefer
    # journal_df = pd.DataFrame(journal_df)
    tmp_time = time.time()
    journal_df = pd.read_parquet('/home/deepread2.0_dev/assets/journal_data/journal_embedding_openai_v5_preprocessed.parquet')

    try:
        if not isinstance(input_paper, str):
            raise ValueError("Input is either empty or not in string format.")

        # reco_result = await sts_journal_recommender_v2(input_paper, journal_df, journal_df['new_embedding'].to_list())
        reco_result, total_price = await sts_journal_recommender_v2(input_paper, journal_df, journal_df['new_embedding'].to_list()) # for price logger
        x = pd.DataFrame(sorted(reco_result, key=lambda x: x['score']))

        # Likert Scale
        await likert_scale(x,'Topic Relevance')# x['Topic Relevance'] = x['likert']

        # Convert float to string
        columns_to_convert = ['Topic Relevance', 'score', 'Impact Factor', 'Cite Score (2022)',
                            'SJR', 'Issues Per Year', 'Open Access % (Gold)']
        x[columns_to_convert] = x[columns_to_convert].astype(str)

        # Check image filenames
        directory = '/home/deepread2.0/assets/journal_img'
        filename_list = [filename for filename in os.listdir(directory)]

        # # Create column for png path
        img_list = []
        for i in x['title']:
            match = journal_df[journal_df['title']==i]['image_filename']#.values[0]
            if not match.empty:
                png_file = match.values[0]
                if png_file in filename_list:
                    img_list.append(f'{os.getenv("IMG_PATH_SYSTEM")}/{png_file}')
                else:
                    img_list.append(f'{os.getenv("IMG_PATH_SYSTEM")}/no_img.png')
            else:
                img_list.append(f'{os.getenv("IMG_PATH_SYSTEM")}/no_img.png')
        x["image_url"] = img_list
        ### nan 값 나오는거 방지 코드 나중에는 추천 알고리즘 자체에 대한 처리가 필요 ###
        x.fillna('N/A', inplace=True)
        df_list = x.to_dict(orient='records')
        for record in df_list:
            for key, value in record.items():
                if value == 'nan' or pd.isna(value):
                    record[key] = 'N/A'
        ### nan 값 나오는거 방지 코드 나중에는 추천 알고리즘 자체에 대한 처리가 필요 ###
        logger.debug(f'추천 저널 추출 >>> {time.time() - tmp_time }')
        # return df_list # for price logger 
        return df_list, total_price 
    
    except ValueError as e:
        logger.error(f"Error: {e}")

# Paper Recommender - Functions - Semantic Scholar
def semantic_scholar(query):
    # Access API key 
    ss_KEY = os.getenv("SEMANTICSCHOLAR_KEY")
    sch = SemanticScholar(api_key=ss_KEY, timeout=60)
    
    # keywords = [keyword.strip() for keyword in query.split(',')]  #might change to set
    keywords = set(keyword.strip() for keyword in query.split(','))
    len_key = 0
   
    # Handling error encountered with Search API
    while True:
        try:
            # Introduce a timeout parameter (adjust the value as needed)
            results = sch.search_paper(query)
            searched_papers = results.items
            break  # Break out of the loop if the search is successful
        except requests.exceptions.Timeout as te:
            # Handle timeout exception
            if not query:
                # If the query is empty, break the loop to avoid an infinite loop
                break
            # Handle the error by removing the last keyword
            keywords.pop()
            query = ', '.join(keywords)
            len_key += 1
        except Exception as e:
            # Handle other exceptions
            if not query:
                # If the query is empty, break the loop to avoid an infinite loop
                break
            # Handle the error by removing the last keyword
            keywords.pop()
            query = ', '.join(keywords)
            len_key += 1

    while len(searched_papers) < 7 and len(keywords) > 3:
        keywords.pop()
        len_key +=1
        query = ', '.join(keywords)
        results = sch.search_paper(query)
        searched_papers = results.items
    
    id1 = searched_papers[0].paperId   
    
    # Getting results using Recommendations API
    recommended_papers = sch.get_recommended_papers(id1)
        
    # Store first-5 Open Access Articles
    top_papers = []
    
    # List to store data of Top 5 Articles
    titles = []
    authors = []
    links = []
    pdf = []
    years = []
    citation_count = []
    abstracts_ss = []
    
    # If more output from Recommendations API
    if len(searched_papers) < 5 and len(recommended_papers) > 10:
        api = 'RECOMMENDATION'

        # Getting the 1st-top 5 relevant papers that are Open Access
        for paper in recommended_papers:
            if len(top_papers) == 5:
                break
            if paper.isOpenAccess == True:
                top_papers.append(paper)
            else:
                pass
            
    # Enough output from Search API
    # Getting the 1st-top 5 relevant papers that are Open Access
    else:
        if len(searched_papers) >= 5:
            n_sp = 5
        else:
            n_sp = len(searched_papers)
            
        api = 'SEARCH'    
        for paper in searched_papers:
            if len(top_papers) == n_sp:
                break
            if paper.isOpenAccess == True:
                top_papers.append(paper)      
            else:
                pass
                
    # Extracting data
    for paper in top_papers:
        ## TITLE
        try:
            title = paper.title
            titles.append(title)
        except Exception as e:
            titles.append('NA-Title')
        ## AUTHORS
        try:
            author_ = paper.authors
            authors_clean = [author['name'] for author in author_]
            authors.append(', '.join(authors_clean))
        except Exception as e:
            authors.append('NA-Author')
        ## LINKS
        try:
            link= paper.url
            links.append(link)
        except Exception as e:
            links.append('NA-Link')
        ## PDF LINKS
        try:
            file = paper.openAccessPdf
            pdf.append(file['url'])
        except Exception as e:
            pdf.append('NA-PDF')   
        ## YEARS
        try:
            year= paper.year
            years.append(year)
        except Exception as e:
            years.append('NA-Year')
        ## CITATION COUNT
        try:
            citation= paper.citationCount
            citation_count.append(citation)
        except Exception as e:
            citation_count.append('NA-Citation Count') 
        ## ABSTRACT
        try:
            abst = paper.abstract
            abstracts_ss.append(abst)
        except Exception as e:
            abstracts_ss.append('NA-Abstract')
            
    data = {
        'Titles': titles,
        'Authors': authors,
        'Links': links,
        'PDF Links': pdf, 
        'Years': years,  
        'Citation_Count': citation_count,
        'Abstracts': abstracts_ss
    }
    
    # Dataframe
    data_df = pd.DataFrame(data)
    
    return data_df

# Google Scholar(Backup)
def google_scholar_backup(query):
    q = query.replace(' ', '+')

    # Scraping google scholar
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
    url = f'https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={q}&btnG='

    def get_content(url):
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.error(f'Status Code: {response.status_code}')

        soup = BeautifulSoup(response.content, 'lxml')
        return soup
    
        # Extracting data from Google Scholar
    content = get_content(url)
    data = content.find_all('div', attrs = {'class': 'gs_r gs_or gs_scl'})
    if len(data) > 5:
        n = 5
    else:
        n = len(data)
        
    titles = []
    authors = []
    links = []
    pdf_links = []
    years = []
    citation_count = []
    
    ## Citation count & Years
    texts =  [data[i].find('div', attrs ={'class':'gs_fl gs_flb'}).get_text() for i in range(len(data))]
    yrs = [data[i].find('div', attrs ={'class':'gs_a'}).get_text() for i in range(len(data))]

    yr_pattern = r'(\d{4})'
    citation_pattern = r'Citations: (\d+)'

    for i in np.arange(n):
        ## TITLE
        try:
            title = data[i].find('h3').get_text()
            titles.append(title)
        except Exception as e:
            titles.append('NA-Title')
        ## AUTHORS
        try:
            author = data[i].find('div', attrs ={'class': 'gs_a'}).get_text().replace('\xa0', '-').split('-')[0]
            authors.append(author)
        except Exception as e:
            authors.append('NA-Author')
        ## LINKS
        try:
            link= data[i].find('h3', attrs={'class':'gs_rt'}).find('a')['href']
            links.append(link)
        except Exception as e:
            links.append('NA-Link')
        ## PDF LINKS
        try:
            pdf = data[i].find('a')['href']
            pdf_links.append(pdf)
        except Exception as e:
            pdf_links.append('NA-PDF')
        ## YEARS
        try:
            year= re.search(yr_pattern, yrs[i]).group(1)
            years.append(year)
        except Exception as e:
            years.append('NA-Year')
        ## Citation Count
        try:
            citation= re.search(citation_pattern, texts[i]).group(1)
            citation_count.append(citation)
        except Exception as e:
            citation_count.append('NA-Citation Count')
    
    data = {
        'Titles': titles,
        'Authors': authors,  
        'Years': years,  
        'Citation_Count': citation_count, 
        'Links': links,
        'PDF Links': pdf_links
    }
    
    # Creating the dataframe --Google Scolar
    gs_df = pd.DataFrame(data)
    
    return gs_df


# Paper Recommender Function
nest_asyncio.apply()
# Paper Recommender Function with Parallelization
async def paper_reco_parallel(query):
    if not query or (isinstance(query, list) == False and query.strip() == 'N/A'):
        return 'N/A'
    else:
        query = ', '.join(query)

        # Using ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for Semantic Scholar and Google Scholar searches
            future_semantic = executor.submit(semantic_scholar, query)
            future_google = executor.submit(google_scholar_backup, query)

            # Wait for results
            try:
                final_df = await asyncio.gather(
                    asyncio.to_thread(future_semantic.result),
                    asyncio.to_thread(future_google.result))

            except Exception as e:
                logger.error(f"Error in parallel execution: {e}")
                # If there's an error, use the result from future_google instead
                final_df = [None, await asyncio.to_thread(future_google.result)]
            
        if final_df[0] is not None and len(final_df[0]) > 0:
            df_list = final_df[0].to_dict(orient='records')
        elif final_df[1] is not None and len(final_df[1]) > 0:
            logger.info('Using results from Google Scholar')
            # If Semantic Scholar has no results, use Google Scholar results
            df_list = final_df[1].to_dict(orient='records')
        else:
            # If both Semantic Scholar and Google Scholar backup fail, return 'NA'
            return 'NA'

        return df_list
    

async def paper_recommender(query):
    tmp_time = time.time()
    result = await paper_reco_parallel(query)
    end_time = time.time()
    logger.debug(f'관련 논문 추출 >>> {time.time() - tmp_time}')
    return result


async def extract_title_keywords_summarization(abstract):
    tmp_time = time.time()
    title_prompts = prompt_engineering.make_prompt_title(abstract)
    keywords_prompts = prompt_engineering.make_prompt_keywords(abstract)
    summarization_prompts = prompt_engineering.make_prompt_summarization_v2(abstract)
    potential_topic_prompts = prompt_engineering.make_prompt_potential_topic(abstract)

    prompt_list = [title_prompts, keywords_prompts, summarization_prompts, potential_topic_prompts]
    async_gpt_result = await process_api_requests_from_prompt_list(
            prompt_list=prompt_list,
            model = "gpt-4-1106-preview",
            temperature = .0,
            api_key = OPENAI_KEY,
            request_url = "https://api.openai.com/v1/chat/completions",
            json_mode = True,
            desc = "extract title, keywords, summarization",
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

    parsed_result = [prompt_engineering.parse_json_string(i, ['result']) for i in result]

    titles, keywords, potential_topics = parsed_result[0]['result'], parsed_result[1]['result'], parsed_result[3]['result']

    # 요약 방식 변경 -> 주제, 연구 방법, 연구 결론 세가지로 나눠서 작성할 것
    summarize = prompt_engineering.parse_json_string(result[2], ['topic','method','conclusion'])
    logger.debug(f'제목 키워드 요약 추가연구제안 추출 >>> {time.time() - tmp_time}')
    # return titles, keywords, summarize, potential_topics
    return titles, keywords, summarize, potential_topics, price_list # for price logger 


async def get_published_information_semantic_scholar(title: str) -> dict:
    tmp_time = time.time()
    api_key = os.getenv("SEMANTICSCHOLAR_KEY")
    search_fields = ['url', 'year', 'authors', 'publicationVenue', 'title', 'abstract']
    timeout = 60
    limit = 1
    title = ','.join(title.split())

    try:
        try:
            sch = SemanticScholar(
                timeout=timeout,
                api_key=api_key,
            )

            results = sch.search_paper(
                query=title, 
                fields=search_fields,
                limit=limit,
            )
            if not isinstance(results, dict):
                results = dict(results[0])
        except:
            time.sleep(1)
            request_url = f"https://api.semanticscholar.org/graph/v1/paper/search?fields={','.join(search_fields)}"
            query_params = {
                'query': title,
                'limit': limit,
                'timeout': timeout
            }
            headers = {'x-api-key': api_key}
            response = requests.get(request_url, params=query_params, headers=headers)
            results = response.json()
            results = results['data'][0]

        publication_venue = results.get('publicationVenue') or {}
        published_info = {
            'published_journal': publication_venue.get('name', 'N/A'),
            'published_journal_url': publication_venue.get('url', 'N/A'),
            'paper_url': results.get('url', 'N/A'),
            'paper_title': results.get('title', 'N/A'),
            'paper_abstract': results.get('abstract', 'N/A'),
            'paper_published_year': results.get('year', 'N/A'),
            'paper_author': ', '.join([i['name'] for i in results.get('authors', [])]) or 'N/A'
        }

    except Exception as e:
        published_info = {
            'published_journal': 'N/A',
            'published_journal_url': 'N/A',
            'paper_url': 'N/A',
            'paper_title': 'N/A',
            'paper_abstract': 'N/A',
            'paper_published_year': 'N/A',
            'paper_author': 'N/A'
        }

    logger.debug(f'출판정보 추출 >>> {time.time() - tmp_time}')
    return published_info

async def get_paper_basic_info(text:str):
    price = .0 # for price logger
    prompt_list = [prompt_engineering.make_prompt_paper_basic_info(text)]
    result = [None] * len(prompt_list)
    result_list = await process_api_requests_from_prompt_list(
            prompt_list=prompt_list,
            model='gpt-4-1106-preview',
            temperature=.0,
            api_key=OPENAI_KEY,
            request_url="https://api.openai.com/v1/chat/completions",
            json_mode=True,
            desc="extract title, authors, abstract",
            timeout=100
    )
    for gpt_result in result_list:
        if gpt_result is not None:
            result[gpt_result[0]] = gpt_result[1][1]['choices'][0]['message']['content']
            price += (gpt_result[1][1]['usage']['prompt_tokens'] * 0.01 / 1000) + (gpt_result[1][1]['usage']['completion_tokens'] * 0.03 / 1000)

    result_dict = prompt_engineering.parse_json_string(result[0], ['title', 'author', 'abstract'])
    if 'title' in result_dict:
        result_dict['title'] = result_dict['title'][:1024]

    if 'author' in result_dict:
        result_dict['author'] = result_dict['author'][:2048]
    
    result_dict['paper_basic_info_price'] = price # for price logger
    return result_dict