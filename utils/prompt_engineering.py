import json
from log.logger_config import logger, log_execution_time, log

def parse_json_string(json_string, key_list):
    result = {key: "N/A" for key in key_list}
    try:
        data = json.loads(json_string)
        for key in key_list:
            result[key] = data.get(key, "N/A")
    except json.JSONDecodeError:
        try:
            json_string = json_string.strip()[1:-1]  # 양 끝의 중괄호 제거
            for key in key_list:
                key_index = json_string.find(f'"{key}":')
                if key_index != -1:
                    start_index = key_index + len(f'"{key}":')
                    end_index = json_string.find(',', start_index)
                    end_index = end_index if end_index != -1 else len(json_string)
                    value = json_string[start_index:end_index].strip()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    result[key] = value
        except Exception:
            pass
    return result

def make_prompt_paper_basic_info(preprocessed_text):
    prompt = f"""Your role is to extract the title, author, and abstract of the paper based on the source taken from the first front part of the paper pdf uploaded by the user.
    
- Extract the results in a json format with the keys: title, author, and abstract.
- All values of the JSON keys must be extracted in string type.
- If you cannot extract the title, author, or abstract from the given information, respond with 'N/A'.
- Be cautious when extracting since there's no guarantee that the user will only upload papers. Do not make assumptions or generate answers arbitrarily.
- Extract information based on the language written in the thesis.

```source:{preprocessed_text}
"""
    return prompt

def make_prompt_title(abstract):
    prompt = f"""Your role is to recommend five paper titles based on the given research abstract. 
- Return the results in JSON format with 'result' as the key and the value as a python list.
- Answer in English

```abstract : {abstract}"""
    return prompt
    
def make_prompt_keywords(abstract):
    prompt = f"""Your role is to recommend a minimum of 5 and a maximum of 10 keywords in order of importance from the given abstract. 
- Return the results in JSON format with 'result' as the key and the value as a python list.
- Answer in English

```abstract : {abstract}"""
    return prompt
    
def make_prompt_summarization(abstract): #(deprecated)
    prompt = f"""Your role is to write a summary based on the given research abstract. 
- Return the results in JSON format with 'result' as the key and the value as a string.
- Answer in English

```abstract : {abstract}"""
    return prompt

def make_prompt_summarization_v2(abstract):
    prompt = f"""Your role is to summarize the abstract based on the provided text, dividing it into three categories: research topic, research method, and research conclusion. 
- Return the results in JSON format with 'topic', 'method', and 'conclusion' as the keys and their summaries as the values in string format.
- Answer in English

```abstract : {abstract}"""
    return prompt

    
def make_prompt_potential_topic(abstract):
    prompt = f"""Your role is to recommend five potential research topics based on the given research abstract. 
- Return the results in JSON format with 'result' as the key and the value as a python list.
- Answer in English

```abstract : {abstract}"""
    return prompt

def make_prompt_chatbot(question, candidate_list, language):
    full_prompt = """You are the "DeepRead2.0 Chatbot," responding to user questions based on information extracted from research papers.

- The system provides information by extracting the most relevant answer from the full text of the paper to the user's question.
- Answer the user's question based on the extracted information.
- Communication should be clear, concise, and in an academic tone.
- If the response information is long, use appropriate line breaks, numbering, or symbols like · for readability.
- All answers to questions must be logically reasoned and must mention and explain the corresponding page information where relevant.
- If a question is too vague to find a definite answer, request the user to ask more specifically.
- If a question is irrelevant to the information in the paper, respond that it cannot be answered because it is unrelated to the paper.\n\n
"""
    full_prompt += f'```Extracted Information from Prompt:\n'
    full_prompt += '\n\n'.join([str(candidate) for candidate in candidate_list])
    full_prompt += '\n\n'
    full_prompt += '```Questions:'
    full_prompt += question
    full_prompt += f'(Answer in {language})'

    return full_prompt  

def make_prompt_chat_classification(input_chat):
    prompt = f"""
Your role is to classify the type of user question based on the following criteria:
A: In cases not related to the following items, or if no question was asked (most cases are included here)
B: Asking for the title, author, and abstract of this paper.
C: Asking for suggested titles, keywords, summaries, or additional research topics for this paper.
D: Inquiring about journal recommendations or suitable journals for submission of this paper.
E: Seeking references or related papers associated with this paper.

Display the probability for each options in percentag (A, B, C, D, E)

```user question: {input_chat}
"""
# 당신의 역할은 아래 기준을 바탕으로 질문을 분류하는 것 입니다.
# 기준
#   - 챗봇 : 채팅의 맥락 및 논문의 내용을 파악해서 질문해야 하는 경우 입니다.
#   - 기본정보 : 논문의 제목, 저자, 초록과 관련된 질문이 포함됩니다.
# ```Question: {input_chat}
# ```Result:
    return prompt

def make_prompt_recommended_questions(user_chat:str, assistant_chat:str, response_language: str) -> str:
    prompt = f"""Your role is to recommend 3 additional questions a user might ask based on the given user's chat and assistant responses. 
- If the user's query is unrelated to the content of the document or if the information requested by the user cannot be found in the document, return 'unk' instead of recommending additional questions.
- Questions should be recommended in {response_language}.
- Return the results in JSON format with 'result' as the key and the value as a python list containing 'unk' if applicable, or three relevant questions otherwise.

```user: {user_chat}
```assistant: {assistant_chat}
"""
    return prompt

def content_summary(user_chat:str, response_language:str):
    prompt = f"""Your role is to write a summary based on the given text. 
- Answer in {response_language}

```text: {user_chat}
"""
    return prompt
    
def translate(user_chat:str, response_language:str):
    prompt = f"""Your role is to write a translate the given text in {response_language}. 

```text: {user_chat}
"""
    return prompt

def make_prompt_chat_web_search_query(user_chat:str, response_language:str):
    prompt=f"""Your role is to generate a query for a Google search from the given text. 
The text below is what the user has highlighted and searched for in a PDF. 
Based on this text, create a search query. 
- Return the result in JSON format with 'result' as the key and the search query as a string value inside.
```text: {user_chat}
"""
    return prompt

# def make_prompt_chat_explain_term(user_chat:str, response_language:str):
#     prompt=f"""Your role is to generate terms for a glossary from the provided text. 
# - The text below is what the user has highlighted and searched for in a glossary within a PDF. 
# - Based on this text, create up to five glossary queries and respond in order of importance. 

# - Return the result in JSON format with 'result' as the key and the search queries as strings inside a list.
# ```text: {user_chat}
# """
#     return prompt

def make_prompt_chat_explain_term(user_chat:str, response_language:str):
    prompt=f"""Your role is to generate terms for a glossary from the provided text. 
- The text below is what the user has highlighted and searched for in a glossary within a PDF. 
- If the input text is less than five words, extract only that word or phrase as a glossary term. 
- If the input text consists of five words or more, create up to five glossary queries based on the text, and respond in order of importance.
- Return the result in JSON format with 'result' as the key and the search queries as strings inside a list.
```text: {user_chat}
"""
    return prompt


def make_prompt_for_term_search(input_text, information_list, response_language):
    prompt=f"""Your task is to analyze the given text and extract relevant information to understand a specific term. 
- You need to identify the term, provide its definition, and explain how it is used in the context of information. 
- Please present your findings in JSON format, using three keys: 'term' for the given term, 'explanation' for the definition of the term, and 'method' for how the term is applied or used in the context of the paper. 

```term: {input_text}
"""
    for info in information_list:
        info = info.split('\n')
        info = ' '.join(info)
        prompt += f"\n\n```information : {info}"
    prompt += f"\n - Answer in {response_language}"    

    return prompt