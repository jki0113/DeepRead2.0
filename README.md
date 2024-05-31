# DeepRead2.0  

## Table of Contents

- [Introduction](#introduction)
- [How to Run](#how-to-run)
   - [Local](#local)
   - [Deploy Server](#deploy-server)
- [File Tree](#file-tree)

## Introduction
- 저널랩의 DeepRead2.0은 논문 분석 및 챗봇 시스템 입니다.
- 유저가 논문을 업로드하면 제목, 요약, 연구 방법론 등을 분석합니다.
- 챗봇을 통해 정해진 항목에 대한 분석 뿐만 아니라 유저가 실시간으로 채팅을 하며 대화할 수 있습니다.
- 또한 챗봇의 드래그 기능을 활용하여 웹 검색, 용어 검색, 부분 요약, 번역 등의 기능이 있습니다.
---
## How to Run
`git clone https://github.com/LexcodeHQ/DeepRead2.0.git`  
`cd DeepRead2.0`  

- 애플리케이션을 실행하려면 다음 파일이 필요합니다. 아래 경로에 따라 담당자를 통해 이 파일들을 업데이트하세요:
    - `/home/deepread2.0_dev/assets/journal_data/journal_embedding_openai_v5_preprocessed.parquet` # journal embedding files
    - `/home/deepread2.0/.env~` # Python Enviorment Files
- `.env` 파일에 `OPENAI_KEY`, `HOST`, `USER`, `PASSWORD`, `DB` 등을 입력하여 초기 값들을 설정하세요.     
   ```
   # Mode
   MODE = <APP_MODE(dev/prod)>
   DOCS_PATH = <SWAGGER_DOCS_PATH>

   # logging level
   LOGGING_LEVEL = 'DEBUG'

   # logger path
   SYSTEM_LOGGER_PATH = '/home/deepread2.0/log'
   HARDWARE_LOGGER_PATH = '/home/deepread2.0/log'

   # API KEY
   OPENAI_KEY = "<OPENAI_API_KEY>"
   SERPAPI_KEY = "<SERP_API_KEY>"
   SEMANTICSCHOLAR_KEY = "<SEMANTIC_SCHOLAR_API_KEY>"
   DEEPL_KEY = "<DEEPL_API_KEY>"

   # MySQL DataBase
   DB_HOST = '<DB_HOST>'
   DB_USER = '<DB_USER>'
   DB_PASSWORD = '<DB_PASSWORD>'
   DB_DATABASE = '<DB_DATABASE>'
   DB_PORT = '<DB_PORT>'
   DB_CHARSET='utf8mb4'

   # system base path
   SYS_ROOT_PATH = '/home/deepread2.0'
   SYS_PATH = '/home/deepread2.0'
   URL_PATH = 'https://deepread.journallab.com/deepread'

   STORAGE_PATH_SYSTEM = '/home/deepread2.0/storage'
   STORAGE_PATH_URL = 'https://deepread.journallab.com/deepread/storage'

   IMG_PATH_SYSTEM = '/home/deepread2.0/assets/journal_img'
   IMG_PATH_URL = 'https://deepread.journallab.com/deepread/journal_img'

   # Survey id info
   # survey info - tutorial 
   SURVEY_TUTORIAL = 'SL_240111_001' 
   SURVEY_TUTORIAL_Q1 = 'SQ_240111_001'

   # survey info - satisfaction 3
   SURVEY_SATISFACTION_V1 = 'SL_240112_001'
   SURVEY_SATISFACTION_V1_Q1 = 'SQ_240112_001'
   SURVEY_SATISFACTION_V1_Q2 = 'SQ_240112_002'

   # survey info - satisfaction 7
   SURVEY_SATISFACTION_V2 = 'SL_240117_001'
   SURVEY_SATISFACTION_V2_Q1 = 'SQ_240117_001'
   SURVEY_SATISFACTION_V2_Q2 = 'SQ_240117_002'
   ```  

   ### Local

   - 환경에 맞게 도커 컴포즈 파일을 구성하세요. (docker-compose.dev.yml)
      ```
      version: '3'
      services:
      deepread:
         build:
            context: <YOUR_DEEPREAD_ROOT_PATH>
            dockerfile: <YOUR_SYSTEM_ARCHITECTURE_DOCKERFILE>
         command: tail -f /dev/null
         container_name: deepread_container
         image: deepread2.0
         ports:
            - "8000:8000"
            - "8080:8080"
            - "8501:8501"
      ```
      
   - 컴포즈 파일을 사용하여 도커 환경을 구축하세요.  
   `docker-compose -f docker-compose.dev.yml up -d --build`  

   - 종속성을 설치하고 백엔드 FastAPI 서버를 실행하세요 (localhost:8000)  
   `docker exec -it deepread_container /bin/bash`  
   `pip install -r requirements.txt`  
   `pm2 start main_dev.py --name backend --max-memory-restart 6000M --interpreter python --log-date-format "YYYY-MM-DD HH:mm:ss" --log /home/deepread2.0/log/sys.log`  
      
   - (사용 중단) 다른 터미널에서 프론트엔드 Streamlit 서버를 실행하세요 (localhost:8080)  
   `docker exec -it deepread_container /bin/bash`  
   `cd web`  
   `streamlit run web.py`  

   ### Deploy Server

   - 컴포즈 파일을 사용하여 도커 환경을 구축하세요.  
   `docker-compose -f docker-compose.yml up -d --build`  

   - 종속성을 설치하세요  
   `pip install -r requirements.txt`  

   - 백엔드 FastAPI 서버를 실행하세요 (localhost:8000)  
   `pm2 start main.py --name backend --max-memory-restart 6000M --interpreter python --log-date-format "YYYY-MM-DD HH:mm:ss" --log /home/deepread2.0/log/sys.log`  
      
   - 로거를 실행하세요  
   `pm2 start log/hw_logger.py --name logger --interpreter python`  
---

## File Tree

   ```
   /home/deepread2.0
   |-- Dockerfile_ARM64
   |-- Dockerfile_x86
   |-- README.md
   |-- app
   |   |-- database
   |   |   |-- db_connection.py
   |   |   `-- models.py
   |   |-- queries
   |   |   |-- analysis.py
   |   |   |-- chat.py
   |   |   |-- storage.py
   |   |   `-- survey.py
   |   |-- routers
   |   |   |-- analysis.py
   |   |   |-- chat.py
   |   |   |-- mkt_analytics.py # 마케팅 팀 이용자 현황 통계 다운로드 용
   |   |   |-- storage.py
   |   |   `-- survey.py
   |   |-- schemas
   |   |   |-- analysis.py
   |   |   |-- chat.py
   |   |   |-- storage.py
   |   |   `-- survey.py
   |   `-- services
   |       |-- analysis.py
   |       |-- chat.py
   |       |-- storage.py
   |       `-- survey.py
   |-- docker-compose.dev.yml
   |-- docker-compose.yml
   |-- assets
   |   |-- journal_data
   |   |       `-- journal_embedding_openai_v5_preprocessed.parquet
   |   `-- journal_img
   |           `-- <JOURNAL_IMG>
   |-- log
   |   |-- hw.log
   |   |-- hw_logger.py
   |   |-- logger_config.py
   |   `-- sys_dev.log
   |-- storage
   |   |-- dev
   |   |   `-- <USER_CODE>
   |   |       `-- <PROJECT_CODE>
   |   |           |-- <PROJECT_FILE>
   |   |           `-- document_info.json
   |   `-- prod
   |       `-- <USER_CODE>
   |           `-- <PROJECT_CODE>
   |               |-- <PROJECT_FILE>
   |               `-- document_info.json
   |-- main.py # 실서버
   |-- main_dev.py # 개발서버
   |-- requirements.txt
   |-- utils # 기능 함수
   |   |-- async_gpt.py
   |   |-- chatbot.py
   |   |-- common.py
   |   |-- document_processor.py
   |   |-- paper_analysis.py
   |   |-- price_logger.py
   |   |-- prompt_engineering.py
   |   `-- translator.py
   `-- web # 웹 데모페이지(현재 사용 안함)
      |-- __init__.py
      |-- run_web.py
      |-- web.py
      |-- web_display.py
      `-- web_session_manager.py
   ```