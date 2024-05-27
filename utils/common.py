import os
import random
import string
from docx import Document
import subprocess
from ftlangdetect import detect
from log.logger_config import logger, log_execution_time, log


############################## Storage ##############################
# 고유한 ID 생성을 위한 string 랜덤 조합
async def generate_random_string(length=16):
    characters = string.ascii_lowercase + string.digits
    random_string = ''.join(random.choice(characters) for i in range(length))
    return random_string


# 프로젝트 저장
async def save_projects(user_id: str, project_id: str, project_file):
    directory_path = os.path.join(os.getenv('STORAGE_PATH_SYSTEM'), os.getenv('MODE'), user_id, project_id)
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(directory_path, project_file.filename)
    file_path = unicode_normalization(file_path)
    with open(file_path, 'wb') as file_out:
        while content := await project_file.read(1024):  # 파일을 조각내어 읽기
            file_out.write(content)
    
    converted_file_path = await convert_file_to_pdf(file_path)
    return converted_file_path


# pdf 파일로 변환
async def convert_file_to_pdf(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.pdf':
        return file_path
    elif ext == '.txt':
        docx_file_path = await convert_txt_to_docx(file_path)
        pdf_file_path = await convert_docx_to_pdf(docx_file_path)
        os.remove(docx_file_path)
        os.remove(file_path)
        return pdf_file_path
    elif ext == '.docx':
        pdf_file_path = await convert_docx_to_pdf(file_path)
        os.remove(file_path)
        return pdf_file_path
    else:
        raise ValueError(f'pdf 변환에 부합하지 않는 확장자임 -> {ext}')


# txt 파일을 docx 파일로 변환
async def convert_txt_to_docx(txt_file_path):
    doc = Document()
    with open(txt_file_path, 'r') as file:
        for line in file:
            doc.add_paragraph(line)
    docx_file_path = txt_file_path.replace('.txt', '.docx')
    doc.save(docx_file_path)
    return docx_file_path


# docx 파일을 pdf 파일로 변환
async def convert_docx_to_pdf(docx_file_path):
    output_pdf_path = docx_file_path.replace('.docx', '.pdf')
    subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', docx_file_path, '--outdir', os.path.dirname(output_pdf_path)])
    return output_pdf_path


# mac 조합형 한글 인코딩
import unicodedata
def unicode_normalization(string: str):
    normalized_string = unicodedata.normalize('NFC', string)
    return normalized_string


############################## Analysis ##############################
# 제목 추천, 요약, 추가 연구주제 추천 번역
async def check_text_lang(text:str) -> str:
    lang_code = {
        "en": "English",
        "ko": "Korean"
    }
    detected_lang = detect(text.replace('\n', ' '), low_memory=True)['lang']
    return lang_code.get(detected_lang, 'N/A')