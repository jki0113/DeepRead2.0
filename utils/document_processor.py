from fastapi import UploadFile
import os
import docx
import fitz 


async def extract_text_from_pdf(file_path: str) -> str:
    print('extract_text_from_pdf 도착')
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    print(text[:300])
    return text

async def extract_text_from_docx(file_path: str) -> str:
    print('extract_text_from_docx 도착')
    text = ""
    doc = docx.Document(file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    print(text[:300])
    return text

async def extract_text_from_txt(file_path: str) -> str:
    print('extract_text_from_txt 도착')
    with open(file_path, 'r', encoding="utf-8-sig") as file:
        text = file.read()
    print(text[:300])
    return text
