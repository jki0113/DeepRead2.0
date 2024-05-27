import os
import sys
from dotenv import load_dotenv
load_dotenv('.env.prod')
sys.path.insert(0, '/home/deepread2.0')

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routers import storage
from app.routers import chat
from app.routers import analysis
from app.routers import survey
from app.routers import mkt_analytics

app = FastAPI(
    root_path="/prod",
    docs_url = os.getenv('DOCS_PATH'),
    redoc_url = None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(storage.router, prefix="/storage", tags=['storage'])
app.include_router(chat.router, prefix="/chat", tags=['chat'])
app.include_router(analysis.router, prefix="/analysis", tags=['analysis'])
app.include_router(survey.router, prefix='/survey', tags=['survey'])
app.include_router(mkt_analytics.router, prefix='/mkt_analytics', tags=['mkt_analytics'])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
    