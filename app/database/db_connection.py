import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import sqlalchemy.exc
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from functools import wraps
from log.logger_config import logger, log_execution_time, log

DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')  
DB_PASSWORD = os.getenv('DB_PASSWORD') 
DB_DATABASE = os.getenv('DB_DATABASE')
DB_PORT = os.getenv('DB_PORT')
DB_CHARSET = os.getenv('DB_CHARSET', 'utf8mb4')

ASYNC_ENGINE = None
ASYNC_SESSION_FACTORY = None
SYNC_ENGINE = None
SYNC_SESSION_FACTORY = None

def create_db_connection(mode: str = "async"):
    global ASYNC_ENGINE, ASYNC_SESSION_FACTORY, SYNC_ENGINE, SYNC_SESSION_FACTORY

    connection_info = f"{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}?charset={DB_CHARSET}"

    if mode == "async":
        if ASYNC_ENGINE is None or ASYNC_SESSION_FACTORY is None:
            database_url = f"mysql+aiomysql://{connection_info}"
            ASYNC_ENGINE = create_async_engine(database_url)
            ASYNC_SESSION_FACTORY = sessionmaker(ASYNC_ENGINE, expire_on_commit=False, class_=AsyncSession)
        return ASYNC_SESSION_FACTORY

    elif mode == "sync":
        if SYNC_ENGINE is None or SYNC_SESSION_FACTORY is None:
            database_url = f"mysql+pymysql://{connection_info}"
            SYNC_ENGINE = create_engine(database_url)
            SYNC_SESSION_FACTORY = sessionmaker(bind=SYNC_ENGINE)
        return SYNC_SESSION_FACTORY

    else:
        raise ValueError(f"{mode} mode is not supported")


connection_info = f"{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}?charset={DB_CHARSET}"
database_url = f"mysql+aiomysql://{connection_info}"

engine = create_async_engine(database_url, pool_recycle=21600)
async def get_async_db():
    async with AsyncSession(bind=engine) as db:
        try:
            yield db
        finally:
            await db.close()

engine_read_committed = create_async_engine(database_url, isolation_level="READ COMMITTED", pool_recycle=21600)
async def get_async_db_read_committed():
    """
    READ COMMITTED isolation level을 사용하는 별도의 엔진으로 트랜잭션 내에서 커밋되지 않은 다른 트랜잭션의 변경 사항을 읽을 수 있도록 허용하므로 데이터 정합성과 일관성에 영향을 줄 수 있음
    Rotuer 단에서 DB 의존성을 주입하는 경우 중 analysis 파트에서 조회 시에만 사용되어야 하며, 다른 부분에서는 일반적인 get_async_db 함수를 사용해야 함
    """
    async with AsyncSession(bind=engine_read_committed) as db:
        try:
            yield db
        finally:
            await db.close()