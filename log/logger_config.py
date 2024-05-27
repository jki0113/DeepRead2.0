import os
import time
import asyncio
import logging
import colorlog
from colorama import Fore, Style
from functools import wraps

# 로깅 레벨 및 로깅 파일 저장 위치 설정
LOGGING_LEVEL = os.environ.get('LOGGING_LEVEL')
SYSTEM_LOGGER_PATH = os.environ.get('SYSTEM_LOGGER_PATH') #pm2에서 log 경로를 지정해줬기 때문에 필요 없음

class ColoredFormatter(colorlog.ColoredFormatter):
    def format(self, record):
        formatted_record = super().format(record)
        lines = formatted_record.split('\n')
        first_line = lines[0]
        log_level_with_color = first_line.split('|')[0]
        colored_lines = [log_level_with_color + '|' + line if line else line for line in lines[1:]]
        return '\n'.join([first_line] + colored_lines)

# logger의 설정을 해줌
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOGGING_LEVEL))
ch = logging.StreamHandler()
ch.setLevel(getattr(logging, LOGGING_LEVEL))

log_colors = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'white,bg_red',
}

ch_formatter = ColoredFormatter(
    "%(log_color)s%(levelname)s | %(message)s",
    log_colors=log_colors
)
ch.setFormatter(ch_formatter)
logger.addHandler(ch)
logger.propagate = False

def log_execution_time(func):
    """ 비동기 함수에 대해 실행 시간을 로깅하는 데코레이터 """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        log('green', f"{func.__name__} | called")
        start_time = time.time()
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        log('green', f"{func.__name__} | executed in {elapsed_time:.5f} seconds")
        return result
    return wrapper

def log(color, text):
    """Log function that applies color to each line of a multi-line message."""
    color_prefix = f'{Style.BRIGHT}{getattr(Fore, color.upper())}'
    color_suffix = f'{Style.RESET_ALL}'
    lines = str(text).split('\n') # Split the text into lines
    colored_lines = [f"{color_prefix}{line}{color_suffix}" for line in lines] # Apply color to each line
    colored_message = '\n'.join(colored_lines) # Rejoin the colored lines
    print(colored_message)

def test_function():
    logger.debug("This is a debug message with multiple lines")
    logger.info("This is an info message with multiple lines")
    logger.warning("This is a warning message with multiple lines")
    logger.error("This is an error message with multiple lines")
    logger.critical("This is a critical message with multiple lines")
test_function()