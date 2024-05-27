import os  
import pandas as pd
import openai
TOKENIZER = "cl100k_base"
openai.api_key = OPENAI_KEY = os.getenv("OPENAI_KEY")
LATENCY = 0.95

pd.set_option('mode.chained_assignment',  None)
from tqdm import tqdm
import pickle
import aiohttp  
import asyncio  
import logging  
import tiktoken 
import time  
from dataclasses import dataclass, field 
import argparse  
import json 
import re  
import time
from tqdm import tqdm
import datetime
import json
import warnings
import asyncio
import nest_asyncio
import os
import numpy as np
import xml.etree.ElementTree as ET
from log.logger_config import logger, log_execution_time, log

nest_asyncio.apply()

max_attempts = 50
logging_level = 20 # 20

def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

def num_tokens_from_string(text: str, model_name: str="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(model_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    try:
        encoding = tiktoken.encoding_for_model(token_encoding_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token

            num_tokens += 2  # every reply is primed with <im_start>assistant
                        
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')

def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search('^https://[^/]+/v\\d+/(.+)$', request_url)
    return match[1]

def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits

@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        desc: str,
        timeout : int
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"{desc} Starting {request_url.rsplit('/', 1)[-1]} >>> #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1

                return self.task_id, data
                # append_to_jsonl(data, save_filepath)
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            # logging.debug(f"Request {self.task_id} saved to {save_filepath}")

            
            return self.task_id, data
            # append_to_jsonl(data, save_filepath)

async def process_api_requests_from_prompt_list(
    prompt_list: list,
    model: str = 'gpt-3.5-turbo',
    temperature: float =0,
    max_tokens: int = 500,
    frequency_penalty: int = 0,
    max_attempts: int = 50,
    max_requests_per_minute: float = 12000.0 * 0.95,
    max_tokens_per_minute: float = 1000000.0 * 0.95,
    logging_level: int = 20,
    request_url: str = "https://api.openai.com/v1/chat/completions",
    api_key: str = "YOUR_API_KEY",
    json_mode: bool = False,
    desc: str = "",
    timeout: int = 60,
    ):

    result_list = []
    if request_url.endswith('embeddings'):
        work_list = [{'model': model, 'input': prompt} for prompt in prompt_list]

    elif request_url.endswith('completions'):
        work_list = []
        for prompt in prompt_list:
            work_item = {
                'model': model,
                'messages': [{"role": "user", "content": prompt}],
                'temperature': temperature,
                'max_tokens': min(max(num_tokens_from_string(prompt) * 3, 1000), 4096),
                'frequency_penalty': frequency_penalty
            }
            if json_mode:
                work_item['response_format'] = {'type': 'json_object'}

            work_list.append(work_item)

    requests = iter(work_list)

    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    logging.debug(f"got prompt list. Entering main loop")

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
            elif file_not_finished:
                try:
                    # get new request
                    request_json = next(requests)
                    next_request = APIRequest(
                        task_id=next(task_id_generator),
                        request_json=request_json,
                        token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint, model),
                        attempts_left=max_attempts,
                        metadata=request_json.pop("metadata", None)
                    )

                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                except StopIteration:
                    # if file runs out, set flag to stop reading it
                    logging.debug("Read file exhausted")
                    file_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.token_consumption
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                api_result = asyncio.create_task(
                            next_request.call_api(
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                status_tracker=status_tracker,
                                desc=desc,
                                timeout=timeout
                            )
                        )

                result_list.append(api_result)

                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

    # after finishing, log final status
    logging.info(f"""Parallel processing complete.""")
    if status_tracker.num_tasks_failed > 0:
        logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed.")
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")

    # return result_list
    return await asyncio.gather(*result_list)

def current_time():
    now = datetime.datetime.now()
    return now.strftime('%y%m%d_%H%M%S')


def price(input_token, output_token, model):
    """자동으로 gpt 모델별 사용 금액을 계산해주는 함수"""
    if model == "gpt-3.5-turbo":
        return ((input_token * (0.0015/1000)) + (output_token * (0.002/1000)))
    elif model == 'gpt-4':
        return ((input_token * (0.03/1000)) + (output_token * (0.06/1000)))


def latency_manager(model, latency, openai_user):
    if openai_user == "OPENAI_LXCD":
        if model == 'gpt-4':
            max_requests_per_minute = 10000 * latency
            max_tokens_per_minute = 300000 * latency
        elif model == 'gpt-3.5-turbo':
            max_requests_per_minute = 12000 * latency
            max_tokens_per_minute = 1000000 * latency
        elif model == "text-embedding-ada-002":
            max_requests_per_minute = 10000 * latency
            max_tokens_per_minute = 10000000 * latency
    elif openai_user == "OPENAI_EQQUI":
        if model == 'gpt-4':
            max_requests_per_minute = 200 * latency
            max_tokens_per_minute = 20000 * latency
        elif model == 'gpt-3.5-turbo':
            max_requests_per_minute = 5000 * latency
            max_tokens_per_minute = 160000 * latency
        elif model == "text-embedding-ada-002":
            max_requests_per_minute = 10000 * latency
            max_tokens_per_minute = 10000000 * latency
    elif openai_user == "OPENAI_PH":
        if model == 'gpt-4':
            max_requests_per_minute = 200 * latency
            max_tokens_per_minute = 10000 * latency
        elif model == 'gpt-3.5-turbo':
            max_requests_per_minute = 5000 * latency
            max_tokens_per_minute = 160000 * latency
        elif model == "text-embedding-ada-002":
            max_requests_per_minute = 10000 * latency
            max_tokens_per_minute = 10000000 * latency
    else:
        raise ValueError(f"Unsupported model: {model}")
    return max_requests_per_minute, max_tokens_per_minute


async def call_gpt_for_chat(
    request_url: str, 
    request_header: dict,
    request_json: dict,
    timeout_seconds: int,
) -> list:
    
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_seconds))
    response = await session.post(url=request_url, headers=request_header, json=request_json)
    response = response.content
    return [request_json, response, session]

async def asyncio_gpt_for_chat(
    request_json_list: list,
    timeout_seconds: int = 120,
    request_url: str = "https://api.openai.com/v1/chat/completions"
):
    gpt_tasks = [
        call_gpt_for_chat(
            request_url=request_url,
            request_header={"Authorization": f"Bearer {OPENAI_KEY}"},
            request_json=request_json,
            timeout_seconds=timeout_seconds
        )
        for request_json in request_json_list
    ]
    gpt_results = await asyncio.gather(*gpt_tasks)
    return {"results": gpt_results}

async def close_session_gpt_for_chat(session):
    await session.close()