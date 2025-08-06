import os
print("pid: ", os.getpid())
import asyncio
from fastapi import FastAPI, Request
from typing import List
import argparse
from support import (
    create_batch_response_generator, 
    create_batch_response_scorer,
)
import threading
import signal
import time


app = FastAPI()

batch_generate_lock = asyncio.Lock()
batch_score_lock = asyncio.Lock()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument("--flag_path", type=str)
    parser.add_argument("--lm_model_path", type=str)
    parser.add_argument("--rm_model_path", type=str)
    return parser.parse_args()

args = parse_args()

get_batch_responses = create_batch_response_generator(
    model_path=args.lm_model_path,
    device="cuda",
)

get_batch_scores = create_batch_response_scorer(
    model_path=args.rm_model_path,
    device="cuda",
)

class TaskOfGenerate:
    def __init__(self, messages):
        self.messages = messages
        self.responses = None
        self.event = asyncio.Event()

class TaskOfScore:
    def __init__(self, raw_messages, responses):
        self.raw_messages = raw_messages
        self.responses = responses
        self.scores = None
        self.avg_score = None
        self.event = asyncio.Event()

async def batch_generate():
    global buffer_generate, timeout_generate
    
    while True:
        await generate_pending_event.wait()

        async with generate_buffer_lock:
            tasks_to_generate = buffer_generate[:generate_buffer_size] if len(buffer_generate) >= generate_buffer_size else buffer_generate[:]
            buffer_generate = buffer_generate[generate_buffer_size:] if len(buffer_generate) >= generate_buffer_size else []
            generate_pending_event.clear()
            timeout_generate = None
            
        if tasks_to_generate:
            async with batch_generate_lock:
                queries = [task.messages for task in tasks_to_generate]
                responses = get_batch_responses(
                    contexts = queries, 
                    num_return_sequences=num_return_responses,
                )
                
                for i, task in enumerate(tasks_to_generate):
                    task.responses = responses[i]
                    task.event.set()

async def batch_score():
    global buffer_score, timeout_score

    while True:
        await score_pending_event.wait()

        async with score_buffer_lock:
            tasks_to_score = buffer_score[:score_buffer_size] if len(buffer_score) >= score_buffer_size else buffer_score[:]
            buffer_score = buffer_score[score_buffer_size:] if len(buffer_score) >= score_buffer_size else []
            score_pending_event.clear()
            timeout_score = None

        if tasks_to_score:
            async with batch_score_lock:
                raw_queries = [task.raw_messages for task in tasks_to_score]
                responses = [task.responses for task in tasks_to_score]
                new_queries = []
                for query, sub_responses in zip(raw_queries, responses):
                    for sub_response in sub_responses:
                        new_query = query + [{
                            "role": "assistant",
                            "content": sub_response,
                        }]
                        new_queries.append(new_query)
                scores = get_batch_scores(new_queries)
                scores = [
                    scores[i:i+num_return_responses] for i in range(0, len(scores), num_return_responses)
                ]
                avg_scores = [sum(score) / len(score) for score in scores]

                for i, task in enumerate(tasks_to_score):
                    task.scores = scores[i]
                    task.avg_score = avg_scores[i]
                    task.event.set()
                
async def start_generate_timeout():
    await asyncio.sleep(generate_timeout_seconds)
    async with generate_buffer_lock:
        if buffer_generate:
            generate_pending_event.set()

async def start_score_timeout():
    await asyncio.sleep(score_timeout_seconds)
    async with score_buffer_lock:
        if buffer_score:
            score_pending_event.set()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_generate())
    asyncio.create_task(batch_score())

@app.post("/generate")
async def generate(request: Request):
    global timeout_generate
    
    data = await request.json()
    task = TaskOfGenerate(
        messages = data['messages'],
    )
    async with generate_buffer_lock:
        buffer_generate.append(task)
        if len(buffer_generate) >= generate_buffer_size:
            generate_pending_event.set()
        elif not timeout_generate:
            timeout_generate = asyncio.create_task(start_generate_timeout())

    await task.event.wait()

    return {
        "responses": task.responses,
    }

@app.post("/score")
async def score(request: Request):
    global timeout_score

    data = await request.json()
    task = TaskOfScore(
        raw_messages = data['raw_messages'],
        responses = data['responses'],
    )
    assert task.responses is not None
    async with score_buffer_lock:
        buffer_score.append(task)
        if len(buffer_score) >= score_buffer_size:
            score_pending_event.set()
        elif not timeout_score:
            timeout_score = asyncio.create_task(start_score_timeout())

    await task.event.wait()

    return {
        "scores": task.scores,
        "avg_score": task.avg_score,
    }

def shutdown_server(
        flag_path,
    ):
    while True:
        time.sleep(10)
        if os.path.exists(
            flag_path
        ):
            pass
        else:
            os.kill(os.getpid(), signal.SIGINT)
            break

if __name__ == "__main__":
    buffer_generate = []
    generate_buffer_lock = asyncio.Lock()
    generate_pending_event = asyncio.Event()
    generate_buffer_size = 10
    buffer_score = []
    score_buffer_lock = asyncio.Lock()
    score_pending_event = asyncio.Event()
    score_buffer_size = 6
    num_return_responses = 3
    generate_timeout_seconds = 10
    score_timeout_seconds = 5
    timeout_generate = None
    timeout_score = None
    flag_path = args.flag_path

    threading.Thread(target=shutdown_server, args=(flag_path,)).start()

    print("Starting FastAPI server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)