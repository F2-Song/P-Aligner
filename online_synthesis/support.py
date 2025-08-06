import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import openai
import requests
import os

def create_batch_response_generator(
    model_path,
    device="cuda",
):
    return create_base_model_generator(model_path, device=device)

def create_batch_response_scorer(
    model_path,
    device="cuda",
):
    return create_armo_scorer(model_path, device=device)   

def create_lm_optimizer(
    optimizer_name="openai",
    model_path="gpt-4",
):
    if optimizer_name == "openai":
        return create_openai_optimizer(model_path)
    elif optimizer_name == "local":
        return create_instruct_model_optimizer(model_path)
    else:
        raise ValueError("Invalid optimizer name.")

def create_base_model_generator(
    model_path,
    device="cuda:0",
):
    def apply_chat_template(
        context, 
        tokenizer
    ):
        message_text = tokenizer.bos_token
        for message_index, message in enumerate(context):
            if message_index == len(context) - 1:
                assert message["role"] == "user"

            if message["role"] == "system":
                message_text += "\n\n[System]\n" + message["content"].strip()
            elif message["role"] == "user":
                message_text += "\n\n[User]\n" + message["content"].strip()
            elif message["role"] == "assistant":
                message_text += "\n\n[Assistant]\n" + message["content"].strip()
        message_text += "\n\n[Assistant]"

        return message_text

    language_model = LLM(
        model=model_path,
        enable_prefix_caching=True,
        device=device,
        gpu_memory_utilization=0.6,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def get_batch_responses(
        contexts, 
        num_return_sequences,
    ):
        prompts = [apply_chat_template(context, tokenizer) for context in contexts]
        if num_return_sequences == 1:
            sampling_params=SamplingParams(
                n=num_return_sequences,
                stop=[tokenizer.eos_token, "[System]", "[User]", "[Assistant]"],
                max_tokens=1024,
                temperature=0.0,
            )
        else:
            sampling_params=SamplingParams(
                n=num_return_sequences,
                stop=[tokenizer.eos_token, "[System]", "[User]", "[Assistant]"],
                max_tokens=1024,
            )
        outputs = language_model.generate(
            prompts,
            sampling_params=sampling_params,
        )
        responses = []
        for output in outputs:
            response = [output.outputs[i].text.strip() for i in range(num_return_sequences)]
            responses.append(response)
        return responses
    
    return get_batch_responses

def create_armo_scorer(
    model_path,
    device="cuda:1",
):
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    batch_size = 1

    def get_batch_scores(
        contexts,
    ):
        total_scores = []
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i+batch_size]
            processed_contexts = [
                tokenizer.apply_chat_template(
                    context,
                    tokenize=False
                ) for context in batch_contexts
            ]
            batch_inputs = tokenizer(
                processed_contexts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt"
            )
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

            with torch.no_grad():
                output = reward_model(**batch_inputs)
                scores = output.score.float().view(-1).tolist()
            total_scores.extend(scores)

        return total_scores

    return get_batch_scores

def create_openai_optimizer(model_path):
    client = openai.OpenAI(
        base_url=os.environ["OPENAI_API_BASE"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    
    def gpt_optimize_instruction(instruction, action, prompt):
        user_prompt = "The user query to be paraphrased is [{}]. \nYou should optimize this query by {}. \nYou should also return the optimized version directly, without any prefix.".format(instruction, prompt)
        
        failed_attempts = 0
        while failed_attempts < 4:
            try:
                completion = client.chat.completions.create(
                    model=model_path,
                    messages=[
                        {"role": "system", "content": "You are an assistant who helps optimize user queries."},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                res = completion.choices[0].message.content
            except Exception as e:
                failed_attempts += 1
                continue
            else:
                return res
                
        return None
    
    return gpt_optimize_instruction

def create_instruct_model_optimizer(
    model_path,
):
    client = openai.OpenAI(
        base_url=os.environ["OPENAI_API_BASE"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def get_optimized_instruction(instruction, action, prompt):
        user_prompt = tokenizer.get_context(
            instruction, prompt
        )

        if user_prompt.startswith(tokenizer.bos_token):
            user_prompt = user_prompt[len(tokenizer.bos_token):]

        failed_attempts = 0
        while failed_attempts < 4:
            try:
                completion = client.completions.create(
                    model=model_path,
                    prompt=user_prompt,
                    max_tokens=4096,
                    temperature=0.0,
                    echo=False,
                )
                res = tokenizer.parse_output(
                    completion.choices[0].text,
                    instruction
                )
            except Exception as e:
                failed_attempts += 1
                continue
            else:
                return res
            
        return None

    return get_optimized_instruction

def generate_responses(
    context,
    port
):
    result_of_responses = None
    try:
        response = requests.post(
            url="http://localhost:{}/generate".format(port),
            json={"messages": context}
        )
        
        if response.status_code == 200:
            result_of_responses = response.json()
        else:
            print(f"Error: {response.status_code}, {response.text}")
    
    except Exception as e:
        print(f"Request failed: {e}")
    
    if result_of_responses is None:
        return None
    
    return result_of_responses["responses"]

def score_responses(
    raw_context, 
    responses,
    port
):
    if responses is None:
        print("Responses are None.")
    result_of_scores = None
    try:
        response = requests.post(
            url="http://localhost:{}/score".format(port),
            json={"raw_messages": raw_context, "responses": responses}
        )

        if response.status_code == 200:
            result_of_scores = response.json()
        else:
            print(f"Error: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"Request failed: {e}")

    if result_of_scores is None:
        return None
    
    return result_of_scores["avg_score"], result_of_scores["scores"]
