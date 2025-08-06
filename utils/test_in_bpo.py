import os
os.environ["OPENAI_API_BASE"] = "xxx"
os.environ["OPENAI_API_KEY"] = "xxx"

import argparse
import json
from tqdm import tqdm
import os
import time
import random
import openai
import concurrent.futures
from utils import load_raw_dataset
    
def chat_gpt_once(message):
    client = openai.OpenAI(
        base_url=os.environ["OPENAI_API_BASE"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=message["message"],
                temperature=0.0,
            )
            res = completion.choices[0].message.content
            message['response'] = res
            break
        except Exception as e:
            if "Can you write \"lol\" 5000 times?" in message["message"][1]["content"]:
                if "[The Start of Assistant A's Answer]\nSorry, but I can't assist with that." in message["message"][1]["content"]:
                    message['response'] = "\n\nFinal Verdict: [[A]]"
                else:
                    message['response'] = "\n\nFinal Verdict: [[B]]"
                break
            print(e)
            time.sleep(1)
    
    return "success"

def get_messages_list(input_file_a, input_file_b):
    d_a = load_raw_dataset(input_file_a)
    d_b = load_raw_dataset(input_file_b)
    
    prompt = {
        "name": "pair-v2", 
        "type": "pairwise", 
        "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.", 
        "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]", 
        "description": "Prompt for general questions", 
        "category": "general", 
        "output_format": "[[A]]"
    }
    
    messages_list = []

    for i,j in zip(d_a, d_b):
        assert i["source"] == j["source"]
        
        # take the same strategy as its original implementation in https://github.com/thu-coai/BPO
        # [Black-Box Prompt Optimization: Aligning Large Language Models without Model Training](https://arxiv.org/abs/2311.04155)
        if random.randint(0, 1) == 0:
            option_a = "golden"
            res_a = i['model_output']
            res_b = j['model_output']
        else:
            option_a = "model"
            res_a = j['model_output']
            res_b = i['model_output']
        
        question = i['context'][0]["content"].strip()
    
        messages_list.append({'message': [
                {"role": 'system', "content": prompt['system_prompt']},
                {"role": "user", "content": prompt['prompt_template'].replace('{question}', question).replace('{answer_a}', res_a).replace('{answer_b}', res_b)}
            ],
            'source': i["source"],
            'option_a': option_a,
        })
        
    return messages_list

def cal_overall(input_file, judge_key):
    with open(input_file) as f:
        l = f.readlines()
    w_l_t = [0, 0, 0]
    num = 0

    for i in l:
        i = json.loads(i)
        if "[[A]]" in i['response'].split('\n\n')[-1]:
            if judge_key in i['option_a']:
                num += 1
                w_l_t[1] += 1
            else:
                w_l_t[0] += 1
        elif "[[B]]" in i['response'].split('\n\n')[-1]:
            if judge_key in i['option_a']:
                num += 1
                w_l_t[0] += 1
            else:
                w_l_t[1] += 1
        elif "[[C]]" in i['response'].split('\n\n')[-1]:
            if judge_key in i['option_a']:
                num += 1
            w_l_t[2] += 1

    print(w_l_t)
    print(f"Origin v.s. {judge_key}, win lose tie: ", [i/len(l) for i in w_l_t])
    print(f"Origin v.s. {judge_key}, win+tie vs lose: ", [(w_l_t[0]+w_l_t[2])/len(l), w_l_t[1]/len(l)])
    print(f"{judge_key} as first: ", num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_a", type=str, default="../data/bpo_eval/bpo_test.json")
    parser.add_argument("--input_file_b", type=str, help="your model's output file")
    parser.add_argument("--output_file", type=str, help="result file of comparison")
    parser.add_argument("--mode", type=str, help="compare | check")
    args = parser.parse_args()

    input_file_a = args.input_file_a
    input_file_b = args.input_file_b
    output_file = args.output_file
    mode = args.mode

    if mode == "compare":
        if not os.path.exists(output_file):
            x = open(output_file, 'w')
            x.close()

        messages_list = get_messages_list(input_file_a, input_file_b)
        print("total num: ", len(messages_list))

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            pbar = tqdm(total=len(messages_list))
            for re in executor.map(chat_gpt_once, messages_list):
                if re == "success":
                    pbar.update(1)
            pbar.close()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for message in messages_list:
                f.write(json.dumps(message, ensure_ascii=False)+'\n')

        print("All tasks are completed.")
    elif mode == "check":
        cal_overall(output_file, "model")
    else:
        raise ValueError(f"Invalid mode: {mode}")