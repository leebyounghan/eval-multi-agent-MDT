from openai import OpenAI
import os
import json
import argparse
import time 
import pdb
from sklearn.metrics import accuracy_score
from utils import load_qa_data
from typing import Literal
from pydantic import BaseModel

import method

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mdt')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--method', type=str, default='cot')
    parser.add_argument('--turns', type=int, default=1)
    parser.add_argument('--error', type=str, default='')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--few_shot', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    qa_dataset= load_qa_data(args.dataset, args.few_shot)

    model = args.model
    client = OpenAI()
    
    text_sample = qa_dataset[0:args.num_samples]

    # Load existing output file if it exists
    output_file = f'./outputs/{args.dataset}/{model}_{args.few_shot}_shot_{args.method}_{args.turns}_turns'+(f'_{args.error}_error' if args.method == 'group_chat_w_recruit_w_initial_error' else '') + f'_seed{args.seed}.jsonl'
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            text_sample = [json.loads(line) for line in f]

    outputs_response = []
    for idx, sample in enumerate(text_sample):
        time.sleep(1)

        query = sample['question']

        if args.method == 'cot':
            args.turns = 1
            decision = method.cot(query, args.seed)
            text_sample[idx][args.method] = decision
        elif args.method == 'sot':
            args.turns = 1
            # if 'chat_history' not in text_sample[idx] or text_sample[idx]['chat_history'] is None:
            chat_history, decision = method.sot(query, args.seed)
            text_sample[idx][args.method] = decision
            text_sample[idx]['chat_history'] = chat_history

        elif args.method == 'majority_vote':
            args.turns = 1
            # if 'chat_history' not in text_sample[idx] or text_sample[idx]['chat_history'] is None:
            chat_history, decision = method.majority_vote(query, args.seed)
            text_sample[idx][args.method] = decision
            text_sample[idx]['chat_history'] = chat_history
        elif args.method == 'majority_vote_w_recruit':
            args.turns = 1
            # if 'chat_history' not in text_sample[idx] or text_sample[idx]['chat_history'] is None:
            chat_history, decision = method.majority_vote_w_recruit(query, args.seed)
            text_sample[idx][args.method] = decision
            text_sample[idx]['chat_history'] = chat_history
        elif args.method == 'group_chat':
            # print(idx)
            # if 'chat_history' not in text_sample[idx] or text_sample[idx]['chat_history'] is None:
            chat_history, decision = method.group_chat(query, args.turns, args.seed)
            text_sample[idx][args.method] = decision
            text_sample[idx]['chat_history'] = chat_history
        elif args.method == 'group_chat_w_recruit':
            # print(idx)
            # if 'chat_history' not in text_sample[idx] or text_sample[idx]['chat_history'] is None:
            chat_history, decision = method.group_chat_w_recruit(query, args.turns, args.seed)
            text_sample[idx][args.method] = decision
            text_sample[idx]['chat_history'] = chat_history
        elif args.method == 'group_chat_w_restricted_role':
            chat_history, decision = method.group_chat_w_restricted_role(query, args.turns, args.seed)
            text_sample[idx][args.method] = decision
            text_sample[idx]['chat_history'] = chat_history
        elif args.method == 'majority_vote_w_recruit_and_group_chat':
            chat_history, decision = method.majority_vote_w_recruit_and_group_chat(query, args.turns, args.seed)
            text_sample[idx][args.method] = decision
            text_sample[idx]['chat_history'] = chat_history
        elif args.method == 'group_chat_w_recruit_w_initial_error':
            chat_history, decision = method.group_chat_w_recruit_w_initial_error(query, args.turns, args.error, args.seed)
            text_sample[idx][args.method] = decision
            text_sample[idx]['chat_history'] = chat_history

        

        with open(output_file, 'w') as f:
            for item in text_sample:
                json.dump(item, f)
                f.write('\n')
    # print(sum(pred_correct)/len(pred_correct))
