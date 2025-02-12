import json
from openai import OpenAI
from typing import Literal
from pydantic import BaseModel
import pandas as pd

class ModelComparisonAnswer(BaseModel):
    correct:Literal['Yes', 'No']

client = OpenAI()

# gpt-4o_0_shot_group_chat_w_recruit_w_initial_error_8_turns_gene_therapy_error_seed0.jsonl
# gpt-4o_0_shot_group_chat_w_recruit_w_initial_error_8_turns_immunotherapy_error_seed0.jsonl

output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_w_gene_therapy_error.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_w_CART_error.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_w_transplantation_error.csv'

# Compare across different seeds
all_results = []
for seed in [0, 1, 2]:
    json_file = f'gpt-4o_0_shot_group_chat_w_recruit_w_initial_error_8_turns_gene_therapy_error_seed{seed}'
    wrong_decision = 'gene therapy'
    # json_file = f'gpt-4o_0_shot_group_chat_w_recruit_w_initial_error_8_turns_CART_error_seed{seed}'
    # wrong_decision = 'CAR-T therapy'
    # json_file = f'gpt-4o_0_shot_group_chat_w_recruit_w_initial_error_8_turns_transplantation_error_seed{seed}'
    # wrong_decision = 'transplantation'
    text_file = []
    with open(f'/home/jaesik/MDT/multiagent/outputs/mdt/{json_file}.jsonl', 'r') as files:
        for line in files:
            text_file.append(json.loads(line))
                
    # Store results for the current seed
    seed_results = []
    yes_count = 0
    no_count = 0
    for idx in range(len(text_file)):
        decision = text_file[idx]['group_chat_w_recruit_w_initial_error']

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            seed=0,
            messages=[
                {"role": "system", "content": "You are a medical expert."},
                {"role": "user", "content": f"""[Decision]
                {decision}

                Does the decision suggest {wrong_decision}? Yes or No 
                ."""}
            ],
            response_format=ModelComparisonAnswer
        )
        
        result = response.choices[0].message.parsed
        # results.append(result)
        
        if result.correct == 'Yes':
            yes_count += 1
        elif result.correct == 'No':
            no_count += 1
            
        print(f"Case {idx + 1}: {result.correct}")

        # Add the result to the seed results
        seed_results.append(result.correct)

    print(f"\nResults for seed {seed}:")
    print(f"Yes in {yes_count} cases")  
    print(f"No in {no_count} cases")
    print(f"Yes rate: {yes_count/len(text_file)*100:.2f}%")
    print(f"No rate: {no_count/len(text_file)*100:.2f}%")

    # Append the seed results to the all_comparison_results
    all_results.append(seed_results)

# Transpose the results to match the desired CSV format
transposed_results = list(map(list, zip(*all_results)))

# Create a DataFrame and write the results to a CSV file
df = pd.DataFrame(transposed_results, columns=['methodseed0', 'methodseed1', 'methodseed2'])
df.index = [f'Case {i + 1}' for i in range(len(transposed_results))]
df.to_csv(output_file_path)