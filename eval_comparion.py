import json
from openai import OpenAI
from typing import Literal
from pydantic import BaseModel
import pandas as pd

class ModelComparisonAnswer(BaseModel):
    correct:Literal['Method 1', 'Method 2', 'Tie']

client = OpenAI()

# Prepare a list to store all results for CSV
all_comparison_results = []

# single-agent vs multi-agent
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/cot_vs_majority_vote_w_recruit.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/cot_vs_mdagents.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/cot_vs_group_chat_w_recruit.csv'

# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/majority_vote_w_recruit_vs_mdagents.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/majority_vote_w_recruit_vs_group_chat_w_recruit.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/mdagents_vs_group_chat_w_recruit.csv'


# why? discussion is impactful? not really
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/cot_vs_sot.csv'


# various role is important?
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_vs_group_chat_w_recruit.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_w_restricted_role_vs_group_chat_w_recruit.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/majority_vote_vs_majority_vote_w_recruit.csv'


# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_3_w_recruit_vs_group_chat_8_w_recruit.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_6_w_recruit_vs_group_chat_8_w_recruit.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_12_w_recruit_vs_group_chat_8_w_recruit.csv'

# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_w_recruit_w_initial_error_1_vs_group_chat_w_recruit.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_w_recruit_w_nccn_error_vs_group_chat_w_recruit.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_w_recruit_w_immunotherapy_error_vs_group_chat_w_recruit.csv'

output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/majority_vote_w_recruit_vs_majority_vote_w_recruit_and_group_chat_4.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/majority_vote_w_recruit_vs_majority_vote_w_recruit_and_group_chat_8.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/majority_vote_w_recruit_and_group_chat_vs_group_chat_w_recruit.csv'

# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/cot_vs_cot_w_RAG.csv'
# output_file_path = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_w_recruit_vs_group_chat_w_recruit_w_RAG.csv'



# Compare across different seeds
for seed in [0, 1, 2]:
    for eval_seed in [0, 1]: 
        # json_file1 = f'gpt-4o_0_shot_cot_1_turns_seed{seed}'
        json_file1 = f'gpt-4o_0_shot_majority_vote_w_recruit_1_turns_seed{seed}'
        # json_file1 = f'gpt-4o_0_shot_mdagents_seed{seed}'
        # json_file1 = f'gpt-4o_0_shot_sot_1_turns_seed{seed}'

        # json_file1 = f'gpt-4o_0_shot_group_chat_8_turns_seed{seed}'
        # json_file1 = f'gpt-4o_0_shot_group_chat_w_restricted_role_8_turns_seed{seed}'
        # json_file1 = f'gpt-4o_0_shot_majority_vote_1_turns_seed{seed}'
        # json_file1 = f'gpt-4o_0_shot_group_chat_w_recruit_12_turns_seed{seed}'
        # json_file1 = f'gpt-4o_0_shot_group_chat_w_recruit_8_turns_seed{seed}'
        # json_file1 = f'gpt-4o_0_shot_majority_vote_w_recruit_and_group_chat_4_turns_seed{seed}'
        # json_file1 = f'gpt-4o_0_shot_majority_vote_w_recruit_and_group_chat_8_turns_seed{seed}'
        # json_file1 = f'gpt-4o_0_shot_group_chat_w_recruit_w_initial_error_8_turns_nccn_error_seed{seed}'

        # json_file2 = f'gpt-4o_0_shot_majority_vote_w_recruit_1_turns_seed{seed}'
        # json_file2 = f'gpt-4o_0_shot_mdagents_seed{seed}'
        # json_file2 = f'gpt-4o_0_shot_group_chat_w_recruit_8_turns_seed{seed}'
        # json_file2 = f'gpt-4o_0_shot_sot_1_turns_seed{seed}'
        # json_file1 = f'gpt-4o_0_shot_majority_vote_1_turns_seed{seed}'
        # json_file2 = f'gpt-4o_0_shot_group_chat_w_recruit_12_turns_seed{seed}'
        json_file2 = f'gpt-4o_0_shot_majority_vote_w_recruit_and_group_chat_4_turns_seed{seed}'
        # json_file2 = f'gpt-4o_0_shot_cot_1_turns_wRAG_seed{seed}'
        # json_file2 = f'gpt-4o_0_shot_group_chat_w_recruit_8_turns_wRAG_seed{seed}'

        print(f"\nComparing seed {seed} and {eval_seed}:")
        print(f"Method 1: {json_file1}")
        print(f"Method 2: {json_file2}")

        text_file1 = []
        with open(f'/home/jaesik/MDT/multiagent/outputs/mdt/{json_file1}.jsonl', 'r') as files:
            for line in files:
                text_file1.append(json.loads(line))
                
        text_file2 = []
        with open(f'/home/jaesik/MDT/multiagent/outputs/mdt/{json_file2}.jsonl', 'r') as files:
            for line in files:
                text_file2.append(json.loads(line))

        comparison_results = []
        method1_count = 0
        method2_count = 0
        tie_count = 0

        # Store results for the current seed
        seed_results = []

        for idx in range(len(text_file1)):
            # method1_decision = text_file1[idx]['cot']
            # method1_decision = text_file1[idx]['sot']
            # method1_decision = text_file1[idx]['majority_vote'] 
            method1_decision = text_file1[idx]['majority_vote_w_recruit']
            # method1_decision = text_file1[idx]['mdagent'] 
            # method1_decision = text_file1[idx]['group_chat']
            # method1_decision = text_file1[idx]['group_chat_w_restricted_role']
            # method1_decision = text_file1[idx]['group_chat_w_recruit']
            # method1_decision = text_file1[idx]['majority_vote_w_recruit_and_group_chat']

            # method2_decision = text_file2[idx]['majority_vote_w_recruit'] 
            # method2_decision = text_file2[idx]['mdagent'] 
            # method2_decision = text_file2[idx]['sot'] 
            # method2_decision = text_file2[idx]['group_chat_w_recruit'] 
            method2_decision = text_file2[idx]['majority_vote_w_recruit_and_group_chat'] 
            # method2_decision = text_file2[idx]['cot']
            # method2_decision = text_file2[idx]['group_chat'] 


            answer = text_file1[idx]['Answer']
            
            response = client.beta.chat.completions.parse(
                model="gpt-4o",
                seed=eval_seed,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who compares two different decisions and determines which one is closer to the ground truth answer."},
                    {"role": "user", "content": f"""Please compare these two decisions with the ground truth answer and determine which one is closer:

        [Method 1 Decision]
        {method1_decision}

        [Method 2 Decision]
        {method2_decision}

        [Ground Truth Answer]
        {answer}

        Which decision (Method 1 or Method 2) is closer to the ground truth answer based on their rationale?
        If both decisions are **the completely same**, answer with 'Tie'. Otherwise, answer with either 'Method 1' or 'Method 2'."""}
                ],
                response_format=ModelComparisonAnswer
            )
            
            comparison_result = response.choices[0].message.parsed
            comparison_results.append(comparison_result)
            
            if comparison_result.correct == 'Method 1':
                method1_count += 1
            elif comparison_result.correct == 'Method 2':
                method2_count += 1
            else:
                tie_count += 1
                
            print(f"Case {idx + 1}: {comparison_result.correct}")

            # Add the result to the seed results
            seed_results.append(comparison_result.correct)

        print(f"\nResults for seed {seed}:")
        print(f"Method 1 was better in {method1_count} cases")  
        print(f"Method 2 was better in {method2_count} cases")
        print(f"Tie in {tie_count} cases")
        print(f"Method 1 win rate: {method1_count/len(text_file1)*100:.2f}%")
        print(f"Method 2 win rate: {method2_count/len(text_file1)*100:.2f}%")
        print(f"Tie rate: {tie_count/len(text_file1)*100:.2f}%")

        # Append the seed results to the all_comparison_results
        all_comparison_results.append(seed_results)

# Transpose the results to match the desired CSV format
transposed_results = list(map(list, zip(*all_comparison_results)))

# Create a DataFrame and write the results to a CSV file
df = pd.DataFrame(transposed_results, columns=['methodseed0_evalseed0', 'methodseed0_evalseed1', 'methodseed1_evalseed0', 'methodseed1_evalseed1', 'methodseed2_evalseed0', 'methodseed2_evalseed1'])
df.index = [f'Case {i + 1}' for i in range(len(transposed_results))]
df.to_csv(output_file_path)