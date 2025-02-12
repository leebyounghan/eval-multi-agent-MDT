import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/cot_vs_majority_vote_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/cot_vs_majority_vote_w_recruit.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/cot_vs_mdagents.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/cot_vs_mdagents.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/cot_vs_group_chat_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/cot_vs_group_chat_w_recruit.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/majority_vote_w_recruit_vs_mdagents.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/majority_vote_w_recruit_vs_mdagents.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/majority_vote_w_recruit_vs_group_chat_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/majority_vote_w_recruit_vs_group_chat_w_recruit.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/mdagents_vs_group_chat_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/mdagents_vs_group_chat_w_recruit.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/cot_vs_sot.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/cot_vs_sot.png'



# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_vs_group_chat_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/group_chat_vs_group_chat_w_recruit.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_w_restricted_role_vs_group_chat_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/group_chat_w_restricted_role_vs_group_chat_w_recruit.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/majority_vote_vs_majority_vote_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/majority_vote_vs_majority_vote_w_recruit.png'




# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_3_w_recruit_vs_group_chat_8_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/group_chat_3_w_recruit_vs_group_chat_8_w_recruit.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_6_w_recruit_vs_group_chat_8_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/group_chat_6_w_recruit_vs_group_chat_8_w_recruit.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_12_w_recruit_vs_group_chat_8_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/group_chat_12_w_recruit_vs_group_chat_8_w_recruit.png'


input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/majority_vote_w_recruit_vs_majority_vote_w_recruit_and_group_chat_4.csv'
output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/majority_vote_w_recruit_vs_majority_vote_w_recruit_and_group_chat_4.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/majority_vote_w_recruit_vs_majority_vote_w_recruit_and_group_chat_8.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/majority_vote_w_recruit_vs_majority_vote_w_recruit_and_group_chat_8.png'


# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/majority_vote_w_recruit_and_group_chat_vs_group_chat_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/majority_vote_w_recruit_and_group_chat_vs_group_chat_w_recruit.png'





# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_w_recruit_w_surgery_error_vs_group_chat_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/group_chat_w_recruit_w_surgery_error_vs_group_chat_w_recruit.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_w_recruit_w_nccn_error_vs_group_chat_w_recruit.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/group_chat_w_recruit_w_nccn_error_vs_group_chat_w_recruit.png'


# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/cot_vs_cot_w_RAG.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/cot_vs_cot_w_RAG.png'

# input_file = '/home/jaesik/MDT/multiagent/outputs/mdt/group_chat_w_recruit_vs_group_chat_w_recruit_w_RAG.csv'
# output_file = '/home/jaesik/MDT/multiagent/outputs/mdt/figs/group_chat_w_recruit_vs_group_chat_w_recruit_w_RAG.png'


df = pd.read_csv(input_file)
df = df.drop(index=8)

# 각 문항별 Method 1, Method 2, Both wrong 계산
method1_wins = (df == 'Method 1').sum(axis=1)
method2_wins = (df == 'Method 2').sum(axis=1)
tie = (df == 'Tie').sum(axis=1) 

# 데이터프레임으로 변환
results = pd.DataFrame({
    'Question Index': range(1, len(df)+1),  # 문항 인덱스
    'Method 1 Wins': method1_wins,     # Method 1이 이긴 개수
    'Method 2 Wins': method2_wins,     # Method 2가 이긴 개수
    'Tie': tie           # tie 개수
})
# 데이터를 long format으로 변환 (seaborn에서 사용하기 위해)
results_long = results.melt(id_vars='Question Index', var_name='Method', value_name='Wins')

# 누적 막대 그래프를 위해 pivot 테이블 생성
pivot_results = results_long.pivot(index='Question Index', columns='Method', values='Wins')
pivot_results = pivot_results[['Method 2 Wins', 'Tie', 'Method 1 Wins']]

# 그래프 그리기
plt.figure(figsize=(8, 10))  # 세로로 길게 설정
sns.set(style="white")  # 그리드 제거
ax = pivot_results.plot(kind='barh', stacked=True, color=['blue', 'lightgray', 'orange'], figsize=(8, 10), width=0.8)

# 그래프 설정
plt.xlabel('Count')
plt.ylabel('Question Index')
plt.yticks(rotation=0)  # y축 레이블 회전
plt.legend(title='Method')

# y축 간격 조정
ax.margins(y=0)  # y축 여백을 0으로 설정

# 그래프 출력
plt.tight_layout()
plt.savefig(output_file)


# Wilcoxon Signed-Rank Test 
differences = [m1 - m2 for m1, m2 in zip(method1_wins, method2_wins)]
stat, p_value = wilcoxon(differences)
print(f"P-value: {p_value:.7f}")



labels = ['Method 1', 'Method 2', 'Tie']
overall_rates = {'Method 1': [], 'Method 2': [], 'Tie': []}

# Analyze for each method seed
for seed in range(3):
    print(f"\n=== Method Seed {seed} ===")
    
    # Get columns for current seed
    col1 = f'methodseed{seed}_evalseed0'
    col2 = f'methodseed{seed}_evalseed1'
    
    # Calculate confusion matrix and concordance
    cm = confusion_matrix(df[col1], df[col2], labels=labels)
    concordance = (cm.diagonal().sum() / cm.sum()) * 100
    
    print("\nConfusion Matrix between eval seeds:")
    print(pd.DataFrame(cm, index=labels, columns=labels))
    print(f"Concordance Score: {concordance:.2f}%")
    
    # Calculate win rates for both eval seeds
    rates = {}
    for col in [col1, col2]:
        for label in labels:
            rate = (df[col] == label).mean() * 100
            rates[f"{col}_{label}"] = rate
    
    # Calculate and store average rates across eval seeds
    for label in labels:
        avg_rate = (rates[f"{col1}_{label}"] + rates[f"{col2}_{label}"]) / 2
        overall_rates[label].append(avg_rate)
        
    print("\nWin Rates:")
    for label in labels:
        print(f"{label}: {overall_rates[label][-1]:.2f}%")

# Calculate and display overall averages
print("\n=== Overall Average Win Rates ===")
for label in labels:
    avg = np.mean(overall_rates[label])
    print(f"{label}: {avg:.2f}%")