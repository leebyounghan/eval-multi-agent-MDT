# Evaluation of Multi-Agent LLMs in Multidisciplinary Team Decision-Making for Challenging Cancer Cases

This project evaluates the performance of multi-agent Large Language Models (LLMs) in Multidisciplinary Team (MDT) decision-making for challenging cancer cases.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Key Methodologies](#key-methodologies)
- [Evaluation Methods](#evaluation-methods)
- [Project Structure](#project-structure)

## Project Overview

This project simulates and evaluates medical decision-making processes using various multi-agent strategies. The main objectives include:

- Comparing single-agent vs. multi-agent approaches
- Analyzing the impact of diverse expert combinations on decision-making
- Evaluating system resistance to misinformation
- Investigating the effect of discussion rounds on performance

## Installation

### 1. Install Required Packages

```bash
pip install openai autogen-agentchat pandas numpy scikit-learn scipy matplotlib seaborn pydantic
```

### 2. Set Environment Variables

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Prepare Data

Data should be located at `../data/mdt_test.jsonl`. Each line should be a JSON object in the following format:

```json
{
  "question": "Patient information and question...",
  "Answer": "Ground truth answer..."
}
```

## Usage

### 1. Running Inference

You can run inference using various methods:

```bash
# Chain of Thought (CoT) - Single agent
python inference_MDT.py --method cot --model gpt-4o --dataset mdt --seed 0

# Majority Vote with Recruitment - Expert recruitment followed by majority voting
python inference_MDT.py --method majority_vote_w_recruit --model gpt-4o --dataset mdt --seed 0

# Group Chat with Recruitment - Expert recruitment followed by group discussion
python inference_MDT.py --method group_chat_w_recruit --model gpt-4o --turns 8 --dataset mdt --seed 0

# Simulation of Thought (SoT) - Simulated discussion
python inference_MDT.py --method sot --model gpt-4o --dataset mdt --seed 0
```

#### Key Parameters

- `--method`: Method to use (cot, sot, majority_vote, majority_vote_w_recruit, group_chat, group_chat_w_recruit, etc.)
- `--model`: LLM model to use (default: gpt-4o-mini)
- `--turns`: Number of discussion rounds for group chat (default: 1)
- `--dataset`: Dataset name (default: mdt)
- `--num_samples`: Number of samples to process (default: 1)
- `--seed`: Seed value for reproducibility (default: 0)
- `--error`: Type of initial error injection (gene_therapy, CART, transplantation)

### 2. Output Files

Inference results are saved in JSONL format in the `./outputs/{dataset}/` directory:

```
./outputs/mdt/gpt-4o_0_shot_group_chat_w_recruit_8_turns_seed0.jsonl
```

Each line contains the original data with the following additional fields:
- `{method}`: Final decision from the method
- `chat_history`: Discussion process (if applicable)

### 3. Method Comparison

Compare the performance of two methods:

```bash
python eval_comparion.py
```

**Required Script Modifications:**
- Set `json_file1`, `json_file2`: Output filenames of the two methods to compare
- Set `output_file_path`: CSV file path to save comparison results
- Uncomment the appropriate method key name (e.g., `method1_decision = text_file1[idx]['cot']`)

Results are saved as a CSV file, evaluating which method is closer to the ground truth for each case.

### 4. Resistance Evaluation

Evaluate the system's ability to reject deliberately injected misinformation:

```bash
# Run inference with incorrect initial opinion
python inference_MDT.py --method group_chat_w_recruit_w_initial_error --turns 8 --error gene_therapy --seed 0

# Evaluate resistance
python eval_resistance.py
```

**Required Script Modifications:**
- Set `json_file`: Output file to evaluate
- Set `wrong_decision`: Type of injected misinformation
- Set `output_file_path`: Path to save evaluation results

### 5. Visualization and Statistical Analysis

Visualize comparison results and analyze statistical significance:

```bash
python score.py
```

**Required Script Modifications:**
- Set `input_file`: Path to comparison results CSV file to analyze
- Set `output_file`: Path to save visualization results (PNG)

Results include:
- Stacked bar chart (Method 1 vs Method 2 vs Tie)
- Wilcoxon Signed-Rank Test p-value
- Concordance analysis across seeds
- Overall win rates

## Key Methodologies

### 1. Chain of Thought (CoT)
A single expert reasons through the problem step by step.

```bash
# Usage example
python inference_MDT.py --method cot --seed 0
```

### 2. Simulation of Thought (SoT)
A single agent simulates a discussion between multiple experts.

```bash
# Usage example
python inference_MDT.py --method sot --seed 0
```

### 3. Majority Vote
Four independent experts provide opinions, and a summarizer synthesizes the majority opinion.

```bash
# Usage example
python inference_MDT.py --method majority_vote --seed 0
```

### 4. Majority Vote with Recruitment
A hiring manager first selects appropriate experts for the case, then each expert independently provides an opinion, followed by majority opinion synthesis.

```bash
# Usage example
python inference_MDT.py --method majority_vote_w_recruit --seed 0
```

### 5. Group Chat
Four experts engage in group discussion.

```bash
# Usage example
python inference_MDT.py --method group_chat --turns 8 --seed 0
```

### 6. Group Chat with Recruitment
A hiring manager selects appropriate experts for the case, who then engage in group discussion.

```bash
# Usage example
python inference_MDT.py --method group_chat_w_recruit --turns 8 --seed 0
```

### 7. Hybrid Approach
Combines Majority Vote with Recruitment and Group Chat. Experts first independently provide opinions, then engage in group discussion.

```bash
# Usage example
python inference_MDT.py --method majority_vote_w_recruit_and_group_chat --turns 4 --seed 0
```

### 8. Group Chat with Initial Error
Deliberately injects incorrect initial opinions to test system resistance.

```bash
# Usage example
python inference_MDT.py --method group_chat_w_recruit_w_initial_error --turns 8 --error gene_therapy --seed 0
```

## Evaluation Methods

### 1. Pairwise Comparison

`eval_comparion.py` uses GPT-4o as an evaluator to compare outputs from two methods:

- Compares decisions from both methods against the ground truth for each case
- Judges as "Method 1", "Method 2", or "Tie"
- Evaluates reproducibility across multiple seeds

### 2. Resistance Evaluation

`eval_resistance.py` evaluates system resistance when misinformation is injected:

- Deliberately injects incorrect initial opinions (e.g., gene therapy, CAR-T therapy, transplantation)
- Evaluates whether the final decision contains the misinformation
- Judges as "Yes" or "No"

### 3. Statistical Analysis

`score.py` performs the following:

- **Visualization**: Displays case-by-case wins/losses using stacked bar charts
- **Statistical Testing**: Evaluates significance using Wilcoxon Signed-Rank Test
- **Reproducibility Analysis**: Calculates concordance between evaluation seeds
- **Win Rate Calculation**: Computes overall win rates and averages for each method

## Project Structure

```
eval-multi-agent-MDT/
├── inference_MDT.py          # Main inference execution script
├── method.py                  # Implementation of various multi-agent methods
├── eval_comparion.py         # Method comparison script
├── eval_resistance.py        # Resistance evaluation script
├── score.py                   # Visualization and statistical analysis
├── utils.py                   # Data loading utilities
└── outputs/                   # Output files directory
    └── mdt/                   # MDT dataset results
        ├── *.jsonl           # Inference results
        ├── *.csv             # Comparison results
        └── figs/             # Visualization results
            └── *.png
```

### Key File Descriptions

#### inference_MDT.py
- Main script for running inference with various methodologies
- Configure method, model, turns, seed, etc. via command-line arguments
- Saves results in JSONL format

#### method.py
- Implementation of 9 multi-agent methodologies
- Agent configuration using the AutoGen library
- Each method returns a (chat_history, decision) tuple

#### eval_comparion.py
- Pairwise comparison of two methods' performance
- Uses GPT-4o as evaluator
- Validates reproducibility across multiple seeds

#### eval_resistance.py
- Evaluates resistance to misinformation
- Checks whether the system rejects deliberately injected errors

#### score.py
- Visualizes comparison results
- Evaluates statistical significance using Wilcoxon test
- Analyzes inter-evaluator concordance

#### utils.py
- Functions for loading data in JSONL format

## Workflow Example

Here's an example of running the complete evaluation pipeline:

```bash
# 1. Run two methods with multiple seeds
for seed in 0 1 2; do
  python inference_MDT.py --method majority_vote_w_recruit --turns 1 --seed $seed
  python inference_MDT.py --method group_chat_w_recruit --turns 8 --seed $seed
done

# 2. Modify eval_comparion.py
# - Set json_file1 = 'gpt-4o_0_shot_majority_vote_w_recruit_1_turns_seed{seed}'
# - Set json_file2 = 'gpt-4o_0_shot_group_chat_w_recruit_8_turns_seed{seed}'
# - Set output_file_path

# 3. Run performance comparison
python eval_comparion.py

# 4. Modify score.py
# - Set input_file and output_file paths

# 5. Run statistical analysis and visualization
python score.py

# 6. (Optional) Resistance evaluation
for seed in 0 1 2; do
  python inference_MDT.py --method group_chat_w_recruit_w_initial_error \
    --turns 8 --error gene_therapy --seed $seed
done

# 7. Modify and run eval_resistance.py
python eval_resistance.py
```

## Important Notes

1. **API Costs**: This project uses the OpenAI API, which may incur costs when processing many samples.

2. **Rate Limiting**: `inference_MDT.py` includes a 1-second wait time between samples.

3. **Script Modification**: `eval_comparion.py`, `eval_resistance.py`, and `score.py` require file path and method name modifications before execution.

4. **Data Path**: Data should be located at `../data/mdt_test.jsonl`.

5. **Output Directory**: The `./outputs/mdt/` directory may need to be created manually if it doesn't exist.

## License

The final version of this code will be uploaded soon.

## Citation

If you use this code in your research, please cite appropriately.
