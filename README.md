# Evaluation of Multi-Agent LLMs in Multidisciplinary Team Decision-Making for Challenging Cancer Cases

이 프로젝트는 어려운 암 사례에 대한 다학제 팀(Multidisciplinary Team, MDT) 의사결정에서 다중 에이전트 LLM의 성능을 평가하는 연구입니다.

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [주요 방법론](#주요-방법론)
- [평가 방법](#평가-방법)
- [프로젝트 구조](#프로젝트-구조)

## 프로젝트 개요

이 프로젝트는 다양한 multi-agent 전략을 사용하여 의료 의사결정 과정을 시뮬레이션하고 평가합니다. 주요 목표는 다음과 같습니다:

- 단일 에이전트 vs 다중 에이전트 접근 방식의 성능 비교
- 다양한 전문가 조합이 의사결정에 미치는 영향 분석
- 잘못된 정보에 대한 시스템의 저항성 평가
- 토론 라운드 수가 성능에 미치는 영향 연구

## 설치 방법

### 1. 필수 패키지 설치

```bash
pip install openai autogen-agentchat pandas numpy scikit-learn scipy matplotlib seaborn pydantic
```

### 2. 환경 변수 설정

OpenAI API 키를 환경 변수로 설정합니다:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. 데이터 준비

데이터는 `../data/mdt_test.jsonl` 경로에 위치해야 합니다. 각 라인은 다음 형식의 JSON 객체여야 합니다:

```json
{
  "question": "환자 정보 및 질문...",
  "Answer": "정답..."
}
```

## 사용 방법

### 1. 기본 추론 실행

다양한 방법으로 추론을 실행할 수 있습니다:

```bash
# Chain of Thought (CoT) - 단일 에이전트
python inference_MDT.py --method cot --model gpt-4o --dataset mdt --seed 0

# Majority Vote with Recruitment - 전문가 채용 후 다수결
python inference_MDT.py --method majority_vote_w_recruit --model gpt-4o --dataset mdt --seed 0

# Group Chat with Recruitment - 전문가 채용 후 그룹 토론
python inference_MDT.py --method group_chat_w_recruit --model gpt-4o --turns 8 --dataset mdt --seed 0

# Simulation of Thought (SoT) - 시뮬레이션된 토론
python inference_MDT.py --method sot --model gpt-4o --dataset mdt --seed 0
```

#### 주요 파라미터

- `--method`: 사용할 방법론 (cot, sot, majority_vote, majority_vote_w_recruit, group_chat, group_chat_w_recruit 등)
- `--model`: 사용할 LLM 모델 (기본값: gpt-4o-mini)
- `--turns`: 그룹 채팅의 토론 라운드 수 (기본값: 1)
- `--dataset`: 데이터셋 이름 (기본값: mdt)
- `--num_samples`: 처리할 샘플 수 (기본값: 1)
- `--seed`: 재현성을 위한 시드 값 (기본값: 0)
- `--error`: 초기 오류 주입 유형 (gene_therapy, CART, transplantation)

### 2. 결과 파일

실행 결과는 `./outputs/{dataset}/` 디렉토리에 JSONL 형식으로 저장됩니다:

```
./outputs/mdt/gpt-4o_0_shot_group_chat_w_recruit_8_turns_seed0.jsonl
```

각 라인은 원본 데이터에 다음 필드가 추가됩니다:
- `{method}`: 해당 방법의 최종 결정
- `chat_history`: 토론 과정 (해당되는 경우)

### 3. 방법 간 성능 비교

두 가지 방법의 성능을 비교합니다:

```bash
python eval_comparion.py
```

**스크립트 수정 필요 사항:**
- `json_file1`, `json_file2`: 비교할 두 방법의 출력 파일명 설정
- `output_file_path`: 비교 결과를 저장할 CSV 파일 경로 설정
- 해당 방법의 키 이름 주석 해제 (예: `method1_decision = text_file1[idx]['cot']`)

결과는 CSV 파일로 저장되며, 각 케이스에 대해 어느 방법이 더 정답에 가까운지 평가합니다.

### 4. 잘못된 정보 저항성 평가

시스템이 의도적으로 주입된 잘못된 정보를 거부하는 능력을 평가합니다:

```bash
# 잘못된 초기 의견으로 추론 실행
python inference_MDT.py --method group_chat_w_recruit_w_initial_error --turns 8 --error gene_therapy --seed 0

# 저항성 평가
python eval_resistance.py
```

**스크립트 수정 필요 사항:**
- `json_file`: 평가할 출력 파일 설정
- `wrong_decision`: 주입된 잘못된 정보 유형 설정
- `output_file_path`: 평가 결과 저장 경로 설정

### 5. 결과 시각화 및 통계 분석

비교 결과를 시각화하고 통계적 유의성을 분석합니다:

```bash
python score.py
```

**스크립트 수정 필요 사항:**
- `input_file`: 분석할 비교 결과 CSV 파일 경로 설정
- `output_file`: 시각화 결과를 저장할 PNG 파일 경로 설정

결과:
- 누적 막대 그래프 (Method 1 vs Method 2 vs Tie)
- Wilcoxon Signed-Rank Test p-value
- 각 시드별 일치도(concordance) 분석
- 전체 평균 승률

## 주요 방법론

### 1. Chain of Thought (CoT)
단일 전문가가 단계적으로 추론하는 방법입니다.

```python
# 사용 예시
python inference_MDT.py --method cot --seed 0
```

### 2. Simulation of Thought (SoT)
단일 에이전트가 여러 전문가 간의 토론을 시뮬레이션합니다.

```python
# 사용 예시
python inference_MDT.py --method sot --seed 0
```

### 3. Majority Vote
4명의 독립적인 전문가가 각자 의견을 제시하고, 요약자가 다수 의견을 종합합니다.

```python
# 사용 예시
python inference_MDT.py --method majority_vote --seed 0
```

### 4. Majority Vote with Recruitment
먼저 채용 관리자가 케이스에 적합한 전문가를 선정한 후, 각 전문가가 독립적으로 의견을 제시하고 다수 의견을 종합합니다.

```python
# 사용 예시
python inference_MDT.py --method majority_vote_w_recruit --seed 0
```

### 5. Group Chat
4명의 전문가가 그룹 채팅을 통해 토론합니다.

```python
# 사용 예시
python inference_MDT.py --method group_chat --turns 8 --seed 0
```

### 6. Group Chat with Recruitment
채용 관리자가 케이스에 적합한 전문가를 선정한 후, 선정된 전문가들이 그룹 채팅을 통해 토론합니다.

```python
# 사용 예시
python inference_MDT.py --method group_chat_w_recruit --turns 8 --seed 0
```

### 7. Hybrid Approach
Majority Vote with Recruitment와 Group Chat을 결합한 방법입니다. 먼저 각 전문가가 독립적으로 의견을 제시한 후, 그룹 토론을 진행합니다.

```python
# 사용 예시
python inference_MDT.py --method majority_vote_w_recruit_and_group_chat --turns 4 --seed 0
```

### 8. Group Chat with Initial Error
의도적으로 잘못된 초기 의견을 주입하여 시스템의 저항성을 테스트합니다.

```python
# 사용 예시
python inference_MDT.py --method group_chat_w_recruit_w_initial_error --turns 8 --error gene_therapy --seed 0
```

## 평가 방법

### 1. 페어와이즈 비교 (Pairwise Comparison)

`eval_comparion.py`는 GPT-4o를 평가자로 사용하여 두 방법의 출력을 비교합니다:

- 각 케이스에 대해 두 방법의 결정을 정답과 비교
- "Method 1", "Method 2", 또는 "Tie"로 판정
- 여러 시드에 걸쳐 재현성 평가

### 2. 저항성 평가 (Resistance Evaluation)

`eval_resistance.py`는 잘못된 정보가 주입되었을 때 시스템의 저항성을 평가합니다:

- 의도적으로 잘못된 초기 의견 주입 (예: gene therapy, CAR-T therapy, transplantation)
- 최종 결정이 잘못된 정보를 포함하는지 평가
- "Yes" 또는 "No"로 판정

### 3. 통계 분석

`score.py`는 다음을 수행합니다:

- **시각화**: 누적 막대 그래프로 케이스별 승패 표시
- **통계 검정**: Wilcoxon Signed-Rank Test로 유의성 평가
- **재현성 분석**: 평가 시드 간 일치도(concordance) 계산
- **승률 계산**: 각 방법의 전체 승률 및 평균 계산

## 프로젝트 구조

```
eval-multi-agent-MDT/
├── inference_MDT.py          # 메인 추론 실행 스크립트
├── method.py                  # 다양한 multi-agent 방법론 구현
├── eval_comparion.py         # 방법 간 성능 비교 스크립트
├── eval_resistance.py        # 저항성 평가 스크립트
├── score.py                   # 결과 시각화 및 통계 분석
├── utils.py                   # 데이터 로딩 유틸리티
└── outputs/                   # 결과 파일 저장 디렉토리
    └── mdt/                   # MDT 데이터셋 결과
        ├── *.jsonl           # 추론 결과
        ├── *.csv             # 비교 결과
        └── figs/             # 시각화 결과
            └── *.png
```

### 주요 파일 설명

#### inference_MDT.py
- 다양한 방법론으로 추론을 실행하는 메인 스크립트
- 명령줄 인자로 방법, 모델, 턴 수, 시드 등을 설정
- 결과를 JSONL 형식으로 저장

#### method.py
- 9가지 multi-agent 방법론 구현
- AutoGen 라이브러리를 사용한 에이전트 구성
- 각 방법은 (chat_history, decision) 튜플 반환

#### eval_comparion.py
- 두 방법의 성능을 페어와이즈로 비교
- GPT-4o를 평가자로 사용
- 여러 시드에 걸쳐 재현성 검증

#### eval_resistance.py
- 잘못된 정보에 대한 저항성 평가
- 의도적으로 주입된 오류를 시스템이 거부하는지 확인

#### score.py
- 비교 결과 시각화
- Wilcoxon 검정으로 통계적 유의성 평가
- 평가자 간 일치도 분석

#### utils.py
- JSONL 형식의 데이터 로딩 함수

## 워크플로우 예시

다음은 전체 평가 파이프라인을 실행하는 예시입니다:

```bash
# 1. 여러 시드로 두 가지 방법 실행
for seed in 0 1 2; do
  python inference_MDT.py --method majority_vote_w_recruit --turns 1 --seed $seed
  python inference_MDT.py --method group_chat_w_recruit --turns 8 --seed $seed
done

# 2. eval_comparion.py 수정
# - json_file1 = 'gpt-4o_0_shot_majority_vote_w_recruit_1_turns_seed{seed}'
# - json_file2 = 'gpt-4o_0_shot_group_chat_w_recruit_8_turns_seed{seed}'
# - output_file_path 설정

# 3. 성능 비교 실행
python eval_comparion.py

# 4. score.py 수정
# - input_file 및 output_file 경로 설정

# 5. 통계 분석 및 시각화
python score.py

# 6. (선택) 저항성 평가
for seed in 0 1 2; do
  python inference_MDT.py --method group_chat_w_recruit_w_initial_error \
    --turns 8 --error gene_therapy --seed $seed
done

# 7. eval_resistance.py 수정 후 실행
python eval_resistance.py
```

## 주의 사항

1. **API 비용**: OpenAI API를 사용하므로 많은 샘플을 처리할 때 비용이 발생할 수 있습니다.

2. **Rate Limiting**: `inference_MDT.py`에는 샘플 간 1초 대기 시간이 포함되어 있습니다.

3. **스크립트 수정**: `eval_comparion.py`, `eval_resistance.py`, `score.py`는 실행 전에 파일 경로와 방법 이름을 수정해야 합니다.

4. **데이터 경로**: 데이터는 `../data/mdt_test.jsonl` 경로에 위치해야 합니다.

5. **출력 디렉토리**: `./outputs/mdt/` 디렉토리가 없으면 자동으로 생성되지 않으므로 수동으로 생성해야 할 수 있습니다.

## 라이선스

이 프로젝트의 최종 버전 코드는 곧 업로드될 예정입니다.

## 인용

이 코드를 연구에 사용하는 경우, 적절한 인용을 부탁드립니다.
