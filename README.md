# Scientific Knowledge Question Answering | 과학 지식 질의 응답 시스템 구축
## Team

| ![성명기](https://avatars.githubusercontent.com/u/104310191?v=4) | ![임동건](https://avatars.githubusercontent.com/u/125024589?v=4) |![유정수](https://avatars.githubusercontent.com/u/50096716?v=4) |
| :----------------------------: | :----------------------------: | :----------------------------: |
|            팀장          |         팀원      |          팀원     |
| [성명기](https://github.com/SUNGMYEONGGI)  | [임동건](https://github.com/LimDG1981) | [유정수](https://github.com/Dream-Forge-Studios) |

## 0. Overview

### Environment

-   AMD Ryzen Threadripper 3960X 24-Core Processor
-   NVIDIA GeForce RTX 3090
-   CUDA Version 12.2

### Requirements

pandas==2.1.4
numpy==1.23.5
tqdm==4.66.1
sentence_transformers==2.2.2
elasticsearch==8.8.0
openai==1.7.2
transformers[torch]==4.35.2

## 1. Competiton Info

### Overview

LLM의 등장 이후 여러 산업 분야에서 지식을 다루는 업무들이 점점 고도화되고 있습니다.

특히 정보를 찾기 위해 검색엔진의 입력창에 키워드를 입력하고 결과를 확인하고 원하는 정보가 없으면 다른 키워드로 다시 검색하기를 반복하는 번거로운 과정을 이제 더이상 자주 할 필요가 없어졌습니다.

이제 LLM한테 물어보면 질문의 의도까지 파악해서 필요한 내용만 잘 정리해서 알려 줍니다.

![image](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/78ef61cd-4da4-46e0-ad07-baa2d5ea980d)

그렇지만 LLM이 가진 근본적인 한계도 있습니다.

먼저, 정보라는 것은 의미나 가치가 시간에 따라 계속 변하기 때문에 모델이 이를 실시간으로 학습하기 힘들고 이 때문에 아래 예시처럼 knowledge cutoff 가 자연스럽게 발생합니다.

![image](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/4f85b2f0-cc29-499b-8b6e-7a99b0c2070c)

그리고 LLM이 알려주는 지식이 항상 사실에 기반한 것이 아닌 경우가 종종 있습니다. 특히 특정 도메인이나 문제 영역은 매우 심각한 거짓 정보들을 생성해 내곤 합니다. 아래 예시에서 추천하는 맛집들은 모두 실재하지 않는 장소들입니다.

![image](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/05fbcf71-2119-4c0e-bc87-e5e69bd01ca8)

이러한 환각 현상은 메타인지를 학습하지 않은 LLM의 근본적인 한계라 볼 수 있습니다.

모델은 학습 과정에서 정보를 압축해서 저장하기 때문에 정보의 손실이 발생할 수밖에 없고, 이 때문에 특정 입력 조건에 대해서는 사실 여부보다는 지식를 표현하는 국소적인 패턴이 더 큰 영향을 주면서 답변이 생성될 수 있기 때문입니다.

이러한 문제를 극복하기 위해서는 RAG(Retrieval Augmented Generation) 기술이 필수입니다.

RAG는 질문에 적합한 레퍼런스 추출을 위해 검색엔진을 활용하고 답변 생성을 위해 LLM(Large Language Model)을 활용합니다.

이때 LLM은 스스로 알고 있는 지식을 출력하기보다는 언어 추론 능력을 극대화하는 것에 방점을 둡니다.

이렇게 사실에 기반한 지식 정보를 토대로 질문에 답을 하고 출처 정보도 같이 줄 수 있기 때문에 사용자는 훨씬 더 안심하고 정보를 소비할 수 있게 됩니다.

![image](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/71add39c-6301-4323-8502-a8f98291f5e1)

이번 대회에서는 과학 상식을 질문하는 시나리오를 가정하고 과학 상식 문서 4200여개를 미리 검색엔진에 색인해 둡니다.

대화 메시지 또는 질문이 들어오면 과학 상식에 대한 질문 의도인지 그렇지 않은 지 판단 후에 과학 상식 질문이라면 검색엔진으로부터 적합한 문서들을 추출하고 이를 기반으로 답변을 생성합니다. 

만일 과학 상식 이외의 질문이라면 검색엔진을 활용할 필요 없이 적절한 답을 바로 생성합니다.

마지막으로, 본 프로젝트는 모델링에 중점을 둔 대회가 아니라 RAG(Retrieval Augmented Generation) 시스템의 개발에 집중하고 있습니다. 이 대회는 여러 모델과 다양한 기법, 그리고 앙상블을 활용하여 모델의 성능을 향상시키는 일반적인 모델링 대회와는 다릅니다. 대신에 검색 엔진이 올바른 문서를 색인했는지, 그리고 생성된 답변이 적절한지 직접 확인하는 것이 중요한 대회입니다.

따라서, 참가자들은 작은 규모의 토이 데이터셋(10개 미만)을 사용하여 초기 실험을 진행한 후에 전체 데이터셋에 대한 평가를 진행하는 것을 권장합니다. 실제로 RAG 시스템을 구축할 때에도 이러한 방식이 일반적으로 적용되며, 이를 통해 실험을 더욱 효율적으로 진행할 수 있습니다. 따라서 이번 대회는 2주간 진행되며, 하루에 제출할 수 있는 횟수가 5회로 제한됩니다.

자, 이제 여러분만의 RAG 시스템을 구축하러 가보실까요~?


## Evaluation Metric

사용자가 입력한 질문에 대해서 답변을 얼마나 잘 생성했는지 정량화하는 작업은 매우 고난도의 작업입니다.

어떤 질문에 대해서도 정답이 정해져 있는 것이 아니라 다양한 형태로 표현해 낼 수 있기 때문입니다.

그나마 어느 정도의 객관성을 확보하기 위해서는 다수의 사람이 직접 평가하는 방식을 사용할 수밖에 없습니다.

그렇지만 대회에서는 자동화된 평가 방법을 적용해야 하기 때문에 RAG에 대한 end-to-end 평가 대신 적합한 레퍼런스를 얼마나 잘 추출했는지에 대한 평가만 진행합니다.

이번 평가에서는 MAP(Mean Average Precision)라는 metric을 사용합니다. MAP는 질의 N개에 대한 Average Precision의 평균 값을 구하고, Average Precision은 Precision-recall curve에서 아래쪽 면적을 의미합니다.

계산 과정은 도식화하면 아래 그림과 같습니다.

![image](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/fcc26d42-8929-4271-93e1-438498ffaba2)

그런데 이번 대회에서는 MAP를 약간 변형하여 RAG 평가에 적합하도록 살짝 수정한 형태의 로직을 사용합니다.

대화 메시지가 과학 상식에 대한 질문일 수도 있고 아닐수도 있기 때문에 과학 상식 질문이 아닌 경우는 문서를 추출할 필요가 없습니다. 그래서 검색이 필요없는 ground truth 항목에 대해서는 검색 결과가 없는 경우를 1점으로 주고 그렇지 않는 경우는 0점으로 계산하게 로직을 추가했습니다.

아래 코드의 else 부분이 이에 해당하고 나머지 로직은 원래 MAP 계산 로직을 그대로 따릅니다.

```
def calc_map(gt, pred):    
    sum_average_precision = 0    
    for j in pred:        
        if gt[j["eval_id"]]:            
            hit_count = 0            
            sum_precision = 0            
            for i,docid in enumerate(j["topk"][:3]):                
                if docid in gt[j["eval_id"]]:                    
                    hit_count += 1                    
                    sum_precision += hit_count/(i+1)            
            average_precision = sum_precision / hit_count if hit_count > 0 else 0        
        else:            
            average_precision = 0 if j["topk"] else 1        
        sum_average_precision += average_precision    
    return sum_average_precision/len(pred)
```

다운로드 받은 data.tar.gz 에 포함된 eval.jsonl을 사용하여 결과물 생성하고, 이 결과물을 제출하면 리더보더에 반영됩니다.

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview

- _Explain using data_

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
