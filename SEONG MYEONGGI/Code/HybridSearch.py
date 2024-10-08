import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import traceback
import ssl

load_dotenv()

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    return model.encode(sentences)

# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings

# 새로운 index 생성
def create_es_index(index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)

# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)

# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)

# 역색인을 이용한 검색
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str,
                "operator": "and"
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")

# Vector 유사도를 이용한 검색
def dense_retrieve(query_str, size):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]

    # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", knn=knn)

# 하이브리드 검색 함수
def hybrid_retrieve(query_str, size, sparse_weight=0.5):
    # sparse 검색 수행
    sparse_results = sparse_retrieve(query_str, size)
    
    # dense 검색 수행
    dense_results = dense_retrieve(query_str, size)
    
    # 결과 병합을 위한 딕셔너리 생성
    merged_results = {}
    
    # sparse 결과 처리
    for hit in sparse_results['hits']['hits']:
        doc_id = hit['_id']
        sparse_score = hit['_score']
        merged_results[doc_id] = {
            'sparse_score': sparse_score,
            'dense_score': 0,
            '_source': hit['_source']
        }
    
    # dense 결과 처리
    for hit in dense_results['hits']['hits']:
        doc_id = hit['_id']
        dense_score = hit['_score']
        if doc_id in merged_results:
            merged_results[doc_id]['dense_score'] = dense_score
        else:
            merged_results[doc_id] = {
                'sparse_score': 0,
                'dense_score': dense_score,
                '_source': hit['_source']
            }
    
    # 최종 점수 계산 및 정렬
    for doc_id, result in merged_results.items():
        result['final_score'] = (sparse_weight * result['sparse_score'] + 
                                 (1 - sparse_weight) * result['dense_score'])
    
    sorted_results = sorted(merged_results.items(), 
                            key=lambda x: x[1]['final_score'], 
                            reverse=True)
    
    # 상위 size개 결과 반환
    top_results = sorted_results[:size]
    
    # Elasticsearch 결과 형식에 맞게 변환
    final_results = {
        'hits': {
            'hits': [
                {
                    '_id': doc_id,
                    '_score': result['final_score'],
                    '_source': result['_source']
                } for doc_id, result in top_results
            ]
        }
    }
    
    return final_results

es_username = "elastic"
es_password = "JMsFtIda3zAACqcQO=ol"

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt")

# Elasticsearch client 정보 확인
print(es.info())

# 색인을 위한 setting 설정
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

# 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "l2_norm"
        }
    }
}

# settings, mappings 설정된 내용으로 'test' 인덱스 생성
create_es_index("test", settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성
index_docs = []
with open('/home/InformationRetrieval/data/documents.jsonl') as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)
                
# 생성한 임베딩을 색인할 필드로 추가
for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

# 'test' 인덱스에 대량 문서 추가
ret = bulk_add("test", index_docs)

# 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
print(ret)

# OpenAI API 키를 환경변수에 설정
OPENAI_APIKEY = os.getenv("OPENAI-APIKEY")
client = OpenAI(api_key=OPENAI_APIKEY)

llm_model = "gpt-4o"

# RAG 구현에 필요한 Question Answering을 위한 LLM 프롬프트
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
persona_function_calling = """
## Role: 과학 상식 전문가

## Instruction
- 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 search api를 호출할 수 있어야 한다.
- 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성한다.
"""

# Function calling에 사용할 함수 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "관련 문서를 검색합니다",
            "parameters": {
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "사용자 메시지 기록에서 추출한, 검색에 적합한 최종 쿼리입니다."
                    }
                },
                "required": ["standalone_query"],
                "type": "object"
            }
        }
    },
]

# LLM과 검색엔진을 활용한 RAG 구현
def answer_question(messages):
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception as e:
        traceback.print_exc()
        return response

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")

        # 하이브리드 검색 사용
        search_result = hybrid_retrieve(standalone_query, 3)

        response["standalone_query"] = standalone_query
        retrieved_context = []
        for i, rst in enumerate(search_result['hits']['hits']):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

        content = json.dumps(retrieved_context)
        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                    model=llm_model,
                    messages=msg,
                    temperature=0,
                    seed=1,
                    timeout=30
                )
        except Exception as e:
            traceback.print_exc()
            return response
        response["answer"] = qaresult.choices[0].message.content

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message.content

    return response

# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 파일 경로 설정
file_root = '/home/InformationRetrieval/data/'
output_file = "/home/InformationRetrieval/result/ko-sroberta-multitask&gpt-4o.csv"

# 평가 데이터에 대해서 결과 생성
eval_rag(os.path.join(file_root, "eval.jsonl"), output_file)

print(f"결과가 {output_file}에 저장되었습니다.")
