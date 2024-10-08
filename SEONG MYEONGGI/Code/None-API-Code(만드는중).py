import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
# model = SentenceTransformer("jhgan/ko-sroberta-multitask")
model = SentenceTransformer("dragonkue/BGE-m3-ko")


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


# 역색인을 이용한 검색 (기존과 동일)
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": query_str
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")


# Vector 유사도를 이용한 검색 (기존과 동일)
def dense_retrieve(query_str, size):
    query_embedding = get_embedding([query_str])[0]
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }
    return es.search(index="test", knn=knn)


# 새로 추가된 하이브리드 검색 함수
def hybrid_retrieve(query_str, size, alpha=0.5):
    sparse_results = sparse_retrieve(query_str, size)
    dense_results = dense_retrieve(query_str, size)
    
    combined_results = {}
    max_sparse_score = max(hit['_score'] for hit in sparse_results['hits']['hits']) if sparse_results['hits']['hits'] else 1
    max_dense_score = max(hit['_score'] for hit in dense_results['hits']['hits']) if dense_results['hits']['hits'] else 1
    
    for hit in sparse_results['hits']['hits']:
        doc_id = hit['_id']
        normalized_sparse_score = hit['_score'] / max_sparse_score
        combined_results[doc_id] = {'document': hit['_source'], 'score': alpha * normalized_sparse_score}
    
    for hit in dense_results['hits']['hits']:
        doc_id = hit['_id']
        normalized_dense_score = hit['_score'] / max_dense_score
        if doc_id in combined_results:
            combined_results[doc_id]['score'] += (1 - alpha) * normalized_dense_score
        else:
            combined_results[doc_id] = {'document': hit['_source'], 'score': (1 - alpha) * normalized_dense_score}
    
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    hybrid_results = {
        'hits': {
            'total': {'value': len(sorted_results)},
            'hits': [
                {
                    '_id': doc_id,
                    '_score': info['score'],
                    '_source': info['document']
                } for doc_id, info in sorted_results[:size]
            ]
        }
    }
    
    return hybrid_results


es_username = "elastic"
es_password = "Ig041gMcR5c*h=zfguAv"

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.15.2/config/certs/http_ca.crt")

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
            "dims": 1024,  # dragonkue/BGE-m3-ko 모델의 출력 차원에 맞춰 1024로 변경
            "index": True,
            "similarity": "l2_norm"
        }
    }
}

# settings, mappings 설정된 내용으로 'test' 인덱스 생성
create_es_index("test", settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성
index_docs = []
with open("/home/InformationRetrieval/data/documents.jsonl") as f:
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

test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

# 역색인을 사용하는 검색 예제 & 결과 출력 테스트
search_result_retrieve = sparse_retrieve(test_query, 3)
print('역색인을 사용하는 검색 예제')
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])

# Vector 유사도 사용한 검색 예제 & 결과 출력 테스트
print('Vector 유사도 사용한 검색 예제')
search_result_retrieve = dense_retrieve(test_query, 3)
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])
    
# Hybrid 검색 예제 & 결과 출력 테스트
print('Hybrid 검색 예제')
search_result_hybrid = hybrid_retrieve(test_query, 3)
for rst in search_result_hybrid['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])


def answer_question(query):
    if isinstance(query, list):
        # 리스트의 각 항목이 딕셔너리인 경우를 처리
        if all(isinstance(item, dict) for item in query):
            # 'content' 키가 있다고 가정하고, 모든 'content' 값을 결합
            query = " ".join(item.get('content', '') for item in query if 'content' in item)
        else:
            # 일반 문자열 리스트인 경우
            query = " ".join(query)
    elif isinstance(query, dict):
        # 단일 딕셔너리인 경우
        query = query.get('content', '')
    
    response = {"standalone_query": query, "topk": []}

    # 하이브리드 검색 사용
    search_result = hybrid_retrieve(query, 5)

    for rst in search_result['hits']['hits']:
        response["topk"].append(rst["_source"]["docid"])

    return response

def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        of.write("eval_id,standalone_query,topk\n")  # CSV 헤더 추가
        for line in f:
            j = json.loads(line)
            response = answer_question(j["msg"])
            
            # CSV 형식으로 출력
            topk_str = ",".join(map(str, response["topk"]))
            # standalone_query에 쉼표가 포함될 수 있으므로 쌍따옴표로 감싸줍니다
            output = f"{j['eval_id']},\"{response['standalone_query']}\",\"{topk_str}\"\n"
            of.write(output)

# 파일 경로 설정
file_root = '/home/InformationRetrieval/data/'
output_file = "/home/InformationRetrieval/result/hybrid_search_results&None-API-Code.csv"

# 평가 데이터에 대해서 결과 생성
eval_rag(os.path.join(file_root, "eval.jsonl"), output_file)
print(f"결과가 {output_file}에 저장되었습니다.")