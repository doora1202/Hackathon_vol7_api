from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import xml.etree.ElementTree as ET

app = FastAPI()

# Hugging Face API設定
hf_api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
hf_headers = {"Authorization": "Bearer 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"}

def hf_api(payload):
    response = requests.post(hf_api_url, headers=hf_headers, json=payload)
    return response.json()

# arXiv APIからアブストラクトを取得する関数
def fetch_abstracts_from_arxiv(query, max_results=5):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "max_results": max_results,
        "start": 0
    }
    response = requests.get(base_url, params=params)
    root = ET.fromstring(response.text)
    abstracts = []
    entries = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        id_link = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
        abstracts.append(abstract)
        entries.append({"abstract": abstract, "link": id_link})
    return abstracts, entries

class QueryData(BaseModel):
    source_abstract: str
    query: str
    max_results: int = 5

@app.post("/similarity/")
async def calculate_similarity(data: QueryData):
    try:
        abstracts, entries = fetch_abstracts_from_arxiv(data.query, data.max_results)
        if not abstracts:
            raise HTTPException(status_code=404, detail="No abstracts found for the given query.")
        
        similarity_payload = {
            "inputs": {
                "source_sentence": data.source_abstract,
                "sentences": abstracts
            }
        }
        similarity_output = hf_api(similarity_payload)
        
        # 各論文にスコアを追加
        for entry, score in zip(entries, similarity_output):
            entry['score'] = score
        
        return entries
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

