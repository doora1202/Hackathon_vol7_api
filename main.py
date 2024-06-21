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
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text
        abstracts.append(abstract.strip())
    return abstracts

class QueryData(BaseModel):
    source_abstract: str
    query: str
    max_results: int = 5

@app.post("/similarity/")
async def calculate_similarity(data: QueryData):
    try:
        candidate_abstracts = fetch_abstracts_from_arxiv(data.query, data.max_results)
        if not candidate_abstracts:
            raise HTTPException(status_code=404, detail="No abstracts found for the given query.")
        similarity_payload = {
            "inputs": {
                "source_sentence": data.source_abstract,
                "sentences": candidate_abstracts
            }
        }
        similarity_output = hf_api(similarity_payload)
        return similarity_output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

