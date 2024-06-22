from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()

# Hugging Face API設定
hf_api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
hf_headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

def hf_api(payload):
    response = requests.post(hf_api_url, headers=hf_headers, json=payload)
    if response.status_code != 200:
        return []  # Hugging Face APIからの応答がエラーの場合は空リストを返す
    return response.json()

class QueryData(BaseModel):
    source_abstract: str
    query: str
    max_results: int = 5
    sort_by: str = 'submittedDate'  # sortBy のデフォルト値
    sort_order: str = 'descending'  # sortOrder のデフォルト値

# arXiv APIから詳細情報を取得する関数
def fetch_details_from_arxiv(data: QueryData):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": data.query,
        "max_results": data.max_results,
        "start": 0,
        "sortBy": data.sort_by,
        "sortOrder": data.sort_order
    }
    response = requests.get(base_url, params=params)
    root = ET.fromstring(response.text)
    abstracts = []
    entries = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        id_link = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
        authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
        abstracts.append(abstract)
        entries.append({"title": title, "authors": authors, "abstract": abstract, "link": id_link, "score": None})
    return abstracts, entries

@app.post("/similarity/")
async def calculate_similarity(data: QueryData):
    try:
        abstracts, entries = fetch_details_from_arxiv(data)
        if not abstracts:
            raise HTTPException(status_code=404, detail="No abstracts found for the given query.")
        
        similarity_payload = {
            "inputs": {
                "source_sentence": data.source_abstract,
                "sentences": abstracts
            }
        }
        similarity_results = hf_api(similarity_payload)
        if not similarity_results:
            raise HTTPException(status_code=404, detail="Could not compute similarity.")

        # 各論文にスコアを追加し、スコアに基づいてソート
        for entry, score in zip(entries, similarity_results):
            entry['score'] = score
        sorted_entries = sorted(entries, key=lambda x: x['score'], reverse=True)
        
        return sorted_entries
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
