import os
import sys
from pypdf import PdfReader
from pinecone import Pinecone
from dotenv import load_dotenv
import requests
import json

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=" + GEMINI_API_KEY
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "quiz-notes"
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(name=index_name, dimension=768)
index = pc.Index(index_name)

def get_gemini_embedding(text):
    headers = {"Content-Type": "application/json"}
    data = {"content": {"parts": [{"text": text}]}}
    response = requests.post(GEMINI_EMBED_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["embedding"]["values"]
    else:
        print("Gemini embedding failed:", response.text)
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_notes.py <path_to_pdf>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print("File not found:", pdf_path)
        sys.exit(1)
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    # Save extracted text to teacher_notes.txt for quiz generation
    with open("teacher_notes.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Extracted text saved to teacher_notes.txt.")
    emb = get_gemini_embedding(text[:4000])
    if emb:
        index.upsert([("teacher_notes", emb, {"desc": "Class notes uploaded by teacher"})])
        print("Notes uploaded and embedded successfully!")
    else:
        print("Embedding failed.")
