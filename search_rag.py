# search_rag.py

import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Настройки ===
MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
DB_DIR = "chroma_db"

# === Шаг 1. Загрузка базы Chroma ===
print("Загрузка Chroma базы...")
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

# === Шаг 2. Загрузка вопросов ===
questions_df = pd.read_csv("data/questions_clean.csv")
print(f"Загружено {len(questions_df)} вопросов")

# === Шаг 3. Поиск релевантных документов ===
results = []

for _, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
    q_id = row["q_id"]
    query = row["query"]

    retrieved_docs = db.similarity_search(query, k=5)
    top_web_ids = [doc.metadata.get("web_id") for doc in retrieved_docs]

    results.append({"q_id": q_id, "web_ids": top_web_ids})

# === Шаг 4. Сохранение результата ===
output_df = pd.DataFrame(results)
output_df.to_csv("RAG_results.csv", index=False)

print("✅ Поиск завершён. Результаты сохранены в 'RAG_results.csv'.")
print(output_df.head())
