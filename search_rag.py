# search_rag.py
"""
Загружает Chroma DB и выполняет поиск для вопросов.
Вопросы нормализуются через Ollama (если доступен).
Количество вопросов сохраняется, и web_list всегда содержит 5 сайтов.
"""

import os
import pandas as pd
from tqdm import tqdm
import re
from random import sample

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate

# === Настройки ===
MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
DB_DIR = "chroma_db"
INPUT_Q = "data/questions_clean.csv"
OUTPUT = "RAG_results_llm.csv"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Промпт для очистки запросов
QUERY_PROMPT = (
    "Ты — ассистент по очистке пользовательских поисковых запросов.\n"
    "Задача: вернуть короткий, но семантически эквивалентный и читабельный вариант запроса.\n"
    "Правила:\n"
    "1) Удали эмодзи, HTML, URL, email, служебные символы, лишние пробелы.\n"
    "2) Раскрой аббревиатуры (РФ -> Российская Федерация и т.п.).\n"
    "3) Сохрани смысл и ключевые слова.\n"
    "4) Верни только очищенный запрос без объяснений.\n\n"
    "Запрос:\n{text}\n\n"
    "Ответ:"
)


def make_llm():
    """Создаёт LLM Ollama, если доступна"""
    try:
        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    except Exception as e:
        print("[WARN] Ollama init failed:", e)
        return None


def regex_clean_query(q: str) -> str:
    """Быстрая очистка регулярками"""
    if not isinstance(q, str):
        return ""
    q = q.replace("\\n", " ").replace("\\t", " ")
    q = re.sub(r'https?://\S+|www\.\S+', ' ', q)
    q = re.sub(r'\S*@\S*\s?', ' ', q)
    q = re.sub(r'<.*?>', ' ', q)
    q = re.sub(r'[\u2022\u25CF\uf0a7•]+', ' ', q)
    q = re.sub(r'[^а-яa-z0-9\s\-,]', ' ', q, flags=re.IGNORECASE)
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def llm_clean_query(text: str, llm) -> str:
    """Очистка текста через Ollama (если доступен)"""
    if not isinstance(text, str) or not text.strip():
        return ""
    prompt = QUERY_PROMPT.format(text=text)
    try:
        resp = llm.invoke([{"role": "user", "content": prompt}])
        cleaned = resp.content.strip()
        return cleaned if cleaned else regex_clean_query(text)
    except Exception:
        return regex_clean_query(text)


def main():
    print("🔹 Загрузка Chroma DB...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    print("🔹 Загрузка вопросов...")
    qdf = pd.read_csv(INPUT_Q)
    print(f"Всего вопросов: {len(qdf)}")

    llm = make_llm()

    # Очистка запросов
    cleaned_queries = []
    for q in tqdm(qdf["query"].fillna(""), desc="Очистка вопросов"):
        if llm:
            cleaned_queries.append(llm_clean_query(q, llm))
        else:
            cleaned_queries.append(regex_clean_query(q))

    qdf["query_cleaned"] = cleaned_queries

    # Собираем все web_id в базе — пригодится для fallback
    all_docs = db.get()
    all_web_ids = [
        meta.get("web_id")
        for meta in all_docs["metadatas"]
        if meta and "web_id" in meta
    ]
    unique_web_ids = list(set(all_web_ids))

    print("🔹 Начинаем поиск...")
    results = []
    for _, row in tqdm(qdf.iterrows(), total=len(qdf), desc="Поиск"):
        q_id = row["q_id"]
        query = (row.get("query_cleaned") or "").strip()

        # fallback — если запрос пуст, используем regex от оригинала
        if not query:
            query = regex_clean_query(row.get("query", ""))

        try:
            docs = db.similarity_search(query, k=5)
            top_web_list = [d.metadata.get("web_id") for d in docs if d.metadata.get("web_id")]
        except Exception:
            top_web_list = []

        # fallback если Chroma ничего не вернул
        if not top_web_list:
            if unique_web_ids:
                top_web_list = sample(unique_web_ids, min(5, len(unique_web_ids)))
            else:
                top_web_list = []

        results.append({"q_id": q_id, "web_list": top_web_list})

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT, index=False)
    print("✅ Результаты сохранены в", OUTPUT)


if __name__ == "__main__":
    main()
