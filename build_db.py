# build_db.py
"""
Создаёт Chroma DB. Тексты очищаются через локальную Ollama-модель (LLM).
Для долгих текстов реализовано разбиение на чанки.
Если Ollama недоступен — используется безопасная regex-очистка.
"""

import os
import math
import pandas as pd
from tqdm import tqdm
import re

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Ollama chat model via langchain community
from langchain_ollama import ChatOllama


# === Настройки ===
MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
DB_DIR = "chroma_db"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")   # изменить при необходимости
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
INPUT_CSV = "data/websites_updated.csv"
CHUNK_SIZE = 2000      # символы в чанке
CHUNK_OVERLAP = 200

# === Промпт очистки для LLM ===
CLEAN_PROMPT = (
    "Ты — ассистент по интеллектуальной очистке текстов для RAG-систем.\n"
    "Обработай фрагмент текста по правилам:\n"
    "1) Удали HTML-теги, ссылки, e-mail, эмодзи, управляющие и мусорные символы.\n"
    "2) Удали повторяющиеся и неинформативные маркеры (***, ---, • и т.п.).\n"
    "3) Раскрой распространённые аббревиатуры/сокращения (напр., РФ → Российская Федерация, т.е. → то есть).\n"
    "4) Нормализуй пробелы и пунктуацию, сохрани смысл.\n"
    "5) Не добавляй новую информацию, дай только очищенный текст.\n\n"
    "Фрагмент:\n{text}\n\n"
    "Верни только очищенный фрагмент без пояснений."
)

# === Простая regex-очистка — fallback если LLM недоступен ===
def regex_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\\n", " ").replace("\\t", " ")
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # ссылки
    text = re.sub(r'\S*@\S*\s?', ' ', text)            # emails
    text = re.sub(r'<.*?>', ' ', text)                 # html
    text = re.sub(r'[\r\n\t\f\v]+', ' ', text)
    text = text.replace('\xa0', ' ')
    # удалить эмодзи / нестандартные юникод-символы
    text = re.sub(r'[^\w\s\.,:;!?\-()/а-яa-z0-9]', ' ', text)
    text = re.sub(r'[\u2022\u25CF\uf0a7•◆◇▶️▪️▫️★☆✔️✳️❖\-\*\+]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # простой семантический фильтр: если стало слишком коротко — пусто
    if len(text) < 20 or re.match(r'^[\d\s\W]+$', text):
        return ""
    return text

# === LLM wrapper: разделение на чанки, вызов Ollama, сборка результата ===
def llm_clean_text(text: str, llm, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    # если текст короткий — один запрос
    if len(text) <= chunk_size:
        prompt = CLEAN_PROMPT.format(text=text)
        try:
            resp = llm.invoke([{"role":"user", "content": prompt}])
            return resp.content.strip()
        except Exception as e:
            # fallback
            return regex_clean(text)

    # длинный текст: разбиваем
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    parts = splitter.split_text(text)
    cleaned_parts = []
    for part in parts:
        prompt = CLEAN_PROMPT.format(text=part)
        try:
            resp = llm.invoke([{"role":"user", "content": prompt}])
            cleaned_parts.append(resp.content.strip())
        except Exception:
            cleaned_parts.append(regex_clean(part))
    # соединяем части аккуратно
    combined = " ".join(p for p in cleaned_parts if p)
    # финальная regex-проход для нормализации пробелов
    combined = re.sub(r'\s+', ' ', combined).strip()
    if len(combined) < 20 or re.match(r'^[\d\s\W]+$', combined):
        return ""
    return combined

def make_llm():
    try:
        # ChatOllama поддерживает base_url и model
        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    except Exception as e:
        print("[WARN] Ollama init failed:", e)
        return None

def main():
    print("Загрузка CSV...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Загружено {len(df)} страниц")

    llm = make_llm()
    cleaned_texts = []
    cleaned_titles = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Обработка"):
        text = row.get("text", "") or ""
        title = row.get("title", "") or ""

        if llm is not None:
            cleaned_t = llm_clean_text(text, llm)
            cleaned_title = llm_clean_text(title, llm) if title else ""
        else:
            cleaned_t = regex_clean(text)
            cleaned_title = regex_clean(title)

        cleaned_texts.append(cleaned_t)
        cleaned_titles.append(cleaned_title)

    df["text"] = cleaned_texts
    df["title"] = cleaned_titles

    # удалить пустые тексты
    before = len(df)
    df = df[df["text"].str.strip() != ""]
    after = len(df)
    print(f"После очистки осталось {after} (удалено {before-after})")

    # Подготовка документов
    docs = []
    for _, row in df.iterrows():
        docs.append(
            Document(
                page_content=row["text"],
                metadata={
                    "web_id": row.get("web_id"),
                    "title": row.get("title"),
                    "url": row.get("url"),
                    "kind": row.get("kind"),
                }
            )
        )

    print("Создание эмбеддингов и Chroma DB (это может занять время)...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=DB_DIR)
    db.persist()
    print("✅ Chroma DB готова и сохранена в", DB_DIR)

if __name__ == "__main__":
    main()
