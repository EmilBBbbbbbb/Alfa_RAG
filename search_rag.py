import re
import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# === Настройки ===
MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
DB_DIR = "chroma_db"


# === Функции очистки текста (из build_db.py) ===
def clean_text_rag(text: str) -> str:
    """Оптимизированная очистка текста для RAG-систем"""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S*@\S*\s?', ' ', text)
    text = re.sub(r'[\r\n\t\f\v]+', ' ', text)
    text = text.replace('\xa0', ' ')
    text = re.sub(r'[^\w\s.,:;!?()\-/а-яa-z0-9]', ' ', text)
    text = re.sub(r'[\u2022\u25CF\uF0A7•◆◇▶️▪️▫️★☆✔️✳️❖\-\*\+]+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'([.,:;!?])\1+', r'\1', text)
    text = re.sub(r'[–—]', '-', text)
    text = re.sub(r'[^а-яa-z0-9\s.,:;!?()\-]', ' ', text)
    text = re.sub(r'\s-\s', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,:;!?])', r'\1', text)
    text = re.sub(r'([.,:;!?])\s+', r'\1 ', text)
    return text.strip()



# === Шаг 1. Загрузка базы Chroma ===
print("Загрузка Chroma базы...")
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)


# === Шаг 2. Загрузка и очистка вопросов ===
questions_df = pd.read_csv("data/questions_clean.csv")
print(f"Загружено {len(questions_df)} вопросов")

# Очистка текста запросов (но без удаления строк)
questions_df["query"] = questions_df["query"].fillna("").apply(clean_text_rag)

print("Примеры после очистки:")
print(questions_df["query"].head(5))


# === Шаг 3. Поиск релевантных документов ===
results = []

for _, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
    q_id = row["q_id"]
    query = row["query"]

    retrieved_docs = db.similarity_search(query, k=5)
    top_web_ids = [doc.metadata.get("web_id") for doc in retrieved_docs]
    results.append({"q_id": q_id, "web_list": top_web_ids})


# === Шаг 4. Сохранение результата ===
output_df = pd.DataFrame(results)
output_df.to_csv("RAG_results_v2.csv", index=False)

print("✅ Поиск завершён. Результаты сохранены в 'RAG_results_v2.csv'.")
print(output_df.head())
