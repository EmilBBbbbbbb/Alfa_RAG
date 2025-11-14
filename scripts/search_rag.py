import pandas as pd
from tqdm import tqdm
import re
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

# === Настройки ===
MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
RERANKER_MODEL = "BAAI/bge-reranker-large"
QDRANT_URL = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "alfa_rag_documents"
ABBR_CSV = "s.csv"  # CSV с колонками: abbr,full

# === Инициализация клиентов ===
qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)
embedding_model = SentenceTransformer(MODEL_NAME)
reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)  # use_fp16=True если есть GPU

# === Загрузка CSV с аббревиатурами ===
abbr_df = pd.read_csv(ABBR_CSV)
abbr_dict = dict(zip(abbr_df["аббревиатура"], abbr_df["расшифровка"]))


# === Функция очистки текста и расшифровки аббревиатур ===
def clean_and_expand_query(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()

    # Очистка лишних символов
    text = re.sub(r'[^\w\s\-\?\!\.,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Расшифровка аббревиатур
    for abbr, full in abbr_dict.items():
        pattern = r'\b' + re.escape(abbr.lower()) + r'\b'
        text = re.sub(pattern, full.lower(), text)

    return text


# === Поиск с реранкингом ===
def search_with_reranking(query: str, top_k: int = 20, rerank_top_k: int = 5):
    """
    Поиск с двухэтапным реранкингом:
    1. Первоначальный поиск в Qdrant (top_k документов)
    2. Реранкинг с помощью модели (rerank_top_k лучших)
    """
    # Шаг 1: Первоначальный поиск в векторной БД
    query_embedding = embedding_model.encode(query).tolist()

    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )

    # Если найдено меньше документов, чем нужно для реранкинга
    if len(search_results) <= rerank_top_k:
        return [hit.payload for hit in search_results]

    # Шаг 2: Реранкинг с помощью cross-encoder модели
    pairs = []
    for hit in search_results:
        doc_content = hit.payload.get('content', '')
        pairs.append((query, doc_content))

    # Получаем скоринги от реранкера
    rerank_scores = reranker.compute_score(pairs)

    # Объединяем результаты с скорингами
    scored_results = []
    for i, hit in enumerate(search_results):
        scored_results.append({
            'payload': hit.payload,
            'vector_score': hit.score,
            'rerank_score': rerank_scores[i] if isinstance(rerank_scores, list) else rerank_scores[i].item(),
            'combined_score': hit.score + (
                rerank_scores[i] if isinstance(rerank_scores, list) else rerank_scores[i].item())
        })

    # Сортируем по комбинированному скорингу и берем топ-N
    scored_results.sort(key=lambda x: x['combined_score'], reverse=True)
    top_reranked = scored_results[:rerank_top_k]

    return [result['payload'] for result in top_reranked]


# === Альтернативная версия с более простым реранкингом ===
def search_simple_reranking(query: str, top_k: int = 20, rerank_top_k: int = 5):
    """
    Упрощенная версия реранкинга - только реранкинг без комбинирования скорингов
    """
    # Первоначальный поиск
    query_embedding = embedding_model.encode(query).tolist()

    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )

    if len(search_results) <= rerank_top_k:
        return [hit.payload for hit in search_results]

    # Подготавливаем пары для реранкинга
    pairs = []
    for hit in search_results:
        doc_content = hit.payload.get('content', '')
        pairs.append((query, doc_content))

    # Реранкинг
    rerank_scores = reranker.compute_score(pairs)

    # Сортируем по скорингу реранкера
    reranked_results = []
    for i, hit in enumerate(search_results):
        reranked_results.append({
            'payload': hit.payload,
            'rerank_score': rerank_scores[i] if isinstance(rerank_scores, list) else rerank_scores[i].item()
        })

    reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
    top_reranked = reranked_results[:rerank_top_k]

    return [result['payload'] for result in top_reranked]


# === Основная функция ===
def main():
    # === Шаг 1. Загрузка вопросов ===
    questions_df = pd.read_csv("../data/questions_clean.csv")
    print(f"Загружено {len(questions_df)} вопросов")

    # === Шаг 2. Очистка и поиск с реранкингом ===
    results = []

    for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Обработка вопросов"):
        q_id = row["q_id"]
        query = row["query"]

        clean_query = clean_and_expand_query(query)

        # Используем поиск с реранкингом
        retrieved_docs = search_simple_reranking(
            query=clean_query,
            top_k=20,  # Первоначальный поиск
            rerank_top_k=5  # Финальный результат после реранкинга
        )

        top_web_ids = [doc.get("web_id") for doc in retrieved_docs]

        # Сохраняем дополнительные метрики для анализа
        results.append({
            "q_id": q_id,
            "original_query": query,
            "cleaned_query": clean_query,
            "web_list": top_web_ids,
            "docs_count": len(retrieved_docs)
        })

    # === Шаг 3. Сохранение результата ===
    output_df = pd.DataFrame(results)
    output_df.to_csv("RAG_results_with_reranking.csv", index=False)
    print("Поиск с реранкингом завершён. Результаты сохранены в 'RAG_results_with_reranking.csv'.")
    print(output_df.head())

    # Статистика
    total_docs = sum(len(result["web_list"]) for result in results)
    print(f"Всего найдено документов: {total_docs}")
    print(f"Среднее количество документов на вопрос: {total_docs / len(results):.2f}")


if __name__ == "__main__":
    main()