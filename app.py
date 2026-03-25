import streamlit as st
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from openai import OpenAI

# ================== CONFIG ==================
st.set_page_config(page_title="HotpotQA Smart RAG (Optimized)", layout="wide")

COLLECTION_NAME = "hotpot_qa"

# ================== INIT ==================
@st.cache_resource
def init_resources():
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    llm_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

    return client, dense_model, sparse_model, llm_client


client, dense_model, sparse_model, llm_client = init_resources()

# ================== UTILS ==================
def rerank(points):
    return sorted(points, key=lambda x: x.score if x.score else 0, reverse=True)


def early_stop(query, results):
    if not results:
        return False, "No results"

    supporting = [p for p in results if p.payload.get("is_supporting")]

    # Multi-hop evidence check
    titles = {p.payload["title"] for p in supporting}
    if len(supporting) >= 2 and len(titles) >= 2:
        return True, "Metadata: đủ multi-hop evidence"

    # High confidence score
    high_score = [p for p in results if p.score and p.score > 0.75]
    if len(high_score) >= 2:
        return True, "Score: độ tin cậy cao"

    return False, "Cần multi-hop"


def expand_query(query, hop1_points):
    titles = [p.payload["title"] for p in hop1_points]
    snippets = [p.payload["text"][:100] for p in hop1_points[:3]]

    expanded = query + " " + " ".join(titles) + " " + " ".join(snippets)
    return expanded


def build_context(evidence):
    context_items = []
    for i, p in enumerate(evidence):
        context_items.append(
            f"[{i+1}] ({p.payload['title']}) {p.payload['text']}"
        )
    return "\n".join(context_items)


# ================== RETRIEVAL ==================
def advanced_retrieval(query_text, top_k=5):
    # ===== HOP-1 =====
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query_text]))[0]

    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist()
    )

    hop1 = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=20),
            models.Prefetch(query=query_sparse, using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points

    hop1 = rerank(hop1)

    # ===== EARLY STOP =====
    should_stop, reason = early_stop(query_text, hop1)
    if should_stop:
        return hop1[:top_k], f"Early Stop ({reason})"

    # ===== HOP-2 (QUERY EXPANSION) =====
    expanded_query = expand_query(query_text, hop1)

    query_dense_2 = list(dense_model.embed([expanded_query]))[0].tolist()
    query_sparse_raw_2 = list(sparse_model.embed([expanded_query]))[0]

    query_sparse_2 = models.SparseVector(
        indices=query_sparse_raw_2.indices.tolist(),
        values=query_sparse_raw_2.values.tolist()
    )

    hop2 = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense_2, using="dense", limit=20),
            models.Prefetch(query=query_sparse_2, using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points

    # ===== MERGE =====
    all_points = {p.id: p for p in hop1}
    for p in hop2:
        all_points[p.id] = p

    final_points = list(all_points.values())
    final_points = rerank(final_points)

    return final_points[:6], "Full Multi-hop (Expanded Query)"


# ================== UI ==================
st.title("🚀 Multi-hop RAG Agent (Optimized)")

query = st.chat_input("Nhập câu hỏi...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.status("🔍 Đang truy vết dữ liệu...", expanded=True) as status:
        evidence, strategy = advanced_retrieval(query)

        st.write(f"**Chiến lược:** {strategy}")

        full_context = build_context(evidence)

        status.update(label="✅ Hoàn tất truy xuất", state="complete")

    with st.chat_message("assistant"):
        with st.spinner("🧠 DeepSeek đang suy luận..."):
            prompt = f"""
                Bạn là hệ thống QA suy luận nhiều bước (multi-hop reasoning).

                QUY TẮC: 
                1. TRÍCH DẪN: Luôn kèm số thứ tự tài liệu [1], [2] khi đưa ra thông tin. 
                2. SO SÁNH: Nếu câu hỏi so sánh, hãy lập luận về từng đối tượng trước khi kết luận. 
                3. TRUNG THỰC: Nếu không có thông tin trong tài liệu, hãy nói 'Tôi không đủ khả năng để trả lời câu hỏi này'. 
                4. CỐ GẮNG SỬ DỤNG TÀI LIỆU: Cố gắng phân tích và sử dụng thông tin từ tài liệu để trả lời, đừng chỉ dựa vào kiến thức chung.
                5. KẾT LUẬN: Đưa ra câu trả lời cuối cùng cho câu hỏi một cách đơn giản và rõ ràng, không vòng vo.

                TÀI LIỆU:
                {full_context}

                CÂU HỎI:
                {query}

                TRẢ LỜI:
                """

            try:
                response = llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )

                st.markdown(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Lỗi API: {e}")

    # ===== DEBUG METADATA =====
    with st.expander("📊 Xem Metadata"):
        st.json([p.payload for p in evidence])