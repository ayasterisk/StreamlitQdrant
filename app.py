import streamlit as st
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from openai import OpenAI
import re

st.set_page_config(page_title="HotpotQA Smart RAG", layout="wide")

@st.cache_resource
def init_resources():
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    
    return client, dense_model, sparse_model, llm_client

client, dense_model, sparse_model, llm_client = init_resources()
COLLECTION_NAME = "hotpot_qa"

# Early Stop Logic: Dựa trên Metadata và Keyword Coverage
def early_stop(query, results):
    if not results: return False, "No results"

    # 1. Metadata Logic: Nếu Top-2 đã là bằng chứng xác thực (is_supporting)
    top2_sup = [p for p in results[:2] if p.payload.get('is_supporting') == True]
    if len(top2_sup) >= 2:
        return True, "Metadata: Đã đủ 2 bằng chứng trong Top-2."

    # 2. Keyword Logic: Kiểm tra độ phủ từ khóa của câu hỏi trong Hop-1
    stop_words = {'which', 'what', 'where', 'who', 'is', 'are', 'the', 'a', 'of', 'and', 'in', 'to'}
    query_keywords = set(re.findall(r'\w+', query.lower())) - stop_words
    context_blob = " ".join([p.payload.get('text', '').lower() for p in results])
    
    if query_keywords:
        found_keywords = sum(1 for word in query_keywords if word in context_blob)
        if (found_keywords / len(query_keywords)) >= 0.85:
            return True, "Keyword: Độ phủ từ khóa cao (>85%)."

    return False, "Multi-hop: Cần tìm thêm dữ liệu cầu nối."

# Main Retrieval Logic: Kết hợp Dense + Sparse + Multi-hop
def advanced_retrieval(query_text, top_k=5):
    # Embedding
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query_text]))[0]
    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist()
    )

    # Hop-1: Hybrid Search
    hop1_points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=20),
            models.Prefetch(query=query_sparse, using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points

    # Kiểm tra Early Stop
    should_stop, reason = early_stop(query_text, hop1_points)
    if should_stop:
        return hop1_points, f"Early Stop ({reason})"

    # Nếu không dừng -> Hop-2
    final_evidence = list(hop1_points)
    seen_ids = {p.id for p in final_evidence}
    bridge_titles = {p.payload['title'] for p in hop1_points if p.payload.get('is_supporting')}

    if bridge_titles:
        hop2_points = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="title", match=models.MatchAny(any=list(bridge_titles))),
                    models.FieldCondition(key="is_supporting", match=models.MatchValue(value=True))
                ]
            ),
            limit=10
        )[0]
        for p in hop2_points:
            if p.id not in seen_ids:
                final_evidence.append(p)
                seen_ids.add(p.id)
    
    return final_evidence, "Full Multi-hop"

# Streamlit UI
st.title("Multi-hop RAG Agent")
with st.sidebar:
    st.header("Cấu hình RAG")
    top_k = st.slider("Số lượng tài liệu Hop-1:", 1, 10, 5)
    st.divider()
    st.info("Hệ thống thực hiện tìm kiếm đồng thời qua Vector ngữ nghĩa (Dense) và Từ khóa chính xác (Sparse/Splade).")

query = st.chat_input("Nhập câu hỏi...")
if query:
    with st.chat_message("user"):
        st.write(query)

    with st.status("Đang truy vết dữ liệu...", expanded=True) as status:
        evidence, strategy = advanced_retrieval(query, top_k)
        st.write(f"Chiến lược: **{strategy}**")
        
        context_items = []
        for i, p in enumerate(evidence):
            context_items.append(f"--- TÀI LIỆU [{i+1}] ---\nNGUỒN: {p.payload['title']}\nNỘI DUNG: {p.payload['text']}")
        
        full_context = "\n\n".join(context_items)
        status.update(label=f"Hoàn tất ({strategy})", state="complete")

    with st.chat_message("assistant"):
        with st.spinner("DeepSeek đang suy luận..."):
            prompt = f"""Bạn là Chuyên gia Suy luận Logic. Hãy trả lời CÂU HỎI dựa trên DANH SÁCH TÀI LIỆU dưới đây.
            
            QUY TẮC:
            1. TRÍCH DẪN: Luôn kèm số thứ tự tài liệu [1], [2] khi đưa ra thông tin.
            2. SO SÁNH: Nếu câu hỏi so sánh, hãy lập luận về từng đối tượng trước khi kết luận.
            3. TRUNG THỰC: Nếu không có thông tin trong tài liệu, hãy nói 'Tôi không đủ khả năng để trả lời câu hỏi này'.
            4. KẾT LUẬN: Đưa ra câu trả lời cuối cùng cho câu hỏi một cách đơn giản và rõ ràng, không vòng vo.

            DANH SÁCH TÀI LIỆU:
            {full_context}

            CÂU HỎI: {query}
            
            CÂU TRẢ LỜI (Trình bày logic Suy luận -> Đối chiếu -> Kết luận):"""

            try:
                response = llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Lỗi API: {e}")

    # Hiển thị Metadata gốc để kiểm chứng
    with st.expander("Xem chi tiết Metadata gốc"):
        st.json([p.payload for p in evidence])