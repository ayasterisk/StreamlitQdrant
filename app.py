import streamlit as st
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from openai import OpenAI

st.set_page_config(page_title="HotpotQA Smart RAG", layout="wide")

# ================= INIT =================
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

# Early stop logic
def early_stop(query, results):
    if not results:
        return False, "No results"

    # Sort lại cho chắc
    results = sorted(results, key=lambda x: x.score or 0, reverse=True)

    scores = [p.score for p in results if p.score is not None]

    if len(scores) < 2:
        return True, "Chỉ có 1 tài liệu → đủ dùng"

    top1 = scores[0]
    top2 = scores[1]
    gap = top1 - top2

    # Diversity check
    titles = [p.payload.get("title", "") for p in results[:3]]
    unique_titles = len(set(titles))

    # Debug log
    print(f"[DEBUG] top1={top1:.3f}, top2={top2:.3f}, gap={gap:.3f}, titles={unique_titles}")

    # Condition 1: top-1 score rất cao
    if top1 > 0.85:
        return True, "Top-1 score rất cao"

    # Condition 2: gap lớn giữa top-1 và top-2
    if gap > 0.2:
        return True, "Score gap lớn → top1 đủ mạnh"

    # Condition 3: nếu top-1 khá cao và không có sự đa dạng (có thể chỉ là 1 tài liệu)
    if unique_titles == 1 and top1 > 0.75:
        return True, "Single document đủ mạnh"

    # Condition 4: nếu có nhiều tài liệu nhưng điểm số rất thấp → có thể không đủ dữ kiện
    if unique_titles >= 2:
        return False, "Nhiều nguồn → cần multi-hop"

    return False, "Không đủ chắc chắn"

def advanced_retrieval(query_text, top_k=5):
    # Embed query
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query_text]))[0]

    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist()
    )

    # Hop-1: Fusion RRF
    hop1_points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=20),
            models.Prefetch(query=query_sparse, using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points
    hop1_points = sorted(hop1_points, key=lambda x: x.score or 0, reverse=True)

    # Early stop check
    should_stop, reason = early_stop(query_text, hop1_points)

    # Chỉ có 1 tài liệu → không cần hop-2
    if len(hop1_points) <= 1:
        return hop1_points, "Early Stop (Chỉ 1 tài liệu)"

    if should_stop:
        return hop1_points, f"Early Stop ({reason})"

    # Hop-2: Tìm kiếm tài liệu hỗ trợ dựa trên metadata "is_supporting" và "title"
    final_evidence = list(hop1_points)
    seen_ids = {p.id for p in final_evidence}

    bridge_titles = {
        p.payload["title"]
        for p in hop1_points
        if p.payload.get("is_supporting")
    }

    if bridge_titles:
        hop2_points = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="title",
                        match=models.MatchAny(any=list(bridge_titles))
                    ),
                    models.FieldCondition(
                        key="is_supporting",
                        match=models.MatchValue(value=True)
                    )
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

query = st.chat_input("Nhập câu hỏi...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.status("Đang truy vết dữ liệu...", expanded=True) as status:
        evidence, strategy = advanced_retrieval(query, top_k=5)

        st.write(f"Chiến lược: **{strategy}**")

        context_items = []
        for i, p in enumerate(evidence):
            context_items.append(
                f"--- TÀI LIỆU [{i+1}] ---\n"
                f"NGUỒN: {p.payload['title']}\n"
                f"NỘI DUNG: {p.payload['text']}"
            )

        full_context = "\n\n".join(context_items)
        status.update(label=f"Hoàn tất ({strategy})", state="complete")

    with st.chat_message("assistant"):
        with st.spinner("DeepSeek đang suy luận..."):
            prompt = f"""Bạn là hệ thống QA suy luận nhiều bước (multi-hop reasoning).

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

    # Debug: show raw metadata
    with st.expander("Xem Metadata gốc"):
        st.json([
            {
                "title": p.payload.get("title"),
                "score": p.score,
                "is_supporting": p.payload.get("is_supporting")
            }
            for p in evidence
        ])