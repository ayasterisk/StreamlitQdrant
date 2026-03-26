import streamlit as st
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from openai import OpenAI

st.set_page_config(page_title="HotpotQA Smart RAG", layout="wide")

@st.cache_resource
def init_resources():
    client = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )

    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    llm_client = OpenAI(
        api_key=st.secrets["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )

    return client, dense_model, sparse_model, llm_client


client, dense_model, sparse_model, llm_client = init_resources()
COLLECTION_NAME = "hotpot_qa"

# Early stop logic dựa trên score và metadata
def early_stop(results):
    if not results:
        return False, "No results"

    scores = [p.score for p in results if hasattr(p, "score")]

    if len(scores) < 2:
        return True, "Chỉ 1 tài liệu → đủ"

    top1, top2 = scores[0], scores[1]
    gap = top1 - top2

    titles = [p.payload.get("title", "") for p in results[:3]]
    unique_titles = len(set(titles))

    # Debug: In ra các giá trị để hiểu rõ hơn
    print(f"[DEBUG] top1={top1:.3f}, top2={top2:.3f}, gap={gap:.3f}, titles={unique_titles}")

    if top1 > 0.85:
        return True, "Top-1 score rất cao"

    if gap > 0.2:
        return True, "Score gap lớn"

    if unique_titles == 1 and top1 > 0.75:
        return True, "Single doc đủ mạnh"

    return False, "Cần multi-hop"


# Main retrieval function với logic multi-hop và early stop
def advanced_retrieval(query_text, top_k=5):

    # Embed query cho cả dense và sparse
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query_text]))[0]

    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist()
    )

    # HOP-1: Truy vấn kết hợp dense + sparse với Fusion RRF
    hop1_points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=20),
            models.Prefetch(query=query_sparse, using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points

    # Sort hop1 results by score (nếu có) để dễ dàng áp dụng early stop
    hop1_points = sorted(
        hop1_points,
        key=lambda x: getattr(x, "score", 0),
        reverse=True
    )

    #Early stop sau HOP-1 nếu thấy đủ tự tin
    if len(hop1_points) <= 1:
        return hop1_points, "Early Stop (1 tài liệu)"

    should_stop, reason = early_stop(hop1_points)

    if should_stop:
        return hop1_points, f"Early Stop ({reason})"

    # HOP-2: Nếu cần multi-hop, truy vấn tiếp với filter dựa trên metadata "is_supporting"
    final_evidence = list(hop1_points)
    seen_ids = {p.id for p in final_evidence}

    bridge_titles = {
        p.payload.get("title")
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

    with st.status("Đang truy vết dữ liệu...", expanded=True):
        evidence, strategy = advanced_retrieval(query)

        st.write(f"Chiến lược: **{strategy}**")

        context = "\n\n".join([
            f"[{i+1}] {p.payload.get('title')}\n{p.payload.get('text')}"
            for i, p in enumerate(evidence)
        ])

    with st.chat_message("assistant"):
        with st.spinner("Đang suy luận..."):
            prompt = f"""Bạn là hệ thống QA suy luận nhiều bước (multi-hop reasoning).

                QUY TẮC: 
                1. TRÍCH DẪN: Luôn kèm số thứ tự tài liệu [1], [2] khi đưa ra thông tin. 
                2. SO SÁNH: Nếu câu hỏi so sánh, hãy lập luận về từng đối tượng trước khi kết luận. 
                3. TRUNG THỰC: Nếu không có thông tin trong tài liệu, hãy nói 'Tôi không đủ khả năng để trả lời câu hỏi này'. 
                4. CỐ GẮNG SỬ DỤNG TÀI LIỆU: Cố gắng phân tích và sử dụng thông tin từ tài liệu để trả lời, đừng chỉ dựa vào kiến thức chung.
                5. KẾT LUẬN: Đưa ra câu trả lời cuối cùng cho câu hỏi một cách đơn giản và rõ ràng, không vòng vo.

                TÀI LIỆU:
                {context}

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

    # Debug: Hiển thị metadata của các tài liệu được truy vết
    with st.expander("Debug Metadata"):
        st.json([
            {
                "title": p.payload.get("title"),
                "text": p.payload.get("text"),
                "score": getattr(p, "score", None),
                "is_supporting": p.payload.get("is_supporting")
            }
            for p in evidence
        ])