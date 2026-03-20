import streamlit as st
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from openai import OpenAI

st.set_page_config(page_title="HotpotQA Multi-hop RAG", layout="wide", page_icon="🤖")

@st.cache_resource
def init_resources():
    # Lấy thông tin từ Streamlit Secrets
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    
    # Khởi tạo DeepSeek Client
    llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    
    return client, dense_model, sparse_model, llm_client

try:
    client, dense_model, sparse_model, llm_client = init_resources()
except Exception as e:
    st.error(f"Lỗi kết nối hệ thống: {e}. Vui lòng kiểm tra lại Secrets Configuration.")
    st.stop()

COLLECTION_NAME = "hotpot_qa"

def advanced_retrieval(query_text, top_k=5):
    # Embedding Query (Dense + Sparse)
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query_text]))[0]
    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist()
    )

    hop1_results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=20),
            models.Prefetch(query=query_sparse, using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    ).points

    final_evidence = list(hop1_results)
    seen_ids = {p.id for p in final_evidence}
    bridge_titles = {p.payload['title'] for p in hop1_results if p.payload['is_supporting']}

    for title in bridge_titles:
        # Tự động nhảy sang tìm tất cả bằng chứng trong cùng tài liệu cầu nối
        hop2_points = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="title", match=models.MatchValue(value=title)),
                    models.FieldCondition(key="is_supporting", match=models.MatchValue(value=True))
                ]
            ),
            limit=5
        )[0]
        for p in hop2_points:
            if p.id not in seen_ids:
                final_evidence.append(p)
                seen_ids.add(p.id)
    
    return final_evidence

st.title("HotpotQA Advanced RAG System")
st.markdown("Hệ thống suy luận đa bước sử dụng **Hybrid Retrieval** và **DeepSeek V3**.")

with st.sidebar:
    st.header("Cấu hình RAG")
    top_k = st.slider("Số lượng tài liệu Hop-1:", 1, 10, 5)
    st.divider()
    st.info("Hệ thống thực hiện tìm kiếm đồng thời qua Vector ngữ nghĩa (Dense) và Từ khóa chính xác (Sparse/Splade).")

# Ô nhập câu hỏi
query = st.chat_input("Nhập câu hỏi so sánh hoặc bắc cầu (Ví dụ: Which magazine started first...)")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.status("Đang truy vết bằng chứng đa bước...", expanded=True) as status:
        evidence = advanced_retrieval(query, top_k)
        
        context_items = []
        for i, p in enumerate(evidence):
            is_sup = p.payload['is_supporting']
            tag = " [XÁC THỰC]" if is_sup else " [NGỮ CẢNH]"
            context_items.append(f"--- TÀI LIỆU {i+1} ---\nNGUỒN: {p.payload['title']}{tag}\nNỘI DUNG: {p.payload['text']}")
            st.write(f"Đã tìm thấy: **{p.payload['title']}** ({'Bằng chứng' if is_sup else 'Nhiễu'})")
        
        full_context = "\n\n".join(context_items)
        status.update(label="Đã thu thập đủ bằng chứng!", state="complete", expanded=False)

    with st.chat_message("assistant"):
        with st.spinner("DeepSeek đang suy luận logic..."):
            prompt = f"""Bạn là Chuyên gia Suy luận Logic. Hãy trả lời CÂU HỎI dựa trên DANH SÁCH TÀI LIỆU dưới đây.
            
            QUY TẮC:
            1. TRÍCH DẪN: Luôn kèm số thứ tự tài liệu [1], [2] khi đưa ra thông tin.
            2. SO SÁNH: Nếu câu hỏi so sánh, hãy lập luận về từng đối tượng trước khi kết luận.
            3. TRUNG THỰC: Nếu không có thông tin trong tài liệu, hãy nói 'Tôi không biết'.
            4. PHÂN BIỆT: Rõ ràng phân biệt giữa bằng chứng hỗ trợ (is_supporting=True) và ngữ cảnh liên quan nhưng không hỗ trợ trực tiếp (is_supporting=False).
            5. KẾT LUẬN: Dựa trên bằng chứng, đưa ra câu trả lời cuối cùng cho câu hỏi một cách đơn giản và rõ ràng, không vòng vo.

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
                st.error(f"Lỗi gọi DeepSeek API: {e}")

    with st.expander("Xem chi tiết Metadata & Bằng chứng gốc"):
        for p in evidence:
            color = "green" if p.payload['is_supporting'] else "gray"
            st.markdown(f":{color}[**[{p.payload['title']}]**] {p.payload['text']}")
            st.json(p.payload)