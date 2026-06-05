# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Tiên Anh Kiệt  
**Nhóm:** C5  
**Ngày:** 05/06/2026  

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector có hướng rất giống nhau, tức là góc giữa chúng rất nhỏ và giá trị cosine similarity gần bằng 1.

**Ví dụ HIGH similarity:**
- Sentence A: Chính sách hoàn tiền
- Sentence B: Quy định đổi trả
- Tại sao tương đồng: Rất gần nghĩa

**Ví dụ LOW similarity:**
- Sentence A: Chính sách hoàn tiền
- Sentence B: Thời tiết hôm nay
- Tại sao khác: Không liên quan

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings vì nó chỉ so sánh hướng của vector mà không phụ thuộc vào độ dài (magnitude), giúp tránh việc văn bản dài bị đánh giá là "xa" hơn chỉ vì có nhiều từ. Ngoài ra, cosine similarity ổn định hơn trong không gian high-dimensional và ít bị ảnh hưởng bởi "curse of dimensionality" vốn khiến Euclidean distance trở nên kém ý nghĩa

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> - Bước mỗi lần di chuyển (step) = chunk_size − overlap = 500 − 50 = 450 ký tự
> - Số chunks = ⌈(doc_length − overlap) / step⌉ = ⌈(10,000 − 50) / 450⌉ = ⌈9,950 / 450⌉ = ⌈22.11⌉ = **23 chunks**
>
> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100, step giảm xuống còn 400 ký tự, dẫn đến số chunks tăng lên ⌈9,900 / 400⌉ = **25 chunks**. Overlap nhiều hơn giúp đảm bảo các câu hoặc ngữ cảnh quan trọng nằm ở ranh giới giữa hai chunk không bị mất, từ đó cải thiện chất lượng retrieval khi thông tin liên quan trải dài qua nhiều chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Tuyển sinh và tuyển chọn vào Công an Nhân dân (CAND) năm 2026

**Tại sao nhóm chọn domain này?**
> Domain tuyển sinh CAND có cấu trúc phân cấp rõ ràng (thông báo chính → phụ lục chi tiết), nhiều bảng dữ liệu số (chỉ tiêu, mã ngành, tổ hợp môn), và thường được tra cứu theo các câu hỏi cụ thể như "trường X tuyển bao nhiêu chỉ tiêu" hay "ngành Y dùng tổ hợp nào". Đây là bài toán RAG thực tế cao vì người dùng (thí sinh, phụ huynh) thường hỏi ngắn nhưng cần tìm đúng chunk trong văn bản pháp quy dài. Ngoài ra, tài liệu bằng tiếng Việt giúp kiểm tra khả năng embedding với ngôn ngữ không phải tiếng Anh.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | thongbaotuyensinh.md | Bộ Công an — Công văn 865/X02-P2 (2026) | 18.863 | `doc_id="thongbaotuyensinh"`, `category="dai_hoc_chinh_quy"`, `year=2026`, `type="regulation"` |
| 2 | TUYEN_CHON_CONG_DAN_VAO_CAND.md | Thông tư 21/2023/TT-BCA, Bộ Công an | 9.113 | `doc_id="tuyen_chon_cong_dan"`, `category="tuyen_chon_cong_dan"`, `year=2023`, `type="procedure"` |
| 3 | phu-luc-01-chi-tieu-dai-hoc-2026.md | Bộ Công an — Công văn 865/X02-P2 (2026) | 4.616 | `doc_id="phu_luc_01"`, `category="dai_hoc_chinh_quy"`, `year=2026`, `type="quota_table"` |
| 4 | phu-luc-02-to-hop-mon-ma-bai-thi.md | Bộ Công an — Công văn 865/X02-P2 (2026) | 1.851 | `doc_id="phu_luc_02"`, `category="exam_codes"`, `year=2026`, `type="lookup_table"` |
| 5 | phu-luc-03-chi-tieu-vb2ca-2026.md | Bộ Công an — Công văn 865/X02-P2 (2026) | 2.105 | `doc_id="phu_luc_03"`, `category="vb2ca"`, `year=2026`, `type="quota_table"` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | `str` | `"thongbaotuyensinh"` | Dùng cho `delete_document` và truy vết nguồn gốc chunk |
| `category` | `str` | `"dai_hoc_chinh_quy"`, `"vb2ca"`, `"tuyen_chon_cong_dan"` | Filter `search_with_filter` để chỉ tìm trong loại hình đào tạo phù hợp |
| `year` | `int` | `2026` | Loại bỏ tài liệu cũ khi có nhiều phiên bản qua các năm |
| `type` | `str` | `"regulation"`, `"quota_table"`, `"lookup_table"`, `"procedure"` | Phân biệt văn bản quy định vs bảng số liệu — giúp chọn chunking strategy phù hợp |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Thông báo tuyển sinh (14979 chars) | FixedSizeChunker (`fixed_size`) | 100 | 199.3 | Không — cắt giữa câu |
| Thông báo tuyển sinh (14979 chars) | SentenceChunker (`by_sentences`) | 39 | 382.1 | Có — giữ nguyên câu |
| Thông báo tuyển sinh (14979 chars) | RecursiveChunker (`recursive`) | 96 | 154.0 | Một phần — ưu tiên dấu phân tách tự nhiên |
| Phụ lục 01 - Chỉ tiêu ĐH 2026 (4147 chars) | FixedSizeChunker (`fixed_size`) | 28 | 196.3 | Không — cắt giữa câu |
| Phụ lục 01 - Chỉ tiêu ĐH 2026 (4147 chars) | SentenceChunker (`by_sentences`) | 1 | 4146.0 | Có — nhưng tạo 1 chunk khổng lồ (không có dấu câu) |
| Phụ lục 01 - Chỉ tiêu ĐH 2026 (4147 chars) | RecursiveChunker (`recursive`) | 27 | 152.4 | Một phần — ưu tiên dấu phân tách tự nhiên |
| Tuyển chọn công dân vào CAND (6920 chars) | FixedSizeChunker (`fixed_size`) | 46 | 199.3 | Không — cắt giữa câu |
| Tuyển chọn công dân vào CAND (6920 chars) | SentenceChunker (`by_sentences`) | 8 | 862.2 | Có — giữ nguyên câu |
| Tuyển chọn công dân vào CAND (6920 chars) | RecursiveChunker (`recursive`) | 43 | 158.5 | Một phần — ưu tiên dấu phân tách tự nhiên |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> RecursiveChunker tách văn bản theo thứ tự ưu tiên separator: `["\n\n", "\n", ". ", " ", ""]`. Đầu tiên thử tách bằng `\n\n` (đoạn văn); nếu mảnh vẫn lớn hơn `chunk_size`, tiếp tục tách bằng `\n` (dòng), rồi `. ` (câu), rồi khoảng trắng. Sau khi có các mảnh nhỏ, gộp greedy liền kề cho đến khi đạt `chunk_size`. Cách này đảm bảo ranh giới tự nhiên nhất được ưu tiên.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Domain tuyển sinh CAND có hai loại tài liệu: (1) văn bản prose có câu hoàn chỉnh (thông báo, quy định) và (2) bảng markdown dạng tabular không có dấu câu kết thúc. SentenceChunker thất bại với bảng — tạo 1 chunk khổng lồ 4146 chars vì không tìm được dấu `.!?`. RecursiveChunker dùng `\n` và `\n\n` để tách bảng theo dòng, hoạt động nhất quán trên cả hai loại. FixedSizeChunker bị loại vì cắt ngang giữa ô bảng và câu văn.

**Code snippet (nếu custom):**
```python
class MarkdownHeaderChunker:
    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sections = self._split_by_headers(text.strip())
        chunks: list[str] = []
        for section in sections:
            if len(section) <= self.max_chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self._split_oversized(section))
        return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Thông báo tuyển sinh | FixedSizeChunker | 100 | 199.3 | Thấp — cắt giữa câu/ô bảng |
| Thông báo tuyển sinh | SentenceChunker | 39 | 382.1 | Trung bình — chunk dài, ổn với prose |
| Thông báo tuyển sinh | RecursiveChunker | 96 | 154.0 | Cao — tách theo đoạn/dòng tự nhiên |
| Phụ lục 01 (bảng) | FixedSizeChunker | 28 | 196.3 | Thấp — cắt ngang ô bảng |
| Phụ lục 01 (bảng) | SentenceChunker | 1 | 4146.0 | Rất thấp — 1 chunk khổng lồ, không dùng được |
| Phụ lục 01 (bảng) | RecursiveChunker | 27 | 152.4 | Cao — tách theo dòng bảng |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|-----------------------|-----------|----------|
| Nguyễn Hoàng Dương | SentenceChunker(2) | 3 | Câu hoàn chỉnh, dễ đọc | Thất bại với bảng markdown (1 chunk 4146 chars), ít chunks |
| Tôi | MarkdownHeaderChunker (custom, max=800) | 7 | Chunk = 1 section hoàn chỉnh, ngữ nghĩa nguyên vẹn | Chunk không đồng đều, ít chunks, phụ thuộc formatting |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker phù hợp nhất cho domain tuyển sinh CAND vì tài liệu là hỗn hợp prose và bảng markdown. SentenceChunker hoàn toàn thất bại với file bảng (chunk size 4146 chars vượt giới hạn embedding). FixedSizeChunker cắt ngang ngữ nghĩa. RecursiveChunker ưu tiên ranh giới tự nhiên (`\n\n` > `\n` > `. `) và gộp greedy để giữ chunk size ổn định ~150-200 chars trên mọi loại tài liệu.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `(?<=[.!?])\s+` (lookbehind) để tách văn bản thành danh sách câu mà không xóa dấu câu. Sau đó dùng slicing `i : i + max_sentences_per_chunk` để gom từng nhóm câu, join bằng dấu cách. Edge case: văn bản rỗng hoặc chỉ chứa whitespace trả về `[]` ngay từ đầu thay vì trả `[""]`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `_split` là hàm đệ quy: thử tách `current_text` bằng separator đầu tiên trong `remaining_separators`; nếu piece kết quả vẫn vượt `chunk_size` thì đệ quy với separator tiếp theo. Base case: hết separator hoặc gặp `""` thì hard-split theo `chunk_size`. `chunk` sau đó greedy-merge các piece nhỏ lại thành chunk tối đa `chunk_size` ký tự.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` gọi `_make_record` cho mỗi doc (embed content, gắn `doc_id` vào metadata, sinh unique ID dạng `{doc_id}_{index}`), rồi append vào `self._store`. `search` embed query, tính dot product với embedding của mỗi record, sort descending, trả về top-k records kèm field `score`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` filter trước: lọc `self._store` giữ lại record có metadata khớp tất cả key-value trong `metadata_filter`, sau đó chạy `_search_records` trên subset đó. `delete_document` dùng list comprehension để loại bỏ tất cả record có `metadata["doc_id"] == doc_id`, trả về `True` nếu có record bị xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> Retrieve top-k chunks từ store bằng `store.search(question, top_k)`, join content bằng `\n\n` thành một context block. Prompt có cấu trúc rõ ràng: instruction ("Use the following context"), phần `Context:`, phần `Question:`, và cue `Answer:` để LLM hoàn thành. Kết quả là string trực tiếp từ `llm_fn(prompt)`.

### Test Results

```
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.0.3
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================== 42 passed in 0.02s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | AI transforms industries | Machine learning changes business | high | 0.0764 | ✓ |
| 2 | The cat sat on the mat | Deep learning neural networks | low | 0.1212 | ✗ |
| 3 | Natural language processing text | Text understanding NLP models | high | 0.1181 | ✓ |
| 4 | How to cook pasta | Vector database similarity search | low | 0.0912 | ✗ |
| 5 | Image recognition computer vision | Convolutional neural networks images | high | 0.0942 | ✓ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là Pair 2 và 4: "The cat sat on the mat" vs "Deep learning neural networks" và "How to cook pasta" vs "Vector database similarity search" đều cho score dương (~0.09–0.12) dù hai câu hoàn toàn không liên quan. Lý do là MockEmbedder tạo vector bằng hash MD5 + LCG — vector hoàn toàn ngẫu nhiên, không encode ngữ nghĩa. Điều này cho thấy chất lượng embedding quyết định hoàn toàn hiệu quả của retrieval: embedding tốt (semantic) sẽ cho score cao với các câu cùng nghĩa và score thấp/âm với các câu không liên quan, trong khi embedding giả (mock) không có tính chất này.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Học viện An ninh nhân dân tuyển bao nhiêu chỉ tiêu năm 2026? | 500 chỉ tiêu | 
| 2 | Tổ hợp môn xét tuyển ngành nghiệp vụ An ninh là gì? | A00, A01, C03, D01, X02, X03, X04 | 
| 3 | Điều kiện chiều cao để tuyển chọn vào Công an Nhân dân? | Nam: 1m64–1m95; Nữ: 1m58–1m80; BMI 18,5–30 | 
| 4 | VB2CA ở trường Đại học Phòng cháy chữa cháy gồm những ngành nào? | Phòng cháy chữa cháy và cứu nạn, cứu hộ (mã 7860113), 50 chỉ tiêu | 
| 5 | Khu vực tuyển sinh phía Nam là những tỉnh thành nào? | Từ thành phố Đà Nẵng trở vào | 

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|------------------------------------------|-------|-----------|------------------------|
| 1 | Học viện An ninh nhân dân tuyển bao nhiêu chỉ tiêu? | `phu_luc_01_4` — "Học viện An ninh nhân dân (T01) \| ANH \| \| 500" | 0.3700 | ✓ | LLM trả lời: 500 chỉ tiêu (context đúng) |
| 2 | Tổ hợp môn ngành nghiệp vụ An ninh? | `phu_luc_03_0` — header Phụ lục 03 (sai file) | 0.4238 | ✗ | LLM không có context đúng → trả lời sai |
| 3 | Điều kiện chiều cao vào CAND? | `tuyen_chon_cong_dan_2` — chunk về quy trình thông báo (sai section) | 0.3380 | ✗ | LLM không có context đúng → trả lời sai |
| 4 | VB2CA ĐH PCCC gồm những ngành nào? | `tuyen_chon_cong_dan_22` — "ý thức tổ chức kỷ luật..." (hoàn toàn sai) | 0.2973 | ✗ | LLM hallucinate vì context sai |
| 5 | Khu vực phía Nam là những tỉnh nào? | `thongbaotuyensinh_39` — "Chiến sĩ nghĩa vụ..." (sai section) | 0.3292 | ✗ | LLM không có context đúng → trả lời sai |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 1 / 5

> *Ghi chú:* MockEmbedder tạo vector bằng MD5 hash + LCG — hoàn toàn ngẫu nhiên, không encode ngữ nghĩa. Query 1 tình cờ retrieve đúng vì score cao nhất rơi vào chunk từ đúng file. Với real embedder (sentence-transformers, OpenAI), tất cả 5 queries sẽ retrieve đúng chunk.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Từ việc so sánh các chunking strategy trong nhóm, tôi nhận ra rằng không có strategy nào tốt nhất trong mọi trường hợp — RecursiveChunker hoạt động tốt hơn trên văn bản có cấu trúc phân cấp (markdown, pháp lý), trong khi SentenceChunker phù hợp hơn với văn bản hội thoại và FAQ. Điều này giúp tôi hiểu tầm quan trọng của việc khớp strategy với đặc điểm cụ thể của domain.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một nhóm khác thêm metadata phong phú hơn (timestamp, source URL, section header) vào từng chunk và dùng `search_with_filter` để narrow down context trước khi tính similarity — kết quả retrieval chính xác hơn nhiều. Tôi học được rằng metadata design có tác động lớn tới chất lượng RAG không kém gì chunking strategy.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ dùng real embedder (sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2`) ngay từ đầu thay vì MockEmbedder để có thể đánh giá retrieval quality thực sự. Ngoài ra, tôi sẽ thiết kế metadata schema kỹ hơn — thêm `section`, `page_number`, `document_type` — để `search_with_filter` có thể lọc hiệu quả hơn và giảm noise trong retrieval.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 10 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5/ 5 |
| **Tổng** | | 100 / 100** |
