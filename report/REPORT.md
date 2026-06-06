# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Le Sy Han
**Nhóm:** 
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> ***High cosine similarity** (độ tương đồng cosine cao) cho thấy hai vector có hướng rất giống nhau, nghĩa là chúng có nội dung hoặc đặc trưng tương đồng. Trong các hệ thống embedding và RAG, cosine similarity cao thường biểu thị mức độ liên quan ngữ nghĩa cao giữa truy vấn và tài liệu được truy xuất.*

**Ví dụ HIGH similarity:**
- Sentence A: "SpatialGlue integrates spatial transcriptomics and proteomics data."
- Sentence B: "SpatialGlue combines spatial transcriptomic and proteomic information."
- Tại sao tương đồng: Cả hai câu diễn đạt cùng một ý nghĩa về việc tích hợp dữ liệu đa omics không gian.

**Ví dụ LOW similarity:**
- Sentence A: "SpatialGlue integrates spatial transcriptomics and proteomics data."
- Sentence B: "Python is a popular programming language for web development."
- Tại sao khác: Hai câu nói về các chủ đề hoàn toàn khác nhau nên ít liên quan về ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Cosine similarity chỉ quan tâm đến hướng của vector nên phản ánh tốt sự tương đồng ngữ nghĩa. Euclidean distance bị ảnh hưởng bởi độ lớn vector, làm giảm hiệu quả khi so sánh embeddings.*

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Bước nhảy (stride) = 500 - 50 = 450  
> Số chunks = ceil((10000 - 500) / 450) + 1  
> = ceil(9500 / 450) + 1  
> = 22 + 1  
> = 23 chunks  
>
> **Đáp án:** 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100, stride giảm còn 400 nên số chunks tăng lên thành 25. Overlap lớn hơn giúp giữ được ngữ cảnh giữa các chunk, giảm nguy cơ mất thông tin ở ranh giới chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Research Paper

**Tại sao nhóm chọn domain này?**

> Nhóm chọn domain này vì các bài báo khoa học chứa nhiều thông tin chuyên sâu, có cấu trúc rõ ràng và phù hợp để đánh giá chất lượng của hệ thống RAG. Ngoài ra, các research paper thường đi kèm metadata như tiêu đề, năm xuất bản, DOI và nhà xuất bản, giúp thử nghiệm cả semantic retrieval và metadata filtering.

### Data Inventory

| # | Tên tài liệu                                                                                                            | Nguồn                  | Số ký tự | Metadata đã gán                     |
| - | ----------------------------------------------------------------------------------------------------------------------- | ---------------------- | -------- | ----------------------------------- |
| 1 | Cooperative integration of spatially resolved multi-omics data with COSMOS                                              | Nature Communications  | 42387 | title, year, publisher, doi, domain |
| 2 | COSMOS: a platform for real-time morphology-based, label-free cell sorting using deep learning                          | Communications Biology | 49307  | title, year, publisher, doi, domain |
| 3 | SMOPCA: spatially aware dimension reduction integrating multi-omics improves the efficiency of spatial domain detection | Genome Biology         | 86870  | title, year, publisher, doi, domain |
| 4 | Spatial epigenome–transcriptome co-profiling of mammalian tissues                                                       | Nature                 | 76242  | title, year, publisher, doi, domain |
| 5 | Deciphering spatial domains from spatial multi-omics with SpatialGlue                                                   | Nature Methods         | 49307  | title, year, publisher, doi, domain |


### Metadata Schema

| Trường metadata | Kiểu    | Ví dụ giá trị                                                           | Tại sao hữu ích cho retrieval?                                                                                                            |
| --------------- | ------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `title`         | String  | "Deciphering spatial domains from spatial multi-omics with SpatialGlue" | Giúp xác định chính xác paper khi truy vấn theo tên hoặc tham chiếu đến một nghiên cứu cụ thể.                                            |
| `year`          | Integer | `2024`                                                                  | Cho phép lọc theo thời gian, ví dụ tìm các nghiên cứu mới nhất hoặc các paper xuất bản trong một năm cụ thể.                              |
| `publisher`     | String  | "Nature Methods"                                                        | Hỗ trợ truy vấn theo tạp chí hoặc nhà xuất bản, giúp so sánh chất lượng và nguồn gốc của nghiên cứu.                                      |
| `doi`           | String  | "https://doi.org/10.1038/s41592-024-02316-4"                                            | Cung cấp định danh duy nhất cho mỗi bài báo, giúp truy xuất chính xác tài liệu và tránh nhầm lẫn giữa các nghiên cứu có tiêu đề tương tự. |
| `domain`        | String  | "Spatial Multi-omics"                                                   | Cho phép giới hạn tìm kiếm trong một lĩnh vực nghiên cứu cụ thể, giúp tăng độ liên quan của kết quả retrieval.                            |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Document  | Strategy         | Chunk Count | Avg Length | Preserves Context? |
| --------- | ---------------- | ----------: | ---------: | ------------------ |
| paper1.md | FixedSizeChunker |         336 |     199.65 | No                 |
| paper1.md | SentenceChunker  |         177 |     283.08 | Yes                |
| paper1.md | RecursiveChunker |         331 |     150.31 | Yes                |
| paper2.md | FixedSizeChunker |         393 |     199.67 | No                 |
| paper2.md | SentenceChunker  |         226 |     259.25 | Yes                |
| paper2.md | RecursiveChunker |         395 |     147.32 | Yes                |
| paper3.md | FixedSizeChunker |         700 |     199.87 | No                 |
| paper3.md | SentenceChunker  |         448 |     232.65 | Yes                |
| paper3.md | RecursiveChunker |         688 |     150.81 | Yes                |


### Strategy Của Tôi

**Loại:** SentenceChunker

**Mô tả cách hoạt động:**

SentenceChunker chia tài liệu dựa trên ranh giới câu thay vì số ký tự cố định. Thuật toán sử dụng các dấu câu như dấu chấm, dấu hỏi và dấu chấm than để xác định điểm tách tự nhiên của văn bản. Các câu liên tiếp được gộp lại cho đến khi đạt ngưỡng kích thước chunk mong muốn. Nhờ đó, mỗi chunk thường chứa các ý hoàn chỉnh và tránh việc cắt ngang nội dung quan trọng.

**Tại sao tôi chọn strategy này cho domain nhóm?**

Domain của nhóm là các bài báo khoa học về Spatial Omics và Spatial Multi-omics, trong đó thông tin quan trọng thường được trình bày dưới dạng các câu mô tả phương pháp, dữ liệu và kết quả nghiên cứu. SentenceChunker giúp giữ nguyên các phát biểu khoa học hoàn chỉnh, từ đó giảm mất mát ngữ cảnh khi thực hiện retrieval. Kết quả thực nghiệm cũng cho thấy strategy này tạo ít chunk hơn nhưng mỗi chunk chứa nhiều thông tin hơn so với các baseline khác.

**Code snippet (nếu custom):**

```python
# Không áp dụng vì sử dụng SentenceChunker có sẵn
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu  | Strategy                         | Chunk Count | Avg Length | Retrieval Quality? |
| --------- | -------------------------------- | ----------- | ---------- | ------------------ |
| paper1.md | RecursiveChunker (best baseline) | 331         | 150.31     | TBD                |
| paper1.md | **SentenceChunker (của tôi)**    | 177         | 283.08     | TBD                |
| paper2.md | RecursiveChunker (best baseline) | 395         | 147.32     | TBD                |
| paper2.md | **SentenceChunker (của tôi)**    | 226         | 259.25     | TBD                |
| paper3.md | RecursiveChunker (best baseline) | 688         | 150.81     | TBD                |
| paper3.md | **SentenceChunker (của tôi)**    | 448         | 232.65     | TBD                |

**Nhận xét:**

SentenceChunker tạo ít chunk hơn đáng kể so với RecursiveChunker, đồng thời độ dài trung bình của mỗi chunk lớn hơn. Điều này giúp mỗi chunk chứa nhiều ngữ cảnh hơn và giảm hiện tượng phân mảnh thông tin. Trong các tài liệu khoa học, việc giữ nguyên các câu hoàn chỉnh giúp bảo toàn ý nghĩa của mô tả phương pháp, kết quả và kết luận nghiên cứu, từ đó hỗ trợ retrieval hiệu quả hơn.

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?*

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*

### Test Results

```
============================================== test session starts ===============================================
platform linux -- Python 3.13.12, pytest-9.0.3, pluggy-1.5.0 -- /home/hanle/miniconda3/bin/python3.13
cachedir: .pytest_cache
rootdir: /media/hanle3012/01DC017E24BCC3D0/2A202600790-LeSyHan-Day07
plugins: anyio-4.10.0
collected 42 items                                                                                               

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                      [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                               [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                        [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                         [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                              [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED              [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                    [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                     [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                   [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                     [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                     [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                            [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                      [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED             [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                 [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED           [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                 [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                     [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                       [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                         [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                               [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                    [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                      [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED          [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                       [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                               [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                          [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                      [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                 [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                     [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                           [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                     [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED  [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED               [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED   [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED              [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED       [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

=============================================== 42 passed in 0.30s ===============================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | high / low | | |
| 2 | | | high / low | | |
| 3 | | | high / low | | |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query                                                                                                                                                     | Gold Answer (câu trả lời đúng)                                                                                                                                                                                                                                                                      |
| - | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | COSMOS (graph neural network for spatial multi-omics integration) sử dụng những thành phần chính nào để tích hợp hai modality và tạo embedding cuối cùng? | COSMOS mã hóa từng modality bằng hai Graph Convolutional Networks (GCN), sau đó tích hợp các biểu diễn bằng Weighted Nearest Neighbor (WNN), sử dụng cơ chế contrastive learning dựa trên Deep Graph Infomax (DGI) cùng spatial regularization để tạo integrated embedding cho downstream analysis. |
| 2 | Theo bài báo SMOPCA, hạn chế chính của SpatialGlue khi xử lý spatial multi-omics với nhiều modality là gì?                                                | SpatialGlue chỉ xử lý tối đa hai modality; phiên bản SpatialGlue_3M được mở rộng cho đúng ba modality nhưng vẫn không thể xử lý dữ liệu có từ bốn modality trở lên.                                                                                                                                 |
| 3 | Trong nghiên cứu Spatial epigenome-transcriptome co-profiling of mammalian tissues, spatial ATAC-RNA-seq được áp dụng trên những loại mô nào?             | Spatial ATAC-RNA-seq được áp dụng trên phôi chuột E13 (mouse embryo), não chuột hậu sinh P21/22 (juvenile mouse brain), và hồi hải mã não người trưởng thành (adult human brain hippocampus).                                                                                                       |
| 4 | COSMOS (cell sorting platform) sử dụng kiến trúc deep learning nào để phân loại tế bào từ ảnh brightfield độ phân giải cao?                               | COSMOS sử dụng kiến trúc **InceptionV3** làm mô hình CNN để học embedding hình thái học và thực hiện phân loại tế bào theo thời gian thực.                                                                                                                                                          |
| 5 | (Metadata filtering) Trong các bài báo được cung cấp, bài nào được xuất bản trực tuyến sớm nhất và vào ngày nào?                                          | Bài **“Spatial epigenome-transcriptome co-profiling of mammalian tissues”** là bài được xuất bản trực tuyến sớm nhất, vào ngày **15 March 2023**. Để trả lời chính xác cần lọc theo metadata ngày xuất bản của các tài liệu.                                                                        |


### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
