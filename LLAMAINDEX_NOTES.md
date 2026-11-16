# LlamaIndex — Study Notes

> Companion notes for the notebooks in this repo (llama-index-core 0.14.x).
> Read a section, then run the notebook it points to.

---

## 0. The one-paragraph version

LlamaIndex is a framework for **connecting an LLM to your own data**. The main pattern is
**RAG (Retrieval-Augmented Generation)**: when a question comes in, find the few relevant
pieces of your data, put them in the prompt, and let the LLM answer *from that context*.
LlamaIndex handles the full path: **load → chunk → embed → store → retrieve → post-process → synthesize**.

```
             INDEXING (offline, once)                          QUERYING (online, per question)
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐     ┌───────────┐   ┌───────────────┐   ┌────────────┐
│  Files / │──▶│ Documents │──▶│  Nodes   │──▶│  Index   │◀────│ Retriever │──▶│ Postprocessor │──▶│ Synthesizer│──▶ Response
│  APIs    │   │ (Reader)  │   │ (Parser) │   │ + Vector │     │ (top-k)   │   │ (rerank/filter)│  │ (LLM)      │   (+ source_nodes)
└──────────┘   └───────────┘   └──────────┘   │  Store   │     └───────────┘   └───────────────┘   └────────────┘
                                   │embed     └──────────┘          ▲
                                   ▼                                │
                             Embedding model ───────────────────────┘ (query is embedded too)
```

**Query Engine** = Retriever + Postprocessors + Response Synthesizer, wrapped up together.

---

## 1. Core vocabulary (learn these first)

| Term | What it is | Analogy |
|---|---|---|
| **Document** | A whole piece of source data (a PDF, a page, a row) with `text` + `metadata` | A book |
| **Node** (`TextNode`) | A chunk of a Document; the unit that gets embedded and retrieved. Carries metadata + relationships | A page of the book |
| **Reader** | Loads data into Documents (`SimpleDirectoryReader`, plus 300+ connectors on LlamaHub) | The librarian |
| **Node Parser / Splitter** | Turns Documents into Nodes (`SentenceSplitter`, `TokenTextSplitter`, `SemanticSplitterNodeParser`) | The scissors |
| **Embedding model** | Text → vector (`OpenAIEmbedding`, `HuggingFaceEmbedding`) | The coordinate system |
| **Index** | Data structure over Nodes that supports retrieval. Most common: `VectorStoreIndex` | The card catalogue |
| **Vector Store** | Where vectors physically live (in-memory `SimpleVectorStore`, Chroma, Qdrant, …) | The shelf |
| **StorageContext** | Bundles docstore + index store + vector store; used to persist/load | The building |
| **Retriever** | Given a query, returns `NodeWithScore` list | The search |
| **Node Postprocessor** | Edits the retrieved list: rerank, filter by score, etc. | The second opinion |
| **Response Synthesizer** | Builds prompts from nodes + query, calls the LLM | The writer |
| **Query Engine** | End-to-end question → `Response` | The help desk |
| **Settings** | Global defaults (LLM, embed model, chunk size) | The config file |

---

## 2. Setup & `Settings` → *Notebook 1*

LlamaIndex is **modular**: `llama-index-core` + one small package per integration.

```bash
pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai
# integrations: llama-index-vector-stores-qdrant, llama-index-vector-stores-chroma,
#               llama-index-embeddings-huggingface, llama-index-postprocessor-cohere-rerank ...
```

Import rule of thumb: **core things come from `llama_index.core`**, integrations from
`llama_index.<type>.<provider>`.

```python
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-6.1-terra", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1536)
Settings.chunk_size = 1024
Settings.chunk_overlap = 200
```

- `Settings` is a **global singleton**. Anything you don't pass explicitly falls back to it.
- You can override per component: `index.as_query_engine(llm=other_llm)`.
- ⚠️ `ServiceContext` is the **old** API. It was removed and now raises in core 0.14. If a
  tutorial online uses it, translate it to `Settings`.
- ⚠️ The **same embedding model** must be used at index time and query time, or the vectors
  can't be compared.

### Minimum working RAG (5 lines)

```python
docs = SimpleDirectoryReader("data/sample_docs").load_data()
index = VectorStoreIndex.from_documents(docs)        # chunk + embed + store
qe = index.as_query_engine(similarity_top_k=2)
resp = qe.query("What is RAG?")
print(resp)                                         # the answer
for n in resp.source_nodes: print(n.score, n.node.get_content()[:100])  # the evidence
```

`from_documents()` is doing a lot for you: it runs the node parser, embeds every node, and
writes the vectors into an in-memory store.

---

## 3. Documents, Nodes & Chunking → *Notebook 2*

### Loading
```python
reader = SimpleDirectoryReader(
    input_dir="data/research_papers",
    required_exts=[".pdf"],
    recursive=True,
    file_metadata=lambda path: {"source": path, "dept": "research"},  # custom metadata
)
docs = reader.load_data()
# or build manually: Document(text="...", metadata={"author": "x"})
```

### Chunking strategies

| Splitter | Splits on | Use when |
|---|---|---|
| `SentenceSplitter` (default) | Token budget, but respects sentence boundaries | General default |
| `TokenTextSplitter` | Pure token count | You need strict, predictable sizes |
| `SemanticSplitterNodeParser` | Embedding similarity jumps between sentences (`breakpoint_percentile_threshold`) | Topic-shifting text; costs embeddings at ingest time |
| Hierarchical / small-to-big | Multiple sizes linked parent↔child | Retrieve small, give the LLM big context (see §6) |

**Chunk size trade-off**
- **Small (128–512)**: precise retrieval, but each chunk carries little context → answers may be incomplete.
- **Large (1024–2048)**: rich context, but embeddings get "blurry" and you pay more tokens per query.
- **Overlap (~10–20%)** keeps a sentence that spans a boundary from being cut in half.
- Don't guess. **Measure** it (see the bonus eval notebook, §8).

### Metadata: the underrated feature
- Metadata on a Document is **inherited by its Nodes**.
- By default, metadata is **included in the text that gets embedded and sent to the LLM**. Control that with
  `excluded_embed_metadata_keys` / `excluded_llm_metadata_keys`.
- It enables **filtering** at query time (year ≥ 2020, category = X).
- **LLM-based extractors** (`TitleExtractor`, `SummaryExtractor`, `QuestionsAnsweredExtractor`, `KeywordExtractor`)
  add richer metadata but cost one LLM call per node or batch.

### Node relationships
Each node can point to `SOURCE` (parent doc), `PREVIOUS`, `NEXT`, `PARENT`, `CHILD` via
`NodeRelationship` + `RelatedNodeInfo`. Advanced retrievers use these to expand context.

### IngestionPipeline: the production way to ingest
```python
from llama_index.core.ingestion import IngestionPipeline
pipeline = IngestionPipeline(transformations=[
    SentenceSplitter(chunk_size=1024, chunk_overlap=200),
    TitleExtractor(),               # optional
    Settings.embed_model,           # last step: embed
])
nodes = pipeline.run(documents=docs)
index = VectorStoreIndex(nodes)     # build index from pre-made nodes
```
Benefits: you control each step, it can **cache** transformations, it can write straight into a
vector store, and with a docstore attached it can **deduplicate / upsert** changed documents.

---

## 4. Indexes, Vector Stores & Persistence → *Notebook 3*

### Plugging in a real vector store
```python
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

col = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection("docs")
storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=col))
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

# Later: reconnect without re-embedding
index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=col))
```

| Store | Good for |
|---|---|
| `SimpleVectorStore` (default, in-memory) | Prototyping, < ~10k nodes |
| Chroma | Local dev, simple persistence |
| Qdrant | Production, filtering, **hybrid search**, scale |
| (also pgvector, Pinecone, Weaviate, Milvus, …) | Whatever your infra already runs |

### Persisting the default in-memory index
```python
index.storage_context.persist(persist_dir="./storage")

from llama_index.core import load_index_from_storage
index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))
```
Persisting saves you from paying for embeddings again on every restart.

### Embedding models
- **OpenAI `text-embedding-3-small`**: strong and cheap, API call, `dimensions` can be reduced (e.g. 512) for speed.
- **HuggingFace local** (`BAAI/bge-small-en-v1.5`, etc.): free, private, runs offline; good quality for its size.
- Switching models means you must **re-index everything**.

---

## 5. Querying: retrievers, query engines, response modes → *Notebooks 1 & 3*

### Two ways to build a query engine
```python
# High-level
qe = index.as_query_engine(similarity_top_k=3, response_mode="compact", streaming=False)

# Low-level (full control over each part)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
qe = RetrieverQueryEngine.from_args(retriever=retriever, response_mode="compact",
                                    node_postprocessors=[...])
```

Retrieve only, with no LLM call (useful for debugging):
```python
nodes = index.as_retriever(similarity_top_k=5).retrieve("my question")
```

### Response modes (how retrieved chunks become an answer)

| Mode | How it works | LLM calls | Use when |
|---|---|---|---|
| `compact` (default) | Stuff as many chunks as fit per prompt, refine across prompts | Few | Default |
| `refine` | One chunk at a time; each call improves the previous answer | = #chunks | Thorough answers, high cost |
| `tree_summarize` | Summarize chunks, then summarize the summaries | log-ish | Summarization questions over many chunks |
| `simple_summarize` | Truncate everything into one prompt | 1 | Quick, may lose info |
| `accumulate` | Answer per chunk, concatenate | = #chunks | Want per-source answers |
| `no_text` | Retrieve only, no LLM | 0 | Debugging retrieval |

### `similarity_top_k`
Low k (1–2) is precise but can miss information. High k (8–10) has better recall, but adds noise and cost.
A common production pattern is to **retrieve 10, rerank, then keep 3** (see §7).

### Streaming
```python
qe = index.as_query_engine(streaming=True)
qe.query("...").print_response_stream()   # or iterate response.response_gen
```

### Auto-retriever (natural-language → metadata filters)
`VectorIndexAutoRetriever` + a `VectorStoreInfo` describing your metadata fields. The LLM
turns *"papers about transformers after 2019"* into `query="transformers"`, `filter year > 2019`.

---

## 6. Advanced retrieval → *Notebook 4*

| Technique | Idea | Key class |
|---|---|---|
| **Recursive / small-to-big** | Index small chunks or summaries as `IndexNode`s that point to bigger content; retrieve the small one, follow the link | `RecursiveRetriever` |
| **Query fusion** | Run several retrievers (or several LLM-rewritten queries), merge the results with **Reciprocal Rank Fusion** | `QueryFusionRetriever(mode="reciprocal_rerank", num_queries=…)` |
| **Custom retriever** | Subclass `BaseRetriever`, implement `_retrieve(query_bundle) -> List[NodeWithScore]` | `BaseRetriever` |
| **HyDE** | LLM writes a *hypothetical answer*; embed that instead of the short question (answers look more like documents than questions do) | `HyDEQueryTransform` + `TransformQueryEngine` |
| **Metadata filters** | Structured pre-filtering with AND/OR | `MetadataFilters`, `MetadataFilter`, `FilterOperator`, `FilterCondition` |

```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator, FilterCondition
filters = MetadataFilters(filters=[
    MetadataFilter(key="year", operator=FilterOperator.GTE, value=2017),
    MetadataFilter(key="category", operator=FilterOperator.EQ, value="transformers"),
], condition=FilterCondition.AND)
retriever = index.as_retriever(similarity_top_k=5, filters=filters)
```

**RRF in one line:** `score(doc) = Σ 1 / (k + rank_i(doc))`. It rewards documents that rank well
in *several* retrievers and ignores raw score scales, so it can merge BM25 scores and cosine scores safely.

**HyDE trade-off:** it adds 1 LLM call of latency. It helps with short or vague queries and can
hurt factual lookups if the hypothetical answer is wrong.

---

## 7. Hybrid search, reranking, production → *Notebook 5*

### Dense vs sparse
- **Dense** (embeddings): understands meaning ("car" ≈ "automobile"), but is weak on exact terms, IDs, and rare jargon.
- **Sparse** (BM25 / SPLADE): exact keyword matching, good for codes, names, and acronyms, but has no notion of synonyms.
- **Hybrid** = both, fused. `alpha` controls the mix (Qdrant: `1.0` = pure dense, `0.0` = pure sparse).

```python
vector_store = QdrantVectorStore(client=client, collection_name="docs", enable_hybrid=True)
index = VectorStoreIndex.from_documents(docs,
            storage_context=StorageContext.from_defaults(vector_store=vector_store))
qe = index.as_query_engine(similarity_top_k=3, vector_store_query_mode="hybrid", alpha=0.5)
```
Tune `alpha` on your own eval set. Keyword-heavy domains (legal, code, product SKUs) usually want a lower alpha.

### Reranking (two-stage retrieval)
Bi-encoder retrieval is fast but rough. A **cross-encoder** reads (query, chunk) *together*, which makes it
much more accurate and much slower, so you only run it on the top-N candidates.

```python
from llama_index.core.postprocessor import SentenceTransformerRerank
reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3)
# or: from llama_index.postprocessor.cohere_rerank import CohereRerank
#     reranker = CohereRerank(model="rerank-v3.5", top_n=3)
qe = index.as_query_engine(similarity_top_k=10, node_postprocessors=[reranker])
```
This is the **retrieve wide (10–20), rerank narrow (3–5)** pattern, one of the cheapest quality wins in RAG.

### Custom query engine
Subclass `BaseQueryEngine` and implement `_query`, `_aquery`, and (**required in 0.14**)
`_get_prompt_modules()`. The notebook's `ProductionQueryEngine` wraps a base engine with:
- input validation + try/except with a graceful fallback answer
- metrics (query count, error count, latency)
- an LRU cache keyed on a hash of the query

### Production checklist (condensed)
- [ ] Persistent vector store (not in-memory) + ingestion pipeline with dedup/upsert
- [ ] Same embedding model at index and query time, with the version pinned
- [ ] Hybrid search if users type exact terms; reranker on top
- [ ] Return `source_nodes` and show citations to users
- [ ] Cache frequent queries; use a cheap LLM for generation and a stronger one for eval
- [ ] Log latency, token usage, retrieval scores; alert on error rate
- [ ] An **evaluation set** that you re-run on every change (next section)

---

## 8. Evaluation → *Notebook 4 §8 + bonus chunk-size notebook*

Evaluate **retrieval** and **generation** separately, because they fail for different reasons.

**Retrieval metrics** (need a set of (question, relevant node id) pairs):
- **Hit Rate**: was a relevant node anywhere in the top-k?
- **MRR**: `1 / rank` of the first relevant node, averaged. Rewards ranking it *first*.
```python
from llama_index.core.evaluation import RetrieverEvaluator, HitRate, MRR
ev = RetrieverEvaluator(metrics=[HitRate(), MRR()], retriever=index.as_retriever(similarity_top_k=3))
```

**Response metrics** (LLM-as-judge, use a strong judge model):
- **Faithfulness**: is the answer supported by the retrieved context? (Hallucination check.)
- **Relevancy**: do the answer and context actually address the question?
```python
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
judge = OpenAI(model="gpt-6.1-sol", temperature=0)
faith = FaithfulnessEvaluator(llm=judge).evaluate_response(response=resp)
print(faith.passing, faith.score)
```
You can generate eval questions automatically with `DatasetGenerator` / `generate_question_context_pairs`.

**Chunk-size experiment recipe** (bonus notebook): for each size in `[128, 256, 512, 1024, 2048]`,
rebuild the index, then run the same questions and record avg response time, faithfulness, and relevancy.
Pick the size with the best quality/latency balance. Often 512–1024 wins, but it depends on your data.

---

## 9. Debugging mental model: "my RAG answer is bad"

```
Bad answer?
 ├─ Look at response.source_nodes first.
 │   ├─ Right chunks NOT retrieved → retrieval problem
 │   │     • chunk size wrong? • need hybrid (exact terms)? • raise top_k + rerank?
 │   │     • metadata filters too strict? • query too short → HyDE / query fusion
 │   └─ Right chunks retrieved, answer still wrong → synthesis problem
 │         • too much noise → rerank / lower top_k • answer split across chunks → refine/tree_summarize
 │         • weak LLM / bad prompt → customize prompt templates, stronger model
 └─ Always measure: Hit Rate/MRR for retrieval, Faithfulness/Relevancy for answers.
```

---

## 10. Cheat sheet

```python
# --- config
Settings.llm, Settings.embed_model, Settings.chunk_size, Settings.chunk_overlap, Settings.node_parser

# --- load
docs = SimpleDirectoryReader("dir", required_exts=[".pdf"]).load_data()

# --- chunk / ingest
nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(docs)
nodes = IngestionPipeline(transformations=[splitter, embed_model]).run(documents=docs)

# --- index
index = VectorStoreIndex.from_documents(docs)                     # or VectorStoreIndex(nodes)
index = VectorStoreIndex.from_documents(docs, storage_context=sc) # external vector store
index.storage_context.persist("./storage"); load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))

# --- retrieve / query
index.as_retriever(similarity_top_k=5, filters=MetadataFilters(...)).retrieve(q)
index.as_query_engine(similarity_top_k=10, response_mode="compact",
                      node_postprocessors=[reranker], streaming=False).query(q)

# --- inspect
resp.response, resp.source_nodes, n.score, n.node.metadata, n.node.get_content()
```

---

## 11. Self-check questions

1. What is the difference between a `Document` and a `Node`? Why not embed whole documents?
2. What happens if you index with embedding model A and query with model B?
3. When would you choose `tree_summarize` over `compact`?
4. Why does "retrieve 10, rerank to 3" beat "retrieve 3"?
5. A user searches for error code `E-4021` and gets irrelevant results. What do you change?
6. What does RRF fix that a plain score average doesn't?
7. Your faithfulness score is high but relevancy is low. Where is the problem?
8. What problem does HyDE solve, and when can it backfire?
9. Why should metadata like `file_path` usually be excluded from embeddings?
10. What replaced `ServiceContext`, and what does a missing `_get_prompt_modules()` break?

<details><summary>Short answers</summary>

1. A Document is the whole source; Nodes are the chunks. Embedding a whole document produces one blurry vector and an oversized context.
2. The vectors live in different spaces, so similarity search returns garbage. Re-index with the new model.
3. When the answer needs a synthesis or summary over many chunks.
4. A cross-encoder is more accurate than a bi-encoder, and wide retrieval gives it more chances to find the right chunk.
5. Add hybrid/BM25 (lower alpha). Dense embeddings handle exact codes poorly.
6. Different retrievers produce scores on incompatible scales. RRF uses only ranks.
7. Retrieval: the answer is grounded in context that doesn't address the question.
8. Short or vague queries don't look like documents. It backfires when the hypothetical answer is factually wrong and pulls in the wrong chunks.
9. It adds noise to the vector without adding meaning. Keep it for the LLM or filtering only.
10. `Settings`. A custom `BaseQueryEngine` subclass without `_get_prompt_modules()` can't be instantiated in 0.14 (it is an abstract method).
</details>

---

## 12. Suggested study order

1. Notebook 1: run it, then re-run with `similarity_top_k` = 1 vs 5 and compare `source_nodes`.
2. Notebook 2: compare the three splitters on one PDF and print node counts and sample chunks.
3. Notebook 3: persist an index, restart the kernel, reload it. Try Chroma.
4. Bonus notebook: run the chunk-size evaluation on *your own* documents.
5. Notebook 4: fusion + HyDE, measured with Hit Rate/MRR.
6. Notebook 5: add a reranker, then hybrid. Benchmark before and after.
7. Capstone: build one end-to-end QA system over your own docs with eval numbers.
