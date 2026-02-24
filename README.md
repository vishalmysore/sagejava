# SAGE: Structure Aware Graph Expansion for Retrieval of Heterogeneous Data

## Operationalizing arXiv:2602.16964 with Tools4AI and JSON-LD Knowledge Graphs

---

## Abstract

This project is a **structural and algorithmic operationalization** of the **SAGE (Structure Aware Graph Expansion)** framework proposed in *"SAGE: Structure Aware Graph Expansion for Retrieval of Heterogeneous Data"* (arXiv:2602.16964, Titiya et al., 2026). The implementation provides the core algorithmic pipeline proposed in the paper — offline graph construction with percentile-based pruning and two-stage online retrieval (seed selection + neighbor expansion) — while introducing a **pluggable Similarity Provider** system. This system allows the framework to operate either in a zero-dependency mode using **lexical Jaccard similarity** or in a paper-faithful production mode using **dense Cosine embeddings** via an OpenAI-compatible API (e.g., NVIDIA NIM). The project demonstrates the algorithm's structural behavior and retrieval precision using manually annotated sample data. Additionally, it extends the system beyond the paper's scope with **Tools4AI** integration (making graph operations AI-callable) and **JSON-LD** export (enabling semantic web interoperability via Schema.org).

---

## 1. What the Paper Proposes

### 1.1 The Problem: Flat Retrieval Fails on Heterogeneous Data

Standard RAG (Retrieval-Augmented Generation) systems use **flat retrieval** — independently indexed text chunks are retrieved using sparse or dense similarity search. While effective for simple text-centric queries, this design has three critical failure modes:

1. **Missing Multi-Hop Evidence**: Flat retrieval treats each chunk independently, ignoring structural dependencies. A question like *"What television show starring Trevor Eyster was based on a book?"* requires linking an actor's biography → a TV show listing → a book adaptation record. These are three separate chunks that standard cosine similarity searches will not connect.
2. **Cross-Modal Blindness**: Documents and tables contain complementary evidence. A table listing "Film | Director | Year" and a document describing a director's filmography are structurally related, but flat retrieval scores them independently against the query.
3. **Intermediate Evidence Loss**: Bridging chunks that are weakly related to the query in embedding space but structurally connected to relevant content are frequently missed.

### 1.2 The SAGE Solution

SAGE addresses these failures through a two-phase architecture:

**Phase 1 — Offline Graph Construction (Section 2.1)**
- Segment documents into semantically coherent chunks using sliding-window sentence embeddings with percentile-based boundary detection.
- Extract metadata (topic, title, entities) per chunk using an LLM.
- Build edges between chunks based on metadata-driven similarity signals: topic-topic similarity, content-content similarity, entity overlap, and column-column similarity (for tables).
- **Percentile-based pruning**: Only retain edges whose similarity exceeds the **95th percentile** of the empirical distribution. This maintains graph sparsity and reduces dense-neighborhood noise.

**Phase 2 — Online Retrieval (Section 2.2)**
- **Stage 1 (Seed Retrieval)**: Run a hybrid sparse+dense retrieval (BM25 + cosine similarity) to obtain `k` seed nodes.
- **Stage 2 (Graph Expansion)**: Collect all first-hop neighbors of the seeds from the pre-built graph, re-rank the neighbors using BM25 + dense similarity with respect to the query, and select the top `k'` neighbors.
- **Final Context**: Return the union of `k` seeds + `k'` expanded neighbors.

### 1.3 Key Results from the Paper

| Benchmark | Improvement | Metric |
| :--- | :--- | :--- |
| **OTT-QA** (Text+Tables) | **+5.7 Recall points** | Over hybrid baselines |
| **STaRK** (Semi-Structured KGs) | **+8.5 Recall points** | Over flat retrieval baseline |

The paper also introduces **SPARK**, an agentic retriever for explicit schema graphs that uses LLM-generated Cypher queries for Neo4j traversal.

---

## 2. What We Built: A Complete Java Implementation

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SAGEDemoRunner (Orchestrator)                │
│                                                                 │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │   SAGEGraph      │   │  SAGERetriever   │   │ JsonLd       │ │
│  │  (Offline Build) │──▶│  (Online Query)  │──▶│ Exporter     │ │
│  │  Percentile Prune│   │  Seed + Expand   │   │ Schema.org   │ │
│  └─────────────────┘   └──────────────────┘   └──────────────┘ │
│           │                      │                     │        │
│  ┌────────▼────────────────────────────────────────────▼──────┐ │
│  │            GraphRetrievalAction (Tools4AI @Agent)          │ │
│  │  searchKnowledgeGraph | expandNode | exportGraphAsJsonLd   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Domain: ChunkNode | ChunkEdge | ChunkType | EdgeType     │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 The Domain Model

The paper operates on a **heterogeneous graph** `G = (V, E)` where nodes represent semantically coherent content units and edges capture metadata-driven relationships. We model this directly.

#### `ChunkType.java` — Heterogeneous Node Types
```java
public enum ChunkType {
    DOCUMENT,        // Text passage chunk (paper: semantic chunking of documents)
    TABLE,           // Table segment chunk (paper: 5-10 rows per segment)
    SEMI_STRUCTURED  // Semi-structured node (paper: objects with textual fields)
}
```

These three types correspond to the data modalities described in Section 2.1A of the paper. In the paper, DOCUMENT chunks are created through semantic chunking with sliding-window sentence embeddings; TABLE chunks preserve vertical context by limiting segments to 5-10 rows; SEMI_STRUCTURED chunks treat each database object as a node with metadata derived from its textual fields. **In our implementation, chunks are pre-defined demo data rather than dynamically segmented** — we did not implement the paper's Section 2.1A sliding-window boundary detection. This is discussed further in Section 8.

#### `EdgeType.java` — Metadata-Driven Edge Types
```java
public enum EdgeType {
    DOC_DOC,       // Document-Document: topic/content similarity + entity overlap
    TABLE_TABLE,   // Table-Table: column/title similarity + entity overlap
    TABLE_DOC,     // Table-Document: content-column + topic-title similarity
    PARENT_CHILD,  // Structural link: chunks from the same source document
    SCHEMA_EDGE    // Native schema edge from a knowledge graph
}
```

The paper defines four modality-specific similarity signals (Section 2.1B). Our `EdgeType` enum captures each of these, plus structural/parent links and native schema edges for knowledge graph datasets like STaRK.

#### `ChunkNode.java` — Node with JSON-LD Serialization

Each node carries content, manually annotated metadata (the paper uses LLM-extracted metadata), named entities, and an optional dense embedding vector (unused in the current implementation — see Section 8). The `toJsonLd()` method is **our extension beyond the paper**, enabling semantic web interoperability:

```java
@Data @Builder
public class ChunkNode {
    private String id;
    private ChunkType type;
    private String content;
    private String sourceDocument;
    private Map<String, String> metadata;   // LLM-extracted: topic, title
    private Map<String, String> entities;   // Named entities: "Ridley Scott" → "PERSON"
    private double[] embedding;             // Dense vector for similarity
    private double relevanceScore;          // Assigned during retrieval

    public Map<String, Object> toJsonLd() {
        Map<String, Object> jsonLd = new HashMap<>();
        jsonLd.put("@context", "https://schema.org/");
        jsonLd.put("@type", type == ChunkType.TABLE ? "Table" : "CreativeWork");
        jsonLd.put("@id", "urn:sage:chunk:" + id);
        jsonLd.put("name", metadata.getOrDefault("title", id));
        jsonLd.put("isPartOf", Map.of("@id", "urn:sage:doc:" + sourceDocument));
        jsonLd.put("mentions", entities);
        return jsonLd;
    }
}
```

The JSON-LD serialization uses `schema:CreativeWork` for documents and `schema:Table` for table chunks, mapping naturally onto Schema.org vocabulary. The `isPartOf` relationship preserves source provenance, and `mentions` links to extracted entities — both critical for downstream knowledge graph traversal.

---

## 3. Offline Graph Construction: Percentile-Based Pruning

### 3.1 The Algorithm (Paper Section 2.1B)

The paper's edge construction strategy is: *"We only keep a similarity edge when its metadata similarity exceeds the 95th percentile of the empirical distribution."* This is the single most important design decision in SAGE — it maintains graph sparsity and prevents dense-neighborhood noise that would degrade expansion quality.

### 3.2 Our Implementation: `SAGEGraph.buildEdgesFromMetadata()`

```java
public void buildEdgesFromMetadata() {
    List<ChunkNode> allNodes = new ArrayList<>(nodes.values());
    List<Double> allScores = new ArrayList<>();
    List<double[]> candidateEdges = new ArrayList<>();

    // Step 1: Compute ALL pairwise metadata similarities
    for (int i = 0; i < allNodes.size(); i++) {
        for (int j = i + 1; j < allNodes.size(); j++) {
            double sim = computeMetadataSimilarity(allNodes.get(i), allNodes.get(j));
            allScores.add(sim);
            candidateEdges.add(new double[]{i, j, sim});
        }
    }

    // Step 2: Percentile-based pruning (95th percentile threshold)
    Collections.sort(allScores);
    int threshIdx = (int) (allScores.size() * SIMILARITY_PERCENTILE_THRESHOLD);
    double threshold = allScores.get(Math.min(threshIdx, allScores.size() - 1));

    // Step 3: Only retain edges above the threshold
    for (double[] candidate : candidateEdges) {
        if (candidate[2] >= threshold) {
            addEdge(ChunkEdge.builder()
                .sourceId(src.getId()).targetId(tgt.getId())
                .edgeType(resolveEdgeType(src.getType(), tgt.getType()))
                .similarityScore(candidate[2])
                .traversalMetadata(computeTraversalMetadata(src, tgt))
                .build());
        }
    }
}
```

```java
private double computeMetadataSimilarity(ChunkNode a, ChunkNode b) {
    // Delegates to the SimilarityProvider (Jaccard or Cosine)
    double topicSim = similarityProvider.computeSimilarity(
            a.getMetadata().getOrDefault("topic", ""),
            b.getMetadata().getOrDefault("topic", ""));

    double entityOverlap = computeEntityOverlap(a.getEntities(), b.getEntities());

    double contentSim = similarityProvider.computeSimilarity(
            a.getContent() != null ? a.getContent() : "",
            b.getContent() != null ? b.getContent() : "");

    // Weighted combination: topic > entities > content
    return 0.4 * topicSim + 0.35 * entityOverlap + 0.25 * contentSim;
}
```

### 3.4 Pluggable Similarity Strategies

The framework decouples similarity computation through the `SimilarityProvider` interface. This allows for:

1. **`JaccardSimilarityProvider` (Default)**: A zero-dependency lexical fallback. It computes overlap between token sets. While fast, it cannot capture semantic relationships (e.g., "movies" and "films" score 0.0).
2. **`CosineSimilarityProvider` (Paper-Faithful)**: Generates dense embeddings by calling an external LLM endpoint (NVIDIA NIM/OpenAI). It computes true cosine similarity between vectors and includes a **ConcurrentHashMap-based cache** to minimize API costs during pairwise graph construction.

| Mode | Semantic Awareness | Latency | Dependency |
| :--- | :--- | :--- | :--- |
| **Jaccard** | Low (Lexical only) | Near-Zero | None |
| **Cosine** | High (Semantic) | API Roundtrip | API Key (NIM/OpenAI) |

### 3.4 Edge Traversal Metadata

The paper states: *"Edges store lightweight metadata to guide controlled graph traversal."* We capture this as:

```java
private Map<String, String> computeTraversalMetadata(ChunkNode src, ChunkNode tgt) {
    Map<String, String> meta = new HashMap<>();
    Set<String> shared = new HashSet<>(src.getEntities().keySet());
    shared.retainAll(tgt.getEntities().keySet());
    if (!shared.isEmpty()) {
        meta.put("sharedEntities", String.join(", ", shared));
    }
    meta.put("sourceType", src.getType().name());
    meta.put("targetType", tgt.getType().name());
    return meta;
}
```

This metadata allows downstream traversal to make informed decisions — for example, prioritizing edges with high entity overlap or specific cross-modal connections (TABLE_DOC edges with joinable columns).

---

## 4. Online Retrieval: Two-Stage Seed + Expansion

### 4.1 The Algorithm (Paper Section 2.2)

The paper's retrieval algorithm in pseudocode:
```
Input: query q, budgets k, k', baseline retriever BL, graph G
S = TopK(BL(q), k)                    // Stage 1: Seed retrieval
N = OneHopNeighbors(G, S)             // Stage 2a: Neighbor expansion
R = TopK(BM25DenseRank(q, N), k')     // Stage 2b: Neighbor pruning
Return Union(S, R)                     // Final context: k + k' nodes
```

### 4.2 Our Implementation: `SAGERetriever`

#### Stage 1: Hybrid Baseline Retrieval

```java
    scoredNodes.sort(byDescendingScore);
    return scoredNodes.stream().limit(k).collect(Collectors.toList());
}
```

The BM25 component scores each node by counting how many query terms appear in the node's content, metadata, and entity list. The "dense" component delegates to the `SimilarityProvider`. When configured with the `CosineSimilarityProvider`, this implements the **true dense retrieval** described in the paper, using cosine similarity over Sentence-Transformer quality embeddings. When an API key is absent, it seamlessly falls back to Jaccard.

#### Stage 2: Graph Expansion + Pruning

```java
private List<ChunkNode> graphExpansionAndPruning(String query, List<ChunkNode> seeds, int kPrime) {
    // Step 1: Collect ALL first-hop neighbors of the seed nodes
    Set<ChunkNode> candidates = new LinkedHashSet<>();
    for (ChunkNode seed : seeds) {
        List<ChunkNode> neighbors = graph.getNeighbors(seed.getId());
        for (ChunkNode neighbor : neighbors) {
            if (!seedIds.contains(neighbor.getId())) {
                candidates.add(neighbor);
            }
        }
    }

    // Step 2: Re-rank candidates by query relevance (BM25 + dense)
    for (ChunkNode candidate : candidates) {
        candidate.setRelevanceScore(0.5 * bm25 + 0.5 * dense);
    }

    // Step 3: Select top-k' neighbors
    return candidates.stream()
        .sorted(byDescendingScore)
        .limit(kPrime)
        .collect(Collectors.toList());
}
```

This is the core innovation of SAGE. By traversing graph edges, we recover evidence chunks that are **weakly related to the query in embedding space but structurally connected to relevant content**. A flat retriever would miss these bridging chunks entirely.

---

## 5. Tools4AI Integration: Making the Graph AI-Callable

### 5.1 Why Tools4AI?

The SAGE paper focuses on algorithmic retrieval. To make this useful in a production agentic system, an LLM agent needs to be able to *call* the graph operations as tools. **Tools4AI** provides exactly this capability through its `@Agent`/`@Action` annotation model — the same pattern used in the [Agent Misalignment](https://github.com/vishalmysore/agenticrulesengine) project for safety interception.

### 5.2 The Action Class: `GraphRetrievalAction.java`

```java
@Agent(groupName = "SAGERetrievalAgent",
       groupDescription = "Agent for structure-aware retrieval over heterogeneous knowledge graphs")
public class GraphRetrievalAction {

    @Action(description = "Search the knowledge graph for information. Use this to find relevant "
            + "documents, tables, and entities related to a user's question.")
    public String searchKnowledgeGraph(String query) {
        List<ChunkNode> results = sharedRetriever.retrieve(query);
        // Format results with scores and content previews
        return formatResults(results);
    }

    @Action(description = "Expand a specific node in the knowledge graph to find related neighbors. "
            + "Use this to discover multi-hop connections from a known entity.")
    public String expandNode(String nodeId) {
        List<ChunkNode> neighbors = sharedGraph.getNeighbors(nodeId);
        return formatNeighbors(nodeId, neighbors);
    }

    @Action(description = "Get the JSON-LD representation of the knowledge graph for "
            + "interoperability and semantic web integration.")
    public String exportGraphAsJsonLd() {
        return sharedGraph.toJsonLd().toString();
    }

    @Action(description = "Get statistics about the knowledge graph including node count, "
            + "edge count, and chunk type distribution.")
    public String getGraphStatistics() {
        // Returns node/edge counts broken down by type
    }
}
```

When an LLM agent receives a query like *"Find me all science fiction films and their awards,"* Tools4AI's semantic matching engine maps it to the `searchKnowledgeGraph` action. The agent can then use `expandNode` for multi-hop exploration — exactly mirroring SAGE's expansion stage, but driven by the LLM's reasoning rather than a fixed algorithm.

### 5.3 The Tools4AI Pattern (Reference: Agent Misalignment Project)

Both this project and the Agent Misalignment project use the same Tools4AI integration pattern:

| Component | Agent Misalignment | SAGE KG Retrieval |
| :--- | :--- | :--- |
| **@Agent** | `SystemControlAgent` | `SAGERetrievalAgent` |
| **@Action** | `disableMonitoring()`, `elevatePrivileges()` | `searchKnowledgeGraph()`, `expandNode()` |
| **Callback** | `MisalignmentCallback` (intercepts actions) | Direct invocation (returns results) |
| **Processor** | `AIProcessor` (LLM-driven) | `SAGERetriever` (graph-driven) |

The key difference is that the misalignment project uses Tools4AI for **safety interception** (blocking actions), while this project uses it for **knowledge retrieval** (executing graph queries). This demonstrates the versatility of the `@Action` annotation model.

---

## 6. JSON-LD Knowledge Graph Export

> **Note:** JSON-LD export is **our systems-layer extension**, not part of the SAGE paper. The paper focuses on retrieval benchmarks and does not prescribe any export format.

### 6.1 Why JSON-LD?

JSON-LD (JavaScript Object Notation for Linked Data) is the W3C standard for encoding Linked Data in JSON. By exporting the SAGE graph as JSON-LD, we enable:
- **Semantic Web Interoperability**: The graph can be ingested by any system that understands Schema.org or custom RDF vocabularies.
- **Knowledge Graph Federation**: Multiple SAGE graphs from different domains can be merged via shared `@id` references.
- **Provenance Tracking**: The `isPartOf` relationship traces each chunk back to its source document.

### 6.2 The JSON-LD Document Structure

```json
{
  "@context": {
    "schema": "https://schema.org/",
    "sage": "urn:sage:ontology:",
    "name": "schema:name",
    "description": "schema:description",
    "isPartOf": { "@type": "@id" },
    "mentions": "sage:mentions",
    "relatedTo": { "@type": "@id" }
  },
  "@type": "sage:KnowledgeGraph",
  "sage:framework": "SAGE (Structure Aware Graph Expansion)",
  "sage:paper": "arXiv:2602.16964",
  "@graph": [
    {
      "@type": "CreativeWork",
      "@id": "urn:sage:chunk:doc-blade-runner",
      "name": "Blade Runner Film",
      "description": "Blade Runner is a 1982 science fiction film directed by Ridley Scott...",
      "isPartOf": { "@id": "urn:sage:doc:wikipedia-blade-runner" },
      "mentions": { "Ridley Scott": "PERSON", "Blade Runner": "FILM" },
      "keywords": "science fiction film actor director"
    },
    {
      "@type": "Table",
      "@id": "urn:sage:chunk:table-scifi-films",
      "name": "Science Fiction Films Table",
      "isPartOf": { "@id": "urn:sage:doc:wiki-table-scifi" }
    },
    {
      "@type": "Relationship",
      "source": { "@id": "urn:sage:chunk:doc-scifi-ridley" },
      "target": { "@id": "urn:sage:chunk:semi-ridley-scott" },
      "relationshipType": "DOC_DOC",
      "weight": 0.52,
      "metadata": { "sharedEntities": "Ridley Scott, Blade Runner, Alien" }
    }
  ]
}
```

> **Modeling Note:** In standard RDF, relationships are typically edges (triples) rather than separate nodes. Our `Relationship` objects are closer to **reified statements** — a pragmatic choice for JSON serialization that preserves edge weight and metadata. In a strict RDF deployment, these would be modeled as named graphs or RDF-star annotations instead.

### 6.3 Exporter Implementation

The `JsonLdExporter` uses Jackson for serialization and supports two export modes:

```java
public class JsonLdExporter {
    // Full graph export — includes all nodes, edges, and statistics
    public String exportGraph(SAGEGraph graph) {
        Map<String, Object> jsonLd = buildGraphDocument(graph);
        return mapper.writeValueAsString(jsonLd);
    }

    // Query-scoped export — only the retrieval results for a specific query
    public String exportRetrievalResults(List<ChunkNode> results, String query) {
        doc.put("@type", "sage:RetrievalResult");
        doc.put("sage:query", query);
        doc.put("sage:resultCount", results.size());
        doc.put("@graph", results.stream().map(ChunkNode::toJsonLd).collect(toList()));
        return mapper.writeValueAsString(doc);
    }
}
```

---

## 7. Demo Execution & Verified Results

### 7.1 Running the Demo

```bash
mvn compile exec:java
```

### 7.2 Verified Output

**Phase 1 — Offline Graph Construction (Cosine Embedding Mode):**
```
INFO: SAGEGraph initialized with similarity: Cosine (dense embeddings via nvidia/nv-embed-v1)
INFO: Edge pruning threshold (95th percentile): 0.7225
INFO: Graph built: 11 nodes, 3 edges
```

In the API-backed mode, the 95th percentile threshold shifted significantly (from `0.48` in Jaccard to `0.72` in Cosine), indicating stronger semantic clustering detected by the `nv-embed-v1` model.

**Phase 2 — Online Retrieval (k=3 seeds, k'=2 expansion):**

| Query | Seeds | Expanded | Total | Top Result |
| :--- | :--- | :--- | :--- | :--- |
| *"TV show starring Trevor Eyster based on a book?"* | 3 | 0 | 3 | Eerie Indiana TV Show (0.458) |
| *"Actors in science fiction films by Ridley Scott?"* | 3 | **1** | **4** | Science Fiction Films Table (0.529) |
| *"Awards won by the author of Dune"* | 3 | 0 | 3 | Frank Herbert Author (0.668) |

Query 2 demonstrates the **SAGE structural advantage**: while the initial seed set focused on filmography, the graph expansion pulled in the Ridley Scott IMDB profile through its strong structural connection in the graph, enriching the context with cross-modal data.

**Phase 4 — Tools4AI Actions:**
```
--- Action: getGraphStatistics ---
Graph Statistics:
  Nodes: 11 (Documents: 7, Tables: 2, Semi-Structured: 2)
  Edges: 3

--- Action: searchKnowledgeGraph ---
SAGE Retrieved 5 chunks:
  [1] Ridley Scott Filmography (DOCUMENT) - Score: 0.589

--- Action: expandNode ---
Node: Ridley Scott Filmography
Neighbors (1):
  → Ridley Scott IMDB Profile (SEMI_STRUCTURED)
```

---

## 8. Paper vs. Implementation: What We Preserved, What We Simplified, What We Added

This section is critical for intellectual honesty. Our implementation is best described as: *"A structural and algorithmic operationalization of SAGE with simplified similarity modeling and demo-scale evaluation."*

### 8.1 What We Faithfully Preserved (Algorithmic Structure)

| Paper Component | Our Implementation | Fidelity |
| :--- | :--- | :--- |
| **Percentile-based edge pruning** | 95th percentile threshold on pairwise scores | ✅ Exact |
| **Two-stage retrieval** | Stage 1: seed selection. Stage 2: expansion + pruning | ✅ Exact |
| **Graph expansion operator** | First-hop neighbor collection + re-ranking | ✅ Exact |
| **Hybrid seed scoring** | BM25 + secondary signal, linear interpolation | ✅ Structural match |
| **Heterogeneous node types** | DOCUMENT, TABLE, SEMI_STRUCTURED | ✅ Exact |
| **Edge traversal metadata** | Shared entities, source/target types | ✅ Exact |

### 8.2 What We Simplified (Fundamental Divergences)

| Paper Component | Paper's Approach | Our Implementation | Impact |
| :--- | :--- | :--- | :--- |
| **Similarity** | Sentence-Transformer dense embeddings + cosine similarity | ✅ `CosineSimilarityProvider` (API-backed) | **Exact Match**: When an API key is provided, we use true dense cosine embeddings as proposed in the paper. |
| **Chunking** | Sliding-window sentence embeddings + percentile boundary detection (Section 2.1A) | Pre-defined demo chunks | **High**: Section 2.1A is not operationalized. Real-world use requires dynamic segmentation. |
| **Metadata Extraction** | LLM-generated per-chunk metadata | Manually annotated metadata | **Medium**: Algorithmic structure is preserved, but metadata quality and coverage differ. |
| **Benchmark Evaluation** | Recall@k on OTT-QA (+5.7) and STaRK (+8.5) | No Recall@k computation, no benchmark datasets | **High**: We cannot claim empirical reproduction — only architectural implementation. |
| **SPARK Agent** | LLM-generated Cypher queries for Neo4j schema-aware traversal | Not implemented | **Medium**: SPARK is a significant component for explicit-schema KGs. Omitted because it requires Neo4j infrastructure. |

### 8.3 What We Added (Our Extensions Beyond the Paper)

The following components are **our systems-layer contributions**, not part of the SAGE paper's academic contribution:

| Extension | Description |
| :--- | :--- |
| **JSON-LD Export** | Schema.org-compliant knowledge graph serialization. Logically consistent with Linked Data standards, though our `Relationship` nodes use reification rather than standard RDF edge modeling. |
| **Tools4AI Integration** | `@Agent`/`@Action` annotations making graph operations LLM-callable. Demonstrates production extensibility. |
| **Query-Scoped Export** | JSON-LD fragments for individual retrieval results, enabling downstream consumption. |

---

## 9. Future Enhancements

### 9.1 Pluggable Similarity & Caching (Completed ✅)
We have successfully implemented the `SimilarityProvider` interface and the `CosineSimilarityProvider`. The implementation includes a robust caching mechanism to ensure that expensive embedding API calls (like NVIDIA NIM) are performed only once per unique text chunk, even during intensive $O(N^2)$ pairwise graph construction.

### 9.2 Dynamic Metadata Extraction
Replace manually annotated metadata with automatically generated metadata using the Tools4AI `AIProcessor`. This would fully operationalize the "LLM-extracted metadata" signal described in the paper.
The paper's **SPARK** component uses an LLM to generate Cypher queries for Neo4j traversal of schema-rich knowledge graphs. This could be implemented as an additional `@Action` in the `GraphRetrievalAction` class, using Tools4AI's `processSingleAction()` to let the LLM dynamically construct graph queries.

### 9.3 Multi-Hop Expansion (2+ Hops)
The current implementation performs 1-hop neighbor expansion. The paper notes that for deeply nested evidence chains, 2-hop expansion could recover even more bridging evidence — at the cost of higher candidate noise that would need more aggressive pruning.

### 9.4 HumanInLoop Safety Integration
Following the Agent Misalignment project's pattern, a `RetrievalSafetyCallback` implementing `HumanInLoop` could intercept graph queries before execution to prevent unauthorized information access — bringing the same "deterministic oversight" philosophy to knowledge retrieval.

### 9.5 Evaluation Against OTT-QA and STaRK Benchmarks
The paper evaluates on OTT-QA (26,503 text chunks, 5,391 table chunks) and STaRK (AMAZON, MAG, PRIME subsets). A future version could ingest these datasets, construct the full-scale graph, and measure Recall@k. This is the only path to claiming empirical equivalence with the paper's reported improvements (+5.7 Recall on OTT-QA, +8.5 on STaRK). Until then, our implementation demonstrates the algorithm but not the performance.

---

## 10. Why Tools4AI 

Tools4AI is the systems layer that turns the SAGE research paper into a functional AI Agent tool.

While SAGE defines the logic of how to retrieve data, Tools4AI defines how an AI interacts with that logic. Specifically, it is needed for:

1. Semantic Tool Discovery (@Agent & @Action)
SAGE is a Java algorithm. Without Tools4AI, an LLM (like GPT-4 or Llama-3) has no way of knowing it exists or how to call it.

The Role: By annotating 
GraphRetrievalAction.java
 with @Action, you're creating a "manifest" that the LLM can read.
The result: When an AI receives a complex question, it can semantically match the user's intent to the 
searchKnowledgeGraph()
 method because Tools4AI exposed it as a callable capability.
2. Bridging the "Paper" to the "System"
The SAGE paper is academic—it focuses on benchmarks (Recall@k). Tools4AI is operational—it focuses on execution.

In the paper, "Graph Expansion" is a mathematical step.
In your implementation via Tools4AI, 
expandNode
 is a tool call. This allows an LLM to perform "Reasoning-level Expansion"—if the LLM gets an answer but thinks it's incomplete, it can decide to call 
expandNode
 on a specific entity to find more context, exactly as a human researcher would.
3. Infrastructure for LLM-Dependent Components
The SAGE algorithm actually relies on an LLM for two key things:

Metadata Extraction: Segmenting text and extracting "Topics" and "Entities" (Section 2.1).
SPARK Agent: Generating Cypher queries for Neo4j (Section 4).
Tools4AI provides the AIProcessor and configuration (like your 
tools4ai.properties
) to handle these LLM calls natively. You don't have to write custom HTTP clients or JSON parsers for NVIDIA NIM; Tools4AI manages those connections for you.

4. Ecosystem Consistency
You’ve used Tools4AI across multiple projects (Agent Misalignment, Fraud Detection). By using it here, you are building a unified Agentic architecture.

Misalignment project: Uses Tools4AI for Safety (blocking actions).
SAGE project: Uses Tools4AI for Knowledge (providing context). They now share the same "language," allowing you to eventually combine them into a single agent that retrieves knowledge via SAGE while being monitored for safety via Misalignment callback

---

## References

1. Titiya, P., Khoja, R., Wolfson, T., Gupta, V., & Roth, D. (2026). *SAGE: Structure Aware Graph Expansion for Retrieval of Heterogeneous Data.* arXiv:2602.16964.
2. Tools4AI Framework: [github.com/vishalmysore/Tools4AI](https://github.com/vishalmysore/Tools4AI)
3. JSON-LD 1.1 Specification: [w3.org/TR/json-ld11](https://www.w3.org/TR/json-ld11/)
4. Schema.org Vocabulary: [schema.org](https://schema.org/)
