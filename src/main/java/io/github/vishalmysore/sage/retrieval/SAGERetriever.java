package io.github.vishalmysore.sage.retrieval;

import io.github.vishalmysore.sage.domain.ChunkNode;
import io.github.vishalmysore.sage.graph.SAGEGraph;
import io.github.vishalmysore.sage.similarity.SimilarityProvider;

import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Implements the SAGE two-stage online retrieval strategy (Section 2.2):
 * Stage 1: Initial baseline retrieval (hybrid sparse+dense) to get k seed
 * nodes.
 * Stage 2: Graph-based neighbor expansion and pruning to select k' additional
 * nodes.
 *
 * The "dense" component delegates to the graph's {@link SimilarityProvider},
 * which may be Jaccard (default) or cosine over dense embeddings.
 */
public class SAGERetriever {
    private static final Logger log = Logger.getLogger(SAGERetriever.class.getName());

    private final SAGEGraph graph;
    private final SimilarityProvider similarityProvider;
    private final int seedCount; // k: number of seed nodes
    private final int expansionCount; // k': number of expansion nodes

    public SAGERetriever(SAGEGraph graph, int seedCount, int expansionCount) {
        this.graph = graph;
        this.similarityProvider = graph.getSimilarityProvider();
        this.seedCount = seedCount;
        this.expansionCount = expansionCount;
    }

    /**
     * Full SAGE retrieval pipeline for a query.
     * Returns the union of seed nodes and expanded neighbors (k + k' total).
     */
    public List<ChunkNode> retrieve(String query) {
        log.info("SAGE Retrieval for query: " + query);

        // Stage 1: Baseline retrieval (BM25 + dense similarity)
        List<ChunkNode> seeds = baselineRetrieval(query, seedCount);
        log.info("Stage 1 - Seeds retrieved: " + seeds.size());

        // Stage 2: Graph expansion + pruning
        List<ChunkNode> expanded = graphExpansionAndPruning(query, seeds, expansionCount);
        log.info("Stage 2 - Expanded neighbors: " + expanded.size());

        // Final context: union of seeds and expanded neighbors
        Set<String> seedIds = seeds.stream().map(ChunkNode::getId).collect(Collectors.toSet());
        List<ChunkNode> result = new ArrayList<>(seeds);
        for (ChunkNode node : expanded) {
            if (!seedIds.contains(node.getId())) {
                result.add(node);
            }
        }

        log.info("Final context: " + result.size() + " nodes (k=" + seeds.size() + " + k'="
                + (result.size() - seeds.size()) + ")");
        return result;
    }

    /**
     * Stage 1: Hybrid sparse-dense baseline retrieval.
     * Uses BM25-style lexical matching combined with content similarity.
     */
    private List<ChunkNode> baselineRetrieval(String query, int k) {
        String[] queryTerms = query.toLowerCase().split("\\s+");

        List<ChunkNode> scoredNodes = new ArrayList<>();
        for (ChunkNode node : graph.getAllNodes()) {
            double bm25Score = computeBM25Score(queryTerms, node);
            double denseScore = computeDenseSimilarity(query, node);

            // Hybrid combination (paper uses linear interpolation)
            double hybridScore = 0.5 * bm25Score + 0.5 * denseScore;
            node.setRelevanceScore(hybridScore);
            scoredNodes.add(node);
        }

        scoredNodes.sort((a, b) -> Double.compare(b.getRelevanceScore(), a.getRelevanceScore()));
        return scoredNodes.stream().limit(k).collect(Collectors.toList());
    }

    /**
     * Stage 2: Graph-based neighbor expansion and pruning.
     * (1) Collect first-hop neighbors of seeds
     * (2) Re-rank neighbors using BM25 + dense similarity
     * (3) Select top-k' neighbors
     */
    private List<ChunkNode> graphExpansionAndPruning(String query, List<ChunkNode> seeds, int kPrime) {
        String[] queryTerms = query.toLowerCase().split("\\s+");

        // Step 1: Neighbor expansion
        Set<String> seedIds = seeds.stream().map(ChunkNode::getId).collect(Collectors.toSet());
        Set<ChunkNode> candidates = new LinkedHashSet<>();

        for (ChunkNode seed : seeds) {
            List<ChunkNode> neighbors = graph.getNeighbors(seed.getId());
            for (ChunkNode neighbor : neighbors) {
                if (!seedIds.contains(neighbor.getId())) {
                    candidates.add(neighbor);
                }
            }
        }

        // Step 2: Neighbor pruning (re-rank by query relevance)
        for (ChunkNode candidate : candidates) {
            double bm25 = computeBM25Score(queryTerms, candidate);
            double dense = computeDenseSimilarity(query, candidate);
            candidate.setRelevanceScore(0.5 * bm25 + 0.5 * dense);
        }

        // Step 3: Select top-k'
        return candidates.stream()
                .sorted((a, b) -> Double.compare(b.getRelevanceScore(), a.getRelevanceScore()))
                .limit(kPrime)
                .collect(Collectors.toList());
    }

    /**
     * BM25-style lexical scoring (simplified).
     */
    private double computeBM25Score(String[] queryTerms, ChunkNode node) {
        String nodeText = (node.getContent() + " " +
                node.getMetadata().getOrDefault("topic", "") + " " +
                node.getMetadata().getOrDefault("title", "") + " " +
                String.join(" ", node.getEntities().keySet())).toLowerCase();

        int hits = 0;
        for (String term : queryTerms) {
            if (nodeText.contains(term))
                hits++;
        }
        return queryTerms.length == 0 ? 0.0 : (double) hits / queryTerms.length;
    }

    /**
     * Dense similarity scoring. Delegates to the pluggable similarity provider.
     */
    private double computeDenseSimilarity(String query, ChunkNode node) {
        String content = node.getContent() != null ? node.getContent() : "";
        return similarityProvider.computeSimilarity(query, content);
    }
}
