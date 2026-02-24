package io.github.vishalmysore.sage.graph;

import io.github.vishalmysore.sage.domain.ChunkEdge;
import io.github.vishalmysore.sage.domain.ChunkNode;
import io.github.vishalmysore.sage.domain.ChunkType;
import io.github.vishalmysore.sage.domain.EdgeType;
import io.github.vishalmysore.sage.similarity.JaccardSimilarityProvider;
import io.github.vishalmysore.sage.similarity.SimilarityProvider;

import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * The SAGE chunk-level graph. Implements the offline graph construction
 * described in Section 2.1 of the paper, using metadata-driven similarities
 * with percentile-based pruning.
 *
 * Similarity computation is delegated to a pluggable
 * {@link SimilarityProvider}.
 * By default, uses {@link JaccardSimilarityProvider} (zero-dependency lexical
 * overlap).
 * For paper-faithful behavior, inject a
 * {@link io.github.vishalmysore.sage.similarity.CosineSimilarityProvider}.
 */
public class SAGEGraph {
    private static final Logger log = Logger.getLogger(SAGEGraph.class.getName());

    private final Map<String, ChunkNode> nodes = new LinkedHashMap<>();
    private final List<ChunkEdge> edges = new ArrayList<>();
    private final Map<String, List<String>> adjacencyList = new HashMap<>();
    private final SimilarityProvider similarityProvider;

    // Percentile threshold for edge pruning (paper uses 95th percentile)
    private static final double SIMILARITY_PERCENTILE_THRESHOLD = 0.95;

    /**
     * Creates a graph with the default Jaccard similarity provider.
     */
    public SAGEGraph() {
        this(new JaccardSimilarityProvider());
    }

    /**
     * Creates a graph with a custom similarity provider.
     * Use CosineSimilarityProvider for paper-faithful dense embeddings.
     */
    public SAGEGraph(SimilarityProvider similarityProvider) {
        this.similarityProvider = similarityProvider;
        log.info("SAGEGraph initialized with similarity: " + similarityProvider.getName());
    }

    public SimilarityProvider getSimilarityProvider() {
        return similarityProvider;
    }

    public void addNode(ChunkNode node) {
        nodes.put(node.getId(), node);
        adjacencyList.putIfAbsent(node.getId(), new ArrayList<>());
    }

    public void addEdge(ChunkEdge edge) {
        edges.add(edge);
        adjacencyList.computeIfAbsent(edge.getSourceId(), k -> new ArrayList<>()).add(edge.getTargetId());
        adjacencyList.computeIfAbsent(edge.getTargetId(), k -> new ArrayList<>()).add(edge.getSourceId());
    }

    public ChunkNode getNode(String id) {
        return nodes.get(id);
    }

    public int getNodeCount() {
        return nodes.size();
    }

    public int getEdgeCount() {
        return edges.size();
    }

    /**
     * Get first-hop neighbors of a node (paper Section 2.2B: Neighbor Expansion).
     */
    public List<ChunkNode> getNeighbors(String nodeId) {
        List<String> neighborIds = adjacencyList.getOrDefault(nodeId, Collections.emptyList());
        return neighborIds.stream()
                .map(nodes::get)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }

    /**
     * Offline graph construction using metadata-driven similarities.
     * Implements percentile-based pruning (Section 2.1B).
     */
    public void buildEdgesFromMetadata() {
        List<ChunkNode> allNodes = new ArrayList<>(nodes.values());
        List<Double> allScores = new ArrayList<>();

        // Compute pairwise similarities
        List<double[]> candidateEdges = new ArrayList<>();
        for (int i = 0; i < allNodes.size(); i++) {
            for (int j = i + 1; j < allNodes.size(); j++) {
                double sim = computeMetadataSimilarity(allNodes.get(i), allNodes.get(j));
                allScores.add(sim);
                candidateEdges.add(new double[] { i, j, sim });
            }
        }

        if (allScores.isEmpty())
            return;

        // Percentile-based pruning
        Collections.sort(allScores);
        int threshIdx = (int) (allScores.size() * SIMILARITY_PERCENTILE_THRESHOLD);
        double threshold = allScores.get(Math.min(threshIdx, allScores.size() - 1));

        log.info("Edge pruning threshold (95th percentile): " + String.format("%.4f", threshold));

        for (double[] candidate : candidateEdges) {
            if (candidate[2] >= threshold) {
                ChunkNode src = allNodes.get((int) candidate[0]);
                ChunkNode tgt = allNodes.get((int) candidate[1]);
                EdgeType edgeType = resolveEdgeType(src.getType(), tgt.getType());

                ChunkEdge edge = ChunkEdge.builder()
                        .sourceId(src.getId())
                        .targetId(tgt.getId())
                        .edgeType(edgeType)
                        .similarityScore(candidate[2])
                        .traversalMetadata(computeTraversalMetadata(src, tgt))
                        .build();
                addEdge(edge);
            }
        }
        log.info("Graph built: " + nodes.size() + " nodes, " + edges.size() + " edges");
    }

    /**
     * Computes metadata similarity between two chunks.
     * Uses topic-topic, entity overlap, and content similarity signals.
     */
    private double computeMetadataSimilarity(ChunkNode a, ChunkNode b) {
        double topicSim = similarityProvider.computeSimilarity(
                a.getMetadata().getOrDefault("topic", ""),
                b.getMetadata().getOrDefault("topic", ""));

        double entityOverlap = computeEntityOverlap(a.getEntities(), b.getEntities());

        double contentSim = similarityProvider.computeSimilarity(
                a.getContent() != null ? a.getContent() : "",
                b.getContent() != null ? b.getContent() : "");

        // Weighted combination
        return 0.4 * topicSim + 0.35 * entityOverlap + 0.25 * contentSim;
    }

    private double computeEntityOverlap(Map<String, String> entA, Map<String, String> entB) {
        if (entA.isEmpty() || entB.isEmpty())
            return 0.0;
        Set<String> keysA = entA.keySet();
        Set<String> keysB = entB.keySet();
        Set<String> intersection = new HashSet<>(keysA);
        intersection.retainAll(keysB);
        Set<String> union = new HashSet<>(keysA);
        union.addAll(keysB);
        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }

    private EdgeType resolveEdgeType(ChunkType src, ChunkType tgt) {
        if (src == ChunkType.TABLE && tgt == ChunkType.TABLE)
            return EdgeType.TABLE_TABLE;
        if (src == ChunkType.TABLE || tgt == ChunkType.TABLE)
            return EdgeType.TABLE_DOC;
        return EdgeType.DOC_DOC;
    }

    private Map<String, String> computeTraversalMetadata(ChunkNode src, ChunkNode tgt) {
        Map<String, String> meta = new HashMap<>();
        // Shared entities
        Set<String> shared = new HashSet<>(src.getEntities().keySet());
        shared.retainAll(tgt.getEntities().keySet());
        if (!shared.isEmpty()) {
            meta.put("sharedEntities", String.join(", ", shared));
        }
        meta.put("sourceType", src.getType().name());
        meta.put("targetType", tgt.getType().name());
        return meta;
    }

    /**
     * Export the entire graph as a JSON-LD document.
     */
    public Map<String, Object> toJsonLd() {
        Map<String, Object> graphDoc = new LinkedHashMap<>();
        graphDoc.put("@context", Map.of(
                "schema", "https://schema.org/",
                "sage", "urn:sage:ontology:",
                "name", "schema:name",
                "description", "schema:description"));
        graphDoc.put("@type", "sage:ChunkGraph");
        graphDoc.put("sage:nodeCount", nodes.size());
        graphDoc.put("sage:edgeCount", edges.size());
        graphDoc.put("@graph", nodes.values().stream()
                .map(ChunkNode::toJsonLd)
                .collect(Collectors.toList()));
        return graphDoc;
    }

    public Collection<ChunkNode> getAllNodes() {
        return nodes.values();
    }

    public List<ChunkEdge> getAllEdges() {
        return edges;
    }
}
