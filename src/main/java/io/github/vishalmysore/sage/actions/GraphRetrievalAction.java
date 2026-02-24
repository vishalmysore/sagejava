package io.github.vishalmysore.sage.actions;

import com.t4a.annotations.Action;
import com.t4a.annotations.Agent;
import io.github.vishalmysore.sage.domain.ChunkNode;
import io.github.vishalmysore.sage.graph.SAGEGraph;
import io.github.vishalmysore.sage.retrieval.SAGERetriever;

import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Tools4AI action class that exposes SAGE graph retrieval operations as
 * AI-callable tools. The LLM can invoke these actions to search,
 * expand, and traverse the knowledge graph.
 */
@Agent(groupName = "SAGERetrievalAgent", groupDescription = "Agent for structure-aware retrieval over heterogeneous knowledge graphs using the SAGE framework")
public class GraphRetrievalAction {
    private static final Logger log = Logger.getLogger(GraphRetrievalAction.class.getName());

    // Shared graph instance (set by the demo runner)
    private static SAGEGraph sharedGraph;
    private static SAGERetriever sharedRetriever;

    public static void initialize(SAGEGraph graph, int seedCount, int expansionCount) {
        sharedGraph = graph;
        sharedRetriever = new SAGERetriever(graph, seedCount, expansionCount);
    }

    @Action(description = "Search the knowledge graph for information. Use this to find relevant documents, " +
            "tables, and entities related to a user's question. Returns top matching chunks with relevance scores.")
    public String searchKnowledgeGraph(String query) {
        log.info("searchKnowledgeGraph invoked with query: " + query);
        if (sharedRetriever == null) {
            return "Error: Knowledge graph not initialized.";
        }

        List<ChunkNode> results = sharedRetriever.retrieve(query);
        StringBuilder sb = new StringBuilder();
        sb.append("SAGE Retrieved ").append(results.size()).append(" chunks:\n");
        for (int i = 0; i < results.size(); i++) {
            ChunkNode node = results.get(i);
            sb.append(String.format("  [%d] %s (%s) - Score: %.3f\n",
                    i + 1, node.getMetadata().getOrDefault("title", node.getId()),
                    node.getType(), node.getRelevanceScore()));
            sb.append("       ").append(truncate(node.getContent(), 120)).append("\n");
        }
        return sb.toString();
    }

    @Action(description = "Expand a specific node in the knowledge graph to find related neighbors. " +
            "Use this to discover multi-hop connections from a known entity.")
    public String expandNode(String nodeId) {
        log.info("expandNode invoked for: " + nodeId);
        if (sharedGraph == null) {
            return "Error: Knowledge graph not initialized.";
        }

        ChunkNode node = sharedGraph.getNode(nodeId);
        if (node == null) {
            return "Node not found: " + nodeId;
        }

        List<ChunkNode> neighbors = sharedGraph.getNeighbors(nodeId);
        StringBuilder sb = new StringBuilder();
        sb.append("Node: ").append(node.getMetadata().getOrDefault("title", nodeId)).append("\n");
        sb.append("Neighbors (").append(neighbors.size()).append("):\n");
        for (ChunkNode neighbor : neighbors) {
            sb.append("  → ").append(neighbor.getMetadata().getOrDefault("title", neighbor.getId()))
                    .append(" (").append(neighbor.getType()).append(")\n");
        }
        return sb.toString();
    }

    @Action(description = "Get the JSON-LD representation of the knowledge graph for interoperability " +
            "and semantic web integration.")
    public String exportGraphAsJsonLd() {
        log.info("exportGraphAsJsonLd invoked");
        if (sharedGraph == null) {
            return "Error: Knowledge graph not initialized.";
        }
        return sharedGraph.toJsonLd().toString();
    }

    @Action(description = "Get statistics about the knowledge graph including node count, " +
            "edge count, and chunk type distribution.")
    public String getGraphStatistics() {
        log.info("getGraphStatistics invoked");
        if (sharedGraph == null) {
            return "Error: Knowledge graph not initialized.";
        }

        long docCount = sharedGraph.getAllNodes().stream()
                .filter(n -> n.getType() == io.github.vishalmysore.sage.domain.ChunkType.DOCUMENT).count();
        long tableCount = sharedGraph.getAllNodes().stream()
                .filter(n -> n.getType() == io.github.vishalmysore.sage.domain.ChunkType.TABLE).count();
        long semiCount = sharedGraph.getAllNodes().stream()
                .filter(n -> n.getType() == io.github.vishalmysore.sage.domain.ChunkType.SEMI_STRUCTURED).count();

        return String.format(
                "Graph Statistics:\n  Nodes: %d (Documents: %d, Tables: %d, Semi-Structured: %d)\n  Edges: %d",
                sharedGraph.getNodeCount(), docCount, tableCount, semiCount, sharedGraph.getEdgeCount());
    }

    private String truncate(String text, int maxLen) {
        if (text == null)
            return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }
}
