package io.github.vishalmysore.sage.jsonld;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import io.github.vishalmysore.sage.domain.ChunkEdge;
import io.github.vishalmysore.sage.domain.ChunkNode;
import io.github.vishalmysore.sage.graph.SAGEGraph;

import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Exports the SAGE graph as a JSON-LD document compliant with Schema.org
 * and a custom SAGE ontology. This enables interoperability with semantic
 * web tools and knowledge graph systems.
 */
public class JsonLdExporter {
    private static final Logger log = Logger.getLogger(JsonLdExporter.class.getName());
    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    /**
     * Export the full graph as a JSON-LD string.
     */
    public String exportGraph(SAGEGraph graph) {
        try {
            Map<String, Object> jsonLd = buildGraphDocument(graph);
            return mapper.writeValueAsString(jsonLd);
        } catch (Exception e) {
            log.severe("Failed to export graph as JSON-LD: " + e.getMessage());
            return "{}";
        }
    }

    /**
     * Builds the complete JSON-LD document with @context, nodes, and edges.
     */
    private Map<String, Object> buildGraphDocument(SAGEGraph graph) {
        Map<String, Object> doc = new LinkedHashMap<>();

        // JSON-LD @context
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("schema", "https://schema.org/");
        context.put("sage", "urn:sage:ontology:");
        context.put("name", "schema:name");
        context.put("description", "schema:description");
        context.put("isPartOf", Map.of("@type", "@id"));
        context.put("mentions", "sage:mentions");
        context.put("relatedTo", Map.of("@type", "@id"));
        doc.put("@context", context);

        doc.put("@type", "sage:KnowledgeGraph");
        doc.put("sage:framework", "SAGE (Structure Aware Graph Expansion)");
        doc.put("sage:paper", "arXiv:2602.16964");

        // Nodes as @graph
        List<Map<String, Object>> graphItems = new ArrayList<>();
        for (ChunkNode node : graph.getAllNodes()) {
            graphItems.add(node.toJsonLd());
        }

        // Edges as relationships
        for (ChunkEdge edge : graph.getAllEdges()) {
            graphItems.add(edge.toJsonLd());
        }

        doc.put("@graph", graphItems);

        // Statistics
        doc.put("sage:statistics", Map.of(
                "nodeCount", graph.getNodeCount(),
                "edgeCount", graph.getEdgeCount()));

        return doc;
    }

    /**
     * Export only the retrieval results as a JSON-LD fragment.
     */
    public String exportRetrievalResults(List<ChunkNode> results, String query) {
        try {
            Map<String, Object> doc = new LinkedHashMap<>();
            doc.put("@context", Map.of(
                    "schema", "https://schema.org/",
                    "sage", "urn:sage:ontology:"));
            doc.put("@type", "sage:RetrievalResult");
            doc.put("sage:query", query);
            doc.put("sage:resultCount", results.size());
            doc.put("@graph", results.stream()
                    .map(ChunkNode::toJsonLd)
                    .collect(Collectors.toList()));
            return mapper.writeValueAsString(doc);
        } catch (Exception e) {
            log.severe("Failed to export results: " + e.getMessage());
            return "{}";
        }
    }
}
