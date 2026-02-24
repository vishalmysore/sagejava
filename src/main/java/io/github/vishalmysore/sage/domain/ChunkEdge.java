package io.github.vishalmysore.sage.domain;

import lombok.Builder;
import lombok.Data;

import java.util.HashMap;
import java.util.Map;

/**
 * An edge in the SAGE graph connecting two chunk nodes.
 * Edges store lightweight metadata for controlled graph traversal
 * (paper Section 2.1B: "Edge metadata for traversal").
 */
@Data
@Builder
public class ChunkEdge {
    private String sourceId;
    private String targetId;
    private EdgeType edgeType;
    private double similarityScore;

    // Traversal metadata (shared entities, joinable columns, confidence)
    @Builder.Default
    private Map<String, String> traversalMetadata = new HashMap<>();

    /**
     * Converts this edge to a JSON-LD relationship representation.
     */
    public Map<String, Object> toJsonLd() {
        Map<String, Object> jsonLd = new HashMap<>();
        jsonLd.put("@type", "Relationship");
        jsonLd.put("source", Map.of("@id", "urn:sage:chunk:" + sourceId));
        jsonLd.put("target", Map.of("@id", "urn:sage:chunk:" + targetId));
        jsonLd.put("relationshipType", edgeType.name());
        jsonLd.put("weight", similarityScore);
        if (!traversalMetadata.isEmpty()) {
            jsonLd.put("metadata", traversalMetadata);
        }
        return jsonLd;
    }
}
