package io.github.vishalmysore.sage.domain;

import lombok.Builder;
import lombok.Data;

import java.util.HashMap;
import java.util.Map;

/**
 * A chunk node in the SAGE graph. Each node represents a semantically coherent
 * unit of content (text passage, table segment, or semi-structured entity).
 * Corresponds to the paper's "chunk-level graph" abstraction.
 */
@Data
@Builder
public class ChunkNode {
    private String id;
    private ChunkType type;
    private String content;
    private String sourceDocument;

    // Metadata extracted by LLM (paper Section 2.1A)
    @Builder.Default
    private Map<String, String> metadata = new HashMap<>();

    // Entities extracted from this chunk
    @Builder.Default
    private Map<String, String> entities = new HashMap<>();

    // Dense embedding vector (for similarity computation)
    private double[] embedding;

    // Relevance score assigned during retrieval
    @Builder.Default
    private double relevanceScore = 0.0;

    /**
     * Converts this node to a JSON-LD representation for knowledge graph
     * interoperability.
     */
    public Map<String, Object> toJsonLd() {
        Map<String, Object> jsonLd = new HashMap<>();
        jsonLd.put("@context", "https://schema.org/");
        jsonLd.put("@type", type == ChunkType.TABLE ? "Table" : "CreativeWork");
        jsonLd.put("@id", "urn:sage:chunk:" + id);
        jsonLd.put("name", metadata.getOrDefault("title", id));
        jsonLd.put("description", content != null && content.length() > 200
                ? content.substring(0, 200) + "..."
                : content);
        jsonLd.put("isPartOf", Map.of("@id", "urn:sage:doc:" + sourceDocument));

        if (!entities.isEmpty()) {
            jsonLd.put("mentions", entities);
        }
        if (!metadata.isEmpty()) {
            jsonLd.put("keywords", metadata.getOrDefault("topic", ""));
        }
        return jsonLd;
    }
}
