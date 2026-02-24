package io.github.vishalmysore.sage.domain;

/**
 * Represents the type of edge connecting two chunks in the SAGE graph.
 * The paper defines edges based on metadata-driven similarities.
 */
public enum EdgeType {
    DOC_DOC, // Document-Document: topic/content similarity + entity overlap
    TABLE_TABLE, // Table-Table: column/title similarity + entity overlap
    TABLE_DOC, // Table-Document: content-column + topic-title similarity
    PARENT_CHILD, // Structural link: chunks from the same source document
    SCHEMA_EDGE // Native schema edge from a knowledge graph
}
