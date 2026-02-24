package io.github.vishalmysore.sage.domain;

/**
 * Represents the type of a chunk node in the SAGE graph.
 * Maps to the heterogeneous data types described in the paper.
 */
public enum ChunkType {
    DOCUMENT, // Text passage chunk
    TABLE, // Table segment chunk
    SEMI_STRUCTURED // Semi-structured node (e.g., product, paper, entity)
}
