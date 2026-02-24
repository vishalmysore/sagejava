package io.github.vishalmysore.sage.similarity;

/**
 * Strategy interface for computing text similarity.
 * Allows swapping between lightweight lexical similarity (Jaccard)
 * and true semantic similarity (cosine over dense embeddings).
 *
 * The SAGE paper uses Sentence-Transformer embeddings with cosine similarity.
 * By default, this implementation uses Jaccard as a zero-dependency fallback
 * so the demo runs without an API key or model download.
 */
public interface SimilarityProvider {

    /**
     * Compute similarity between two text strings.
     * 
     * @return a score between 0.0 (no similarity) and 1.0 (identical)
     */
    double computeSimilarity(String textA, String textB);

    /**
     * Descriptive name of this provider (for logging/reporting).
     */
    String getName();
}
