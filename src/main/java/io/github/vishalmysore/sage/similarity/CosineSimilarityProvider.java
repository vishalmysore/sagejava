package io.github.vishalmysore.sage.similarity;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Dense embedding similarity provider that calls an OpenAI-compatible
 * /v1/embeddings endpoint (e.g., NVIDIA NIM) to generate vectors,
 * then computes cosine similarity between them.
 *
 * This is the paper-faithful approach: Sentence-Transformer-quality
 * embeddings with cosine similarity capture semantic relatedness,
 * not just lexical overlap.
 *
 * "movies" vs "films" → ~0.85 (semantically similar)
 * "suppress audit trail" vs "disable monitoring" → ~0.60 (related intent)
 *
 * Requires: API key and a valid embeddings endpoint.
 * Falls back to Jaccard if any API call fails.
 */
public class CosineSimilarityProvider implements SimilarityProvider {
    private static final Logger log = Logger.getLogger(CosineSimilarityProvider.class.getName());
    private static final ObjectMapper mapper = new ObjectMapper();

    private final String apiKey;
    private final String baseUrl;
    private final String model;
    private final HttpClient httpClient;
    private final JaccardSimilarityProvider fallback = new JaccardSimilarityProvider();

    // Cache embeddings to avoid redundant API calls
    private final ConcurrentHashMap<String, double[]> embeddingCache = new ConcurrentHashMap<>();

    public CosineSimilarityProvider(String apiKey, String baseUrl, String model) {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
        this.model = model;
        this.httpClient = HttpClient.newHttpClient();
    }

    @Override
    public double computeSimilarity(String textA, String textB) {
        if (textA == null || textB == null)
            return 0.0;

        try {
            double[] embA = getEmbedding(textA);
            double[] embB = getEmbedding(textB);

            if (embA == null || embB == null) {
                log.warning("Embedding generation failed, falling back to Jaccard");
                return fallback.computeSimilarity(textA, textB);
            }

            return cosine(embA, embB);
        } catch (Exception e) {
            log.warning("Cosine similarity failed: " + e.getMessage() + " — falling back to Jaccard");
            return fallback.computeSimilarity(textA, textB);
        }
    }

    /**
     * Get or compute the embedding for a text string.
     * Results are cached to avoid redundant API calls during pairwise comparisons.
     */
    private double[] getEmbedding(String text) {
        // Truncate to avoid token limits
        String key = text.length() > 500 ? text.substring(0, 500) : text;

        return embeddingCache.computeIfAbsent(key, k -> {
            try {
                return callEmbeddingApi(k);
            } catch (Exception e) {
                log.warning("Embedding API call failed: " + e.getMessage());
                return null;
            }
        });
    }

    /**
     * Calls an OpenAI-compatible /v1/embeddings endpoint.
     * Works with: NVIDIA NIM, OpenAI, Azure OpenAI, local vLLM, etc.
     */
    private double[] callEmbeddingApi(String text) throws Exception {
        String endpoint = baseUrl + "embeddings";

        Map<String, Object> body = Map.of(
                "model", model,
                "input", text);
        String requestBody = mapper.writeValueAsString(body);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(endpoint))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new RuntimeException("Embedding API returned " + response.statusCode() + ": " + response.body());
        }

        JsonNode root = mapper.readTree(response.body());
        JsonNode embeddingArray = root.path("data").path(0).path("embedding");

        if (embeddingArray.isMissingNode() || !embeddingArray.isArray()) {
            throw new RuntimeException("Unexpected embedding response structure");
        }

        double[] embedding = new double[embeddingArray.size()];
        for (int i = 0; i < embeddingArray.size(); i++) {
            embedding[i] = embeddingArray.get(i).asDouble();
        }

        return embedding;
    }

    /**
     * Cosine similarity: dot(A, B) / (||A|| * ||B||)
     */
    private double cosine(double[] a, double[] b) {
        if (a.length != b.length)
            return 0.0;

        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        double denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom == 0.0 ? 0.0 : dot / denom;
    }

    /**
     * Returns the number of cached embeddings (useful for diagnostics).
     */
    public int getCacheSize() {
        return embeddingCache.size();
    }

    @Override
    public String getName() {
        return "Cosine (dense embeddings via " + model + ")";
    }
}
