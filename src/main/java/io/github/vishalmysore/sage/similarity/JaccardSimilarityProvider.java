package io.github.vishalmysore.sage.similarity;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Default zero-dependency similarity provider using Jaccard coefficient
 * over tokenized text. Captures lexical overlap only — NOT semantic
 * relatedness.
 *
 * "movies" vs "films" → 0.0 (different tokens)
 * "science fiction" vs "science fiction" → 1.0 (identical tokens)
 *
 * Used as the fallback when no embedding API is available.
 */
public class JaccardSimilarityProvider implements SimilarityProvider {

    @Override
    public double computeSimilarity(String textA, String textB) {
        if (textA == null || textB == null)
            return 0.0;

        Set<String> setA = new HashSet<>(Arrays.asList(textA.toLowerCase().split("\\s+")));
        Set<String> setB = new HashSet<>(Arrays.asList(textB.toLowerCase().split("\\s+")));

        setA.remove("");
        setB.remove("");

        if (setA.isEmpty() && setB.isEmpty())
            return 0.0;

        Set<String> intersection = new HashSet<>(setA);
        intersection.retainAll(setB);

        Set<String> union = new HashSet<>(setA);
        union.addAll(setB);

        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }

    @Override
    public String getName() {
        return "Jaccard (lexical overlap)";
    }
}
