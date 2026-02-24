package io.github.vishalmysore.sage.examples;

import io.github.vishalmysore.sage.actions.GraphRetrievalAction;
import io.github.vishalmysore.sage.domain.ChunkNode;
import io.github.vishalmysore.sage.domain.ChunkType;
import io.github.vishalmysore.sage.graph.SAGEGraph;
import io.github.vishalmysore.sage.jsonld.JsonLdExporter;
import io.github.vishalmysore.sage.retrieval.SAGERetriever;
import io.github.vishalmysore.sage.similarity.CosineSimilarityProvider;
import io.github.vishalmysore.sage.similarity.JaccardSimilarityProvider;
import io.github.vishalmysore.sage.similarity.SimilarityProvider;

import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * End-to-end demonstration of the SAGE framework:
 * 1. Offline graph construction with heterogeneous data
 * 2. Online retrieval with graph expansion
 * 3. JSON-LD knowledge graph export
 * 4. Tools4AI action integration
 */
public class SAGEDemoRunner {

        public static void main(String[] args) {
                System.out.println("╔════════════════════════════════════════════════════════════╗");
                System.out.println("║     SAGE: Structure Aware Graph Expansion Demo (Tools4AI) ║");
                System.out.println("║     Paper: arXiv:2602.16964                               ║");
                System.out.println("╚════════════════════════════════════════════════════════════╝\n");

                // === Phase 0: Configuration & Similarity Provider ===
                Properties props = new Properties();
                try (InputStream is = SAGEDemoRunner.class.getClassLoader()
                                .getResourceAsStream("tools4ai.properties")) {
                        if (is != null)
                                props.load(is);
                } catch (Exception e) {
                        System.err.println("Warning: Could not load tools4ai.properties");
                }

                String apiKey = props.getProperty("openAiKey");
                String baseUrl = props.getProperty("openAiBaseURL", "https://integrate.api.nvidia.com/v1");
                String model = props.getProperty("openAiModelName", "nvidia/llama-3.1-nemotron-70b-instruct");

                SimilarityProvider provider;
                if (apiKey != null && !apiKey.trim().isEmpty()) {
                        System.out.println("Using CosineSimilarityProvider (API-backed dense embeddings)\n");
                        provider = new CosineSimilarityProvider(apiKey, baseUrl, model);
                } else {
                        System.out.println("No API key found. Using JaccardSimilarityProvider (lexical fallback)\n");
                        provider = new JaccardSimilarityProvider();
                }

                // === Phase 1: Offline Graph Construction ===
                System.out.println("═══ PHASE 1: OFFLINE GRAPH CONSTRUCTION ═══\n");

                SAGEGraph graph = buildSampleGraph(provider);
                graph.buildEdgesFromMetadata();

                System.out.println("Graph constructed: " + graph.getNodeCount() + " nodes, " + graph.getEdgeCount()
                                + " edges\n");

                // === Phase 2: Online Retrieval ===
                System.out.println("═══ PHASE 2: ONLINE RETRIEVAL WITH GRAPH EXPANSION ═══\n");

                SAGERetriever retriever = new SAGERetriever(graph, 3, 2); // k=3 seeds, k'=2 expansion

                // Query 1: Multi-hop question requiring cross-modal evidence
                String query1 = "What television show starring Trevor Eyster was based on a book?";
                System.out.println("Query: " + query1);
                List<ChunkNode> results1 = retriever.retrieve(query1);
                printResults(results1);

                // Query 2: Question requiring table-document bridging
                String query2 = "Which actors appeared in science fiction films directed by Ridley Scott?";
                System.out.println("\nQuery: " + query2);
                List<ChunkNode> results2 = retriever.retrieve(query2);
                printResults(results2);

                // Query 3: Entity-centric question
                String query3 = "Awards won by the author of Dune";
                System.out.println("\nQuery: " + query3);
                List<ChunkNode> results3 = retriever.retrieve(query3);
                printResults(results3);

                // === Phase 3: JSON-LD Export ===
                System.out.println("\n═══ PHASE 3: JSON-LD KNOWLEDGE GRAPH EXPORT ═══\n");

                JsonLdExporter exporter = new JsonLdExporter();
                String graphJsonLd = exporter.exportGraph(graph);
                System.out.println("JSON-LD Graph Document (preview):");
                System.out.println(graphJsonLd.substring(0, Math.min(graphJsonLd.length(), 800)) + "\n...\n");

                String resultsJsonLd = exporter.exportRetrievalResults(results1, query1);
                System.out.println("JSON-LD Retrieval Results:");
                System.out.println(resultsJsonLd.substring(0, Math.min(resultsJsonLd.length(), 600)) + "\n...\n");

                // === Phase 4: Tools4AI Action Integration ===
                System.out.println("═══ PHASE 4: TOOLS4AI ACTION INTEGRATION ═══\n");

                GraphRetrievalAction.initialize(graph, 3, 2);
                GraphRetrievalAction action = new GraphRetrievalAction();

                System.out.println("--- Action: getGraphStatistics ---");
                System.out.println(action.getGraphStatistics());

                System.out.println("\n--- Action: searchKnowledgeGraph ---");
                System.out.println(action.searchKnowledgeGraph("science fiction films and awards"));

                System.out.println("\n--- Action: expandNode ---");
                System.out.println(action.expandNode("doc-scifi-ridley"));

                System.out.println("\n╔════════════════════════════════════════════════════════════╗");
                System.out.println("║                    DEMO COMPLETE                          ║");
                System.out.println("╚════════════════════════════════════════════════════════════╝");
        }

        /**
         * Builds a sample heterogeneous graph with documents, tables, and
         * semi-structured
         * nodes that demonstrate the SAGE framework's cross-modal retrieval
         * capabilities.
         */
        private static SAGEGraph buildSampleGraph(SimilarityProvider provider) {
                SAGEGraph graph = new SAGEGraph(provider);

                // --- Document chunks ---
                graph.addNode(ChunkNode.builder()
                                .id("doc-eyster-bio")
                                .type(ChunkType.DOCUMENT)
                                .sourceDocument("wikipedia-eyster")
                                .content(
                                                "Trevor Eyster is an American actor known for his role as Arvid Engen in the television series Head of the Class. He appeared in various TV shows during the late 1980s and early 1990s.")
                                .metadata(Map.of("title", "Trevor Eyster Biography", "topic",
                                                "actor television biography"))
                                .entities(Map.of("Trevor Eyster", "PERSON", "Head of the Class", "TV_SHOW",
                                                "Arvid Engen", "CHARACTER"))
                                .build());

                graph.addNode(ChunkNode.builder()
                                .id("doc-eerie-indiana")
                                .type(ChunkType.DOCUMENT)
                                .sourceDocument("wikipedia-eerie")
                                .content(
                                                "Eerie, Indiana is an American television series that aired on NBC. The show follows Marshall Teller and Simon Holmes as they investigate strange occurrences in the fictional town of Eerie, Indiana. Trevor Eyster appeared as Simon Holmes.")
                                .metadata(Map.of("title", "Eerie Indiana TV Show", "topic",
                                                "television science fiction show series"))
                                .entities(Map.of("Eerie Indiana", "TV_SHOW", "Trevor Eyster", "PERSON", "Simon Holmes",
                                                "CHARACTER",
                                                "NBC", "NETWORK"))
                                .build());

                graph.addNode(ChunkNode.builder()
                                .id("doc-eerie-book")
                                .type(ChunkType.DOCUMENT)
                                .sourceDocument("wikipedia-eerie-book")
                                .content(
                                                "Eerie, Indiana was originally based on a book series of the same name. The book series inspired the television adaptation which aired on NBC in 1991.")
                                .metadata(Map.of("title", "Eerie Indiana Book Adaptation", "topic",
                                                "book television adaptation"))
                                .entities(Map.of("Eerie Indiana", "TV_SHOW", "NBC", "NETWORK"))
                                .build());

                graph.addNode(ChunkNode.builder()
                                .id("doc-scifi-ridley")
                                .type(ChunkType.DOCUMENT)
                                .sourceDocument("wikipedia-ridley-scott")
                                .content(
                                                "Ridley Scott is a British filmmaker known for directing science fiction films including Blade Runner and Alien. His films have won numerous awards and are considered classics of the genre.")
                                .metadata(Map.of("title", "Ridley Scott Filmography", "topic",
                                                "director science fiction film awards"))
                                .entities(Map.of("Ridley Scott", "PERSON", "Blade Runner", "FILM", "Alien", "FILM"))
                                .build());

                graph.addNode(ChunkNode.builder()
                                .id("doc-blade-runner")
                                .type(ChunkType.DOCUMENT)
                                .sourceDocument("wikipedia-blade-runner")
                                .content(
                                                "Blade Runner is a 1982 science fiction film directed by Ridley Scott, starring Harrison Ford. The film is based on the novel Do Androids Dream of Electric Sheep? by Philip K. Dick.")
                                .metadata(Map.of("title", "Blade Runner Film", "topic",
                                                "science fiction film actor director"))
                                .entities(Map.of("Blade Runner", "FILM", "Ridley Scott", "PERSON", "Harrison Ford",
                                                "PERSON",
                                                "Philip K. Dick", "PERSON"))
                                .build());

                graph.addNode(ChunkNode.builder()
                                .id("doc-dune-author")
                                .type(ChunkType.DOCUMENT)
                                .sourceDocument("wikipedia-frank-herbert")
                                .content(
                                                "Frank Herbert was an American science fiction author, best known for the novel Dune. He won the Hugo Award and the Nebula Award for Dune in 1966.")
                                .metadata(Map.of("title", "Frank Herbert Author", "topic",
                                                "author science fiction awards"))
                                .entities(Map.of("Frank Herbert", "PERSON", "Dune", "BOOK", "Hugo Award", "AWARD",
                                                "Nebula Award",
                                                "AWARD"))
                                .build());

                graph.addNode(ChunkNode.builder()
                                .id("doc-dune-film")
                                .type(ChunkType.DOCUMENT)
                                .sourceDocument("wikipedia-dune-film")
                                .content(
                                                "Dune is a 2021 science fiction film directed by Denis Villeneuve, based on the 1965 novel by Frank Herbert. The film stars Timothee Chalamet and won six Academy Awards.")
                                .metadata(Map.of("title", "Dune Film 2021", "topic",
                                                "science fiction film adaptation awards"))
                                .entities(Map.of("Dune", "FILM", "Frank Herbert", "PERSON", "Denis Villeneuve",
                                                "PERSON",
                                                "Academy Awards", "AWARD"))
                                .build());

                // --- Table chunks ---
                graph.addNode(ChunkNode.builder()
                                .id("table-scifi-films")
                                .type(ChunkType.TABLE)
                                .sourceDocument("wiki-table-scifi")
                                .content(
                                                "Film | Director | Year | Stars\nBlade Runner | Ridley Scott | 1982 | Harrison Ford\nAlien | Ridley Scott | 1979 | Sigourney Weaver\nDune | Denis Villeneuve | 2021 | Timothee Chalamet\nArrival | Denis Villeneuve | 2016 | Amy Adams")
                                .metadata(Map.of("title", "Science Fiction Films Table", "topic",
                                                "science fiction films directors actors"))
                                .entities(Map.of("Ridley Scott", "PERSON", "Harrison Ford", "PERSON", "Blade Runner",
                                                "FILM", "Dune",
                                                "FILM"))
                                .build());

                graph.addNode(ChunkNode.builder()
                                .id("table-awards")
                                .type(ChunkType.TABLE)
                                .sourceDocument("wiki-table-awards")
                                .content(
                                                "Recipient | Award | Year | Category\nFrank Herbert | Hugo Award | 1966 | Best Novel\nFrank Herbert | Nebula Award | 1966 | Best Novel\nRidley Scott | BAFTA | 2001 | Best Director\nDune (2021) | Academy Award | 2022 | Best Cinematography")
                                .metadata(Map.of("title", "Science Fiction Awards Table", "topic",
                                                "awards science fiction author director"))
                                .entities(Map.of("Frank Herbert", "PERSON", "Hugo Award", "AWARD", "Ridley Scott",
                                                "PERSON", "Dune",
                                                "FILM"))
                                .build());

                // --- Semi-structured nodes ---
                graph.addNode(ChunkNode.builder()
                                .id("semi-trevor-eyster")
                                .type(ChunkType.SEMI_STRUCTURED)
                                .sourceDocument("imdb-trevor-eyster")
                                .content("Name: Trevor Eyster, Occupation: Actor, Known For: Head of the Class, Eerie Indiana")
                                .metadata(Map.of("title", "Trevor Eyster IMDB Profile", "topic", "actor television"))
                                .entities(Map.of("Trevor Eyster", "PERSON", "Head of the Class", "TV_SHOW",
                                                "Eerie Indiana", "TV_SHOW"))
                                .build());

                graph.addNode(ChunkNode.builder()
                                .id("semi-ridley-scott")
                                .type(ChunkType.SEMI_STRUCTURED)
                                .sourceDocument("imdb-ridley-scott")
                                .content("Name: Ridley Scott, Occupation: Director, Known For: Blade Runner, Alien, Gladiator")
                                .metadata(Map.of("title", "Ridley Scott IMDB Profile", "topic",
                                                "director science fiction film"))
                                .entities(Map.of("Ridley Scott", "PERSON", "Blade Runner", "FILM", "Alien", "FILM"))
                                .build());

                return graph;
        }

        private static void printResults(List<ChunkNode> results) {
                System.out.println("  Retrieved " + results.size() + " chunks:");
                for (int i = 0; i < results.size(); i++) {
                        ChunkNode node = results.get(i);
                        System.out.printf("    [%d] %s (%s) Score=%.3f%n",
                                        i + 1,
                                        node.getMetadata().getOrDefault("title", node.getId()),
                                        node.getType(),
                                        node.getRelevanceScore());
                }
        }
}
