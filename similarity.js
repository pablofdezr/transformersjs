import { pipeline } from "@xenova/transformers";

class SemanticSimilarityCalculator {
  constructor() {
    this.embedder = null;
  }

  async initialize() {
    try {
      console.log("Initializing embeddings model...");
      this.embedder = await pipeline("embeddings", "Xenova/all-MiniLM-L6-v2");
      console.log("Model initialized successfully");
    } catch (error) {
      throw new Error(
        `Failed to initialize embeddings model: ${error.message}`
      );
    }
  }

  async getEmbedding(text) {
    if (!this.embedder) {
      throw new Error("Model not initialized. Call initialize() first.");
    }

    try {
      // The model returns an object with the embedding data
      const output = await this.embedder(text, {
        pooling: "mean",
        normalize: true,
      });

      // Extract the embedding array from the output
      // The exact path depends on the model's output format
      const embedding = Array.from(output.data);

      return embedding;
    } catch (error) {
      throw new Error(`Failed to generate embedding: ${error.message}`);
    }
  }

  // This function calculates the cosine similarity between two embeddings
  cosineSimilarity(embedding1, embedding2) {
    if (!Array.isArray(embedding1) || !Array.isArray(embedding2)) {
      throw new Error("Embeddings must be arrays");
    }

    if (embedding1.length !== embedding2.length) {
      throw new Error("Embeddings must have the same dimension");
    }

    // Calculate dot product
    const dotProduct = embedding1.reduce(
      (sum, value, i) => sum + value * embedding2[i],
      0
    );

    // Calculate magnitudes
    const magnitude1 = Math.sqrt(
      embedding1.reduce((sum, value) => sum + value * value, 0)
    );
    const magnitude2 = Math.sqrt(
      embedding2.reduce((sum, value) => sum + value * value, 0)
    );

    // Calculate cosine similarity
    return dotProduct / (magnitude1 * magnitude2);
  }

  async comparePhrases(phrase1, phrase2) {
    try {
      const embedding1 = await this.getEmbedding(phrase1);
      const embedding2 = await this.getEmbedding(phrase2);

      const similarity = this.cosineSimilarity(embedding1, embedding2);
      return {
        similarity,
        interpretation: this.interpretSimilarity(similarity),
        phrases: {
          phrase1,
          phrase2,
        },
      };
    } catch (error) {
      throw new Error(`Failed to compare phrases: ${error.message}`);
    }
  }

  interpretSimilarity(similarity) {
    if (similarity >= 0.9) return "Nearly identical meaning";
    if (similarity >= 0.7) return "Very similar meaning";
    if (similarity >= 0.5) return "Moderately similar";
    if (similarity >= 0.3) return "Slightly similar";
    return "Different meanings";
  }
}

// Example usage
async function demonstrateSimilarity() {
  const calculator = new SemanticSimilarityCalculator();

  try {
    await calculator.initialize();

    // Example comparisons
    const examples = [
      {
        phrase1: "I love programming",
        phrase2: "I enjoy writing code",
      },
      {
        phrase1: "The weather is nice today",
        phrase2: "It's a beautiful sunny day",
      },
      {
        phrase1: "The cat is sleeping",
        phrase2: "The dog is barking",
      },
    ];

    console.log("Analyzing semantic similarities...\n");

    for (const { phrase1, phrase2 } of examples) {
      const result = await calculator.comparePhrases(phrase1, phrase2);
      console.log(`Comparing:\n"${phrase1}"\nwith:\n"${phrase2}"\n`);
      console.log(`Similarity score: ${result.similarity.toFixed(4)}`);
      console.log(`Interpretation: ${result.interpretation}\n`);
    }
  } catch (error) {
    console.error("Error:", error.message);
  }
}

// Run the demonstration
demonstrateSimilarity();
