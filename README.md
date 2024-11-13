# Xenova Transformers.js Guide

Xenova Transformers.js is a JavaScript library that provides an easy interface to work with various machine learning models, such as those for text generation, sentiment analysis, embeddings, and semantic similarity. Below, you'll find a detailed explanation of what it is, how it works, and examples of usage.

## What is Xenova Transformers.js?

Xenova Transformers.js is a library designed to bring the power of transformer-based models to JavaScript developers. It enables:

- Natural Language Processing (NLP) tasks.
- Execution of models directly in the browser or Node.js.
- Use of pre-trained models for quick and efficient results.

The library is based on Hugging Face Transformers but is specifically tailored for JavaScript environments, making it a great choice for web and server-side applications.

## How Does Xenova Transformers.js Work?

Xenova Transformers.js uses pre-trained transformer models hosted on platforms like Hugging Face. The library supports various tasks such as:

1. **Text Generation**
2. **Sentiment Analysis**
3. **Embeddings Generation**
4. **Semantic Similarity Calculation**

Each task is implemented using a pipeline, which abstracts the complexity of model loading and execution.

## Installing Xenova Transformers.js

To use the library in your project, install it via npm:

```bash
npm install @xenova/transformers
```

Ensure that your project uses ES Modules, as Xenova Transformers.js requires this format.

## Available Models

Xenova Transformers.js supports a variety of pre-trained models for different tasks:

- **Text Generation:** `Xenova/gpt2`
- **Sentiment Analysis:** `Xenova/distilbert-base-uncased-finetuned-sst-2-english`
- **Embeddings:** `Xenova/all-MiniLM-L6-v2`

## Repository

The Xenova Transformers.js repository can be found on [npm](https://www.npmjs.com/package/@xenova/transformers).

## Examples of Usage

### Text Generation

Generate text based on a prompt:

```javascript
import { pipeline } from "@xenova/transformers";

async function runTextGeneration() {
  const generator = await pipeline("text-generation", "Xenova/gpt2");
  const result = await generator("Once upon a time it was such", {
    max_length: 30,
  });
  console.log(JSON.stringify(result, null, 2));
}

runTextGeneration();
```

### Sentiment Analysis

Classify the sentiment of a given text:

```javascript
import { pipeline } from "@xenova/transformers";

async function runModel() {
  const classifier = await pipeline(
    "sentiment-analysis",
    "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
  );
  const result = await classifier("I love using Transformers.js!");
  console.log(result);
}

runModel();
```

### Generating Embeddings

Create embeddings for a given text to use in downstream tasks like similarity comparison:

```javascript
import { pipeline } from "@xenova/transformers";

async function generateEmbeddings(text) {
  const embedder = await pipeline("embeddings", "Xenova/all-MiniLM-L6-v2");
  const embeddings = await embedder(text);
  console.log(embeddings.data);
}

generateEmbeddings("Hello, Transformers!");
```

### Semantic Similarity

Compare the similarity between two phrases:

```javascript
import { pipeline } from "@xenova/transformers";

class SemanticSimilarityCalculator {
  constructor() {
    this.embedder = null;
  }

  async initialize() {
    this.embedder = await pipeline("embeddings", "Xenova/all-MiniLM-L6-v2");
  }

  async comparePhrases(phrase1, phrase2) {
    const embedding1 = await this.embedder(phrase1, {
      pooling: "mean",
      normalize: true,
    });
    const embedding2 = await this.embedder(phrase2, {
      pooling: "mean",
      normalize: true,
    });

    const similarity = this.cosineSimilarity(embedding1.data, embedding2.data);
    return similarity;
  }

  cosineSimilarity(vec1, vec2) {
    const dot = vec1.reduce((acc, v, i) => acc + v * vec2[i], 0);
    const mag1 = Math.sqrt(vec1.reduce((acc, v) => acc + v * v, 0));
    const mag2 = Math.sqrt(vec2.reduce((acc, v) => acc + v * v, 0));
    return dot / (mag1 * mag2);
  }
}

(async () => {
  const calculator = new SemanticSimilarityCalculator();
  await calculator.initialize();
  const similarity = await calculator.comparePhrases(
    "I love programming",
    "I enjoy writing code"
  );
  console.log("Similarity:", similarity);
})();
```

## Summary

Xenova Transformers.js brings powerful machine learning capabilities to JavaScript developers. With pre-trained models and intuitive pipelines, it simplifies complex tasks like text generation, sentiment analysis, and semantic similarity. This makes it a versatile tool for web and server-side applications.
