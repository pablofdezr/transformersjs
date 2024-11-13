import { pipeline } from "@xenova/transformers";

async function generateEmbeddings(text) {
  // We load the embeddings pipeline
  const embedder = await pipeline("embeddings", "Xenova/all-MiniLM-L6-v2");

  // Generate the embeddings
  const embeddings = await embedder(text);

  // Then show the embeddings
  console.log(embeddings.data);
}

generateEmbeddings("Hello, Transformers!");
