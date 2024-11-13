import { pipeline } from "@xenova/transformers";

async function runTextGeneration() {
  const generator = await pipeline("text-generation", "Xenova/gpt2");
  const result = await generator("Once upon a time it was such", {
    max_length: 30,
  });

  // Convert the result to JSON format
  console.log(JSON.stringify(result, null, 2));
}

runTextGeneration();
