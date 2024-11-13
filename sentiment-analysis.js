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
