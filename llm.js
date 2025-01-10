import { ChatOpenAI } from "@langchain/openai";

import 'dotenv/config';

const model = new ChatOpenAI({
  // openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
  maxTokens: 1000,
  verbose: true,
});

const response = await model.invoke(
  "Write a haiku about a robotic dream."
);
console.log(response);
/*
const batch = await model.batch([
  "What is the capital of France?",
  "What is the capital of Germany?",
]);
console.log('batch::', batch);

const stream = await model.stream(
  "Write a haiku about a robotic dream."
);
for await (const chunk of stream) {
  console.log(`chunk::`, chunk?.content);
}

const streamLog = await model.streamLog(
  "Write a haiku about a robotic dream."
);
for await (const chunk of streamLog) {
  console.log(`chunk::`, chunk);
}
*/

