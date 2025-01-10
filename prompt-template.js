import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import 'dotenv/config';

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});

// Suppose we want to create an AI application
// that will take in any word from the user
// and return a joke based on that word.
// We do not want to have a general conversation
// but rather we want the model to only generate
// jokes based on the word provided by the user.
// We can do this by creating a prompt template.

// Create Prompt Template using fromTemplate
// const prompt = ChatPromptTemplate.fromTemplate(
//   'You are a comedian. Tell a joke based on the followuig word: {input}'
// );
// console.log(await prompt.format({ input: 'chicken' }));
// output::
// Human: You are a comed+ian. Tell a joke based on the followuig word: chicken

// Create Prompt Template using fromMessages
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "Generate a haiku based on the following word: {input}"],
  ["human", "{input}"],
]);

// Create chain instead of using format
const chain = prompt.pipe(model); 

// Call chain
const response = await chain.invoke({ input: 'chicken' });
console.log(response);

