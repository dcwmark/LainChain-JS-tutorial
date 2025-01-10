import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';

// Use this import for Chain Classes memory approaches
import { ConversationChain } from 'langchain/chains';
import { RunnableSequence } from '@langchain/core/runnables';

// Memory Imports
import { BufferMemory } from 'langchain/memory';
import { UpstashRedisChatMessageHistory } from '@langchain/community/stores/message/upstash_redis';

import 'dotenv/config';

const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
  You are a helpful assistant.
  History: {history}
  {input}
`);

const upstashChatHistory = new UpstashRedisChatMessageHistory({
  sessionId: 'chat1',
  config: {
    url: process.env.UPSTASH_REDIS_REST_URL,
    token: process.env.UPSTASH_REDIS_REST_TOKEN,
  },
});

// *BufferMemory* append messages from the conversation, and the messages
// that has been collected are forwarded into a string and then inject into
// a variable in the prompt called "history".
const memory = new BufferMemory({
  memoryKey: 'history',
  chatHistory: upstashChatHistory,
});

/* Two approaches to ceate a use long term memory: */

// Using the Chain Classes Memory Approach
// const chain = new ConversationChain({
//   llm: model,
//   prompt,
//   memory,
// });

// Using LCEL Memory Approach
// const chain = prompt.pipe(model);
const chain = RunnableSequence.from([
  {
    input: (initialInput) => initialInput.input,  // This is the *input*
                                                  // of input1 or input2
                                                  // following when the
                                                  // chains are invoked.
    memory: () => memory.loadMemoryVariables(),
  },
  {
    input: (previousOutput) => previousOutput.input,
    history: (previousOutput) => previousOutput.memory.history,
  },
  prompt,
  model,
]);

// Get Responses
console.log(`Starting memory`, await memory.loadMemoryVariables());
const input1 = {
  input: 'The passphrase is HELLOWORLD.',
};
const resp1 = await chain.invoke(input1);
console.log(`The Chain Class Reponse::`, resp1);

// This is needed for RunnableSequence class
await memory.saveContext(input1, {
  output: resp1.content,
});

// This is needed for RunnableSequence class
await memory.saveContext(input1, {
  output: resp1.content,
});

console.log(`Updated memory`, await memory.loadMemoryVariables());
const input2 = {
  input: 'What is the passphrase?',
};
const resp2 = await chain.invoke(input2);
console.log(`The Chain Class Reponse::`, resp2);

// This is needed for RunnableSequence class
await memory.saveContext(input2, {
  output: resp2.content,
});
