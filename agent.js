import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { createOpenAIFunctionsAgent, AgentExecutor } from 'langchain/agents';
import readline from 'readline';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { AIMessage, HumanMessage } from '@langchain/core/messages';

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { createRetrieverTool } from 'langchain/tools/retriever';

import 'dotenv/config';

const loader = new CheerioWebBaseLoader(
  'https://js.langchain.com/docs/how_to/#langchain-expression-language-lcel'
);
const docs = await loader.load();

/**
 * As the loader scraps the full page, the content can be too long
 * for the model to handle.  Not to mention the model's token limit.
 * So we need to split the docs into smaller chunks.
 * We can use the RecursiveCharacterTextSplitter utility to split
 * the docs into smaller chunks.
 * 
 * After splitting the docs, we would want to pass the splitDocs
 *  to a vector store.
 * 
 * In this example, we will use an *in-memory* vector store.
 * In production, you would want to use a vector store like Pinecone or Supabase or others:
 * See https://js.langchain.com/docs/integrations/vectorstores/
 * for a list of vector stores.
 * 
 * Source (splitDocs here) ==> Load ==> Transorm ==> Embed ==> Store ===> Retrieve
**/
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 20,
});
const splitDocs = await splitter.splitDocuments(docs);

const embeddings = new OpenAIEmbeddings();
const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings,
);

const retriever = vectorStore.asRetriever({
  k: 2, // default is 4
});

const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo-1106',
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromMessages([
  ("system", "You are a helpful assistant called Max."),
  new MessagesPlaceholder("chat_history"),
  ("user", "{input}"),
  new MessagesPlaceholder("agent_scratchpad"),
]);

// Create and Assign Tools
const searchTool = new TavilySearchResults();
const retrieverTool = createRetrieverTool(retriever, {
  name: 'lcel_search',
  description: 'Use this tool when searching for information about LangChain Expression Language (LCEL).',
});
const tools = [
  searchTool, retrieverTool,
];

// Create Agent
const agent = await createOpenAIFunctionsAgent({
  llm: model,
  prompt,
  tools,
});

// Create Agent Executor
const agentExecutor = AgentExecutor.fromAgentAndTools({
  agent,
  tools,
});

// Get user input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const chatHistory = [];

const askQuestion = () => {
  rl.question('User:: ', async (input) => {
    if (input.toLowerCase() === 'exit') {
      rl.close();
      return;
    }

    // Call agent
    const response = await agentExecutor.invoke({
      input: input,
      chat_history: chatHistory,
    });
    console.log(`Agent:: `, response.output);
    chatHistory.push(new HumanMessage(input));
    chatHistory.push(new AIMessage(response.output));

    askQuestion();
  });
};

askQuestion();
