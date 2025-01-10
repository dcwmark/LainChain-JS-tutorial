import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';

import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

import 'dotenv/config';

const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7,
});

// In retrieval chain, the user input NEEDS to be called "input"
const prompt = ChatPromptTemplate.fromTemplate(`
  Answer the user's question.
  Context: {context}
  Question: {input}
`);

// const chain = prompt.pipe(model);
const chain = await createStuffDocumentsChain({  // notice the await needed
  llm: model,
  prompt,
});

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
// console.log(`splitDocs::`, splitDocs);

const embeddings = new OpenAIEmbeddings();
const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings,
);

const retriever = vectorStore.asRetriever({
  k: 2, // default is 4
});

const retrievalChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever,
});

// The retrieval chain will fetch the revelant docs
// from the vector store and pass them into the context of the prompt.
// The retrieval chain also expects the cotetxt to be called "context"
// in the prompt.
const response = await retrievalChain.invoke({
  input: 'What is LCEL?',
});
console.log(response);
