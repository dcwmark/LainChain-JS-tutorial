import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';

// For scrapping the web, setting up the loader, and splitting the docs
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';

// For creating the vector store
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

// For creating chat history
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever';

import 'dotenv/config';

/**
 * Load Data and Create Vector Store
 * @returns 
 */
const createVectorStore = async () => {
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
   * to a vector store.
   * 
   * In this example, we will use an *in-memory* vector store.
   * In production, you would want to use a vector store like
   * Pinecone or Supabase or others:
   * See https://js.langchain.com/docs/integrations/vectorstores/
   * for a list of vector stores.
   * 
   * Source (splitDocs here) ==> Load ==> Transorm ==> Embed ==> Store ===> Retrieve
  **/
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 20,
  });
  const splitDocs = await splitter.splitDocuments(docs);
  
  const embeddings = new OpenAIEmbeddings();

  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings,
  );

  return vectorStore;
};

/**
 * Create Retrieval Chain
 * @param {*} vectorStore 
 * @returns 
 */
const createChain = async (vectorStore) => {
  const model = new ChatOpenAI({
    modelName: 'gpt-3.5-turbo',
    temperature: 0.7,
  });
  
  // In retrieval chain, the user input NEEDS to be called "input".
  // N.B. *****
  // The prompt template for the function fromMessages() is wrapped
  // in an array, with sub-arrays inside of it.
  // Spent HOURS to debug why it didn't work due to missuing the
  // outer bracket.
  // **** *****
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's questions base on the following context: {context}.",
    ],
    new MessagesPlaceholder("chat_history"),
    [    
      "user", "{input}",
    ]
  ]);
  
  // "createStuffDocumentsChain" takes an array of documents,
  // which will be used as context. 
  const chain = await createStuffDocumentsChain({  // notice the await needed
    llm: model,
    prompt,
  });

  // The retriever function is responsible for fetching
  // the relevant documents from the vector store.
  const retriever = vectorStore.asRetriever({
    k: 2, // default is 4
  });

  const retrieverPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    ["user",
     "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ]);

  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt: retrieverPrompt,
  });
  
  // Now, tie the chain and the retriever together.
  const conversationChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: historyAwareRetriever,
  });

  return conversationChain;
};

const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);

// Chat History
const chatHistory = [
  new HumanMessage('Hello!'),
  new AIMessage('Hi! How can I help you?'),
  new HumanMessage('My name is Leonardo'),
  new AIMessage('Hi Leonardo, how can I help you?'),
  new HumanMessage('What is LCEL?'),
  new AIMessage('LCEL is a way to write LangChain expressions.'),
];

// The retrieval chain will fetch the revelant docs
// from the vector store and pass them into the context of the prompt.
// The retrieval chain also expects the cotetxt to be called "context"
// in the prompt.
const response = await chain.invoke({
  input: 'What is it?',
  chat_history: chatHistory,
});
console.log(response);
