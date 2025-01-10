import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  CommaSeparatedListOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";

import { z } from "zod";
import { StructuredOutputParser } from "langchain/output_parsers";

import 'dotenv/config';

// Instantiate the model
const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.9,
});

async function callStringOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate("Tell a joke about {word}.");
  const outputParser = new StringOutputParser();

  // Create the Chain
  const chain = prompt.pipe(model).pipe(outputParser);

  return await chain.invoke({ word: "dog" });
}

async function callListOutputParser() {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Provide 5 synonyms, seperated by commas, for a word that the user will provide.",
    ],
    ["human", "{word}"],
  ]);
  const outputParser = new CommaSeparatedListOutputParser();  // This would be a list of strings

  const chain = prompt.pipe(model).pipe(outputParser);

  return await chain.invoke({
    word: "happy",
  });
}

async function callStructuredParser() {
  const prompt = ChatPromptTemplate.fromTemplate(`
    Extract information from the following phrase.
    Formatting Instructions: {format_instructions}
    Phrase: {phrase}
  `);

  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "name of the person",
    age: "age of person",
  });

  const chain = prompt.pipe(model).pipe(outputParser);

  return await chain.invoke({
    phrase: "Max is 30 years old",
    format_instructions: outputParser.getFormatInstructions(),
  });
}

async function callZodStructuredParser() {
  const prompt = ChatPromptTemplate.fromTemplate(`
    Extract information from the following phrase.
    Formatting Instructions: {format_instructions}
    Phrase: {phrase}
  `);

  const outputParser = StructuredOutputParser.fromZodSchema(
    z.object({
      recipe: z.string().describe("name of recipe"),
      ingredients: z.array(z.string()).describe("ingredients"),
    })
  );

  // Create the Chain
  const chain = prompt.pipe(model).pipe(outputParser);

  return await chain.invoke({
    phrase:
      "The ingredients for a Spaghetti Bolognese recipe are tomatoes, minced beef, garlic, wine and herbs.",
    format_instructions: outputParser.getFormatInstructions(),
  });
}

console.log(`***** Calling callStringOutputParser::${await callStringOutputParser()}\n`);
console.log(`***** Calling callListOutputParser::${await callListOutputParser()}\n`);
console.log(`***** Calling callStructuredParser::${JSON.stringify(await callStructuredParser())}\n`);
console.log(`***** Calling callZodStructuredParser::${JSON.stringify(await callZodStructuredParser())}\n`);
