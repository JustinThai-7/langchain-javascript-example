// Import required dependencies
import 'dotenv/config'
import { ChatOpenAI } from '@langchain/openai'
import { ChatAnthropic } from '@langchain/anthropic'
import { ChatGoogleGenerativeAI } from '@langchain/google-genai'
import { ChatDeepSeek } from "@langchain/deepseek";

// Langchain frameworks
import { HumanMessage, SystemMessage } from '@langchain/core/messages' // to get user imputs
import { StringOutputParser } from '@langchain/core/output_parsers' // to return plaintext LLM outputs
import { ChatPromptTemplate } from '@langchain/core/prompts' // to structure conversations
import { StructuredOutputParser } from 'langchain/output_parsers' // to return JSON

// Choose the LLM model to use
const getOpenAiModel = () => new ChatOpenAI({
    model: process.env.OPENAI_MODEL,
    apiKey: process.env.OPENAI_API_KEY,
})

const getAnthropicModel = () => new ChatAnthropic({
    model: process.env.ANTHROPIC_MODEL,
    apiKey: process.env.ANTHROPIC_API_KEY,
})

const getGoogleModel = () => new ChatGoogleGenerativeAI({
    model: process.env.GOOGLE_MODEL,
    apiKey: process.env.GOOGLE_API_KEY,
})

const getDeepSeekModel = () => new ChatDeepSeek({
    model: process.env.DEEPSEEK_MODEL,
    apiKey: process.env.DEEPSEEK_API_KEY,
})

// Parse command line arguments to determine which model to use
const modelParamIndex = process.argv.findIndex(e => e === '--model')
const modelParamValue = (modelParamIndex >= 2) ? process.argv[modelParamIndex + 1] : "openai"

// Initialize the selected model based on command line argument
let model;
if (modelParamValue === 'google') {
    model = getGoogleModel();
} else if (modelParamValue === 'openai') {
    model = getOpenAiModel();
} else if (modelParamValue === 'anthropic') {
    model = getAnthropicModel();
} else if (modelParamValue === 'deepseek') {
    model = getDeepSeekModel();
} else {
    console.log(`Unsupported model: ${modelParamValue}`);
    process.exit(1);
}

// Function to generate a dad joke based on an input word
// Uses a simple string output parser
const callStringOutputParser = async (input) => {
    // Alternative prompt template using messages
    const prompt = ChatPromptTemplate.fromMessages([
         new SystemMessage("You are a talented comedian.  Tell a joke based on a word provided by the user."),
         new HumanMessage("{input}"),
    ])

    // Create a prompt template for generating dad jokes
    //const prompt = ChatPromptTemplate.fromTemplate(`
    //    You are a talented comedian.
    //    Tell a dad joke based on the word {input}.
    //`)

    // Initialize the string output parser
    const outputParser = new StringOutputParser()

    // Create a processing chain: prompt -> model -> output parser
    const chain = prompt.pipe(model).pipe(outputParser)

    // Execute the chain with the provided input
    return await chain.invoke({
        input,
    })
}

// Function to extract structured information from a text phrase
// Uses a structured output parser to extract name and age
const callStructuredOutputParser = async (phrase) => {
    const prompt = ChatPromptTemplate.fromTemplate(`
        Extract information from the following phrase.
        Formatting Instructions: {format_instructions}
        Phrase: {phrase}
    `)

    const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
        name: "the name of the person",
        age: "the age of the person",
    })

    const chain = prompt.pipe(model).pipe(outputParser)

    console.log('\n=====  Formatting Instructions  =====')
    console.log(outputParser.getFormatInstructions())
    console.log('======================================')

    return await chain.invoke({
        phrase,
        format_instructions: outputParser.getFormatInstructions(),
    })
}


// Example usage of string output parser (dad joke generation)
console.log("\nString Response")
console.log("---------------")
const input = 'LLM'
console.log(`input word: ${input}`)
const stringResponse = await callStringOutputParser(input)
console.log(stringResponse)

console.log("\n")

// Example usage of structured output parser (information extraction)
console.log("Structured Response")
console.log("-------------------")
const phrase = 'Joe is 20 years old'
console.log(`input phrase: ${phrase}`)
const structuredResponse = await callStructuredOutputParser(phrase)
console.log(structuredResponse)
console.log("\n")
console.log("Done")
