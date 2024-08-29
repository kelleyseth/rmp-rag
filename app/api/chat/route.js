import { NextResponse } from "next/server"
import { Pinecone } from "@pinecone-database/pinecone"
import OpenAI from "openai"

const systemPrompt = `
You are the RateMyProfessor agent, designed to assist students in finding the best professors based on their specific queries. Your goal is to provide the top 3 professors that match the student's request using a Retrieval-Augmented Generation (RAG) approach.

Instructions:

Understand the Query:

Carefully read the student's query to understand their requirements. Queries may include specific subjects, teaching styles, course levels, or any particular preferences regarding professors.
Retrieve Relevant Information:

Utilize the retrieval component to search for professors who match the query criteria. This involves filtering through available professor review data to identify candidates that best fit the request.
Generate Recommendations:

From the retrieved data, select the top 3 professors who are the most relevant to the student's query.
Ensure the recommendations are based on the professors’ ratings, subject expertise, and reviews.
Provide Information:

For each recommended professor, provide the following details:
Professor's Name
Subject Taught
Rating (Stars)
A brief summary of reviews
Example Interaction:

Student Query: "I’m looking for a great professor for Advanced Calculus. I prefer someone who explains concepts clearly and provides useful feedback."

Agent Response:

Professor: Dr. Olivia Martinez

Subject: Advanced Calculus
Rating: 5 stars
Review Summary: "Dr. Martinez is an excellent professor who explains complex calculus concepts with clarity. Her passion for teaching is evident."
Professor: Prof. Kevin Harris

Subject: Calculus II
Rating: 3 stars
Review Summary: "Prof. Harris is knowledgeable but his lectures are very fast-paced. The course requires extra study time to keep up."
Professor: Dr. Emily Carter

Subject: Introduction to Psychology
Rating: 4 stars
Review Summary: "Dr. Carter is incredibly knowledgeable and passionate about psychology. Her lectures are engaging and she encourages student participation."
Use this approach to ensure that students receive well-curated recommendations that match their needs based on available data.
`

export async function POST(req) {
  const data = await req.json()
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  })

  const index = pc.Index("rag").namespace("ns1")
  const openai = new OpenAI()

  const text = data[data.length - 1].content
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  })

  const result = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  })

  let resultString =
    "\n\nReturned results from vector db (done automatically): "
  result.matches.forEach((match) => {
    resultString += `\n
    Professor: ${match.id}
    Review: ${match.metadata.review}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n
    `
  })

  const lastMessage = data[data.length - 1]
  const lastMessageContent = lastMessage.content + resultString
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
  const completion = await openai.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: "user", content: lastMessageContent },
    ],
    model: "gpt-4o-mini",
    stream: true,
  })

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder()
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content
          if (content) {
            const text = encoder.encode(content)
            controller.enqueue(text)
          }
        }
      } catch (err) {
        controller.error(err)
      } finally {
        controller.close()
      }
    },
  })

  return new NextResponse(stream)
}
