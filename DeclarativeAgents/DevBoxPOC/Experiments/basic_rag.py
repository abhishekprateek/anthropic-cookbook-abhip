# ## Level 1 - Basic RAG

# To get started, we'll set up a basic RAG pipeline using a bare bones approach. This is sometimes called 'Naive RAG' by many in the industry. A basic RAG pipeline includes the following 3 steps:

# 1) Chunk documents by heading - containing only the content from each subheading

# 2) Embed each document

# 3) Use Cosine similarity to retrieve documents in order to answer query

from functools import partial
import os
from typing import List, Dict, Any
from Helpers.eval_helpers import evaluate_e2e_v3

import anthropic

scenario_basicRAG = 'BasicRAG'

client = anthropic.Anthropic(
    # This is the default and can be omitted
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

def basic_rag_search(db, query):
    chunk_links, context = retrieve_base(query, db)

    generated_answer = query_llm(query, context)

    return generated_answer, chunk_links

def retrieve_base(query, db):
    chunk_links = []
    results = db.search(query)
    context = ""
    for result in results:
        chunk = result['metadata']
        chunk_links.append(chunk['chunk_link'])
        context += f"\n{chunk['text']}\n"
    return chunk_links, context

def answer_query_base(query, db):
    chunk_links, context = retrieve_base(query, db)
    return query_llm(query, context)

def query_llm(query, context):
    prompt = f"""
    You have been tasked with helping us to answer the following query: 
    <query>
    {query}
    </query>
    You have access to the following documents which are meant to provide context as you answer the query:
    <documents>
    {context}
    </documents>
    Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already. 
    Answer the question now, and avoid providing preamble such as 'Here is the answer', etc
    """
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=2500,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.content[0].text

def evaluate_basic_rag_v2(eval_data, db, start_index = 0, num_items = None):
    
    rag_query_function = partial(basic_rag_search, db)
    evaluate_e2e_v3(scenario_basicRAG, rag_query_function, eval_data, start_index, num_items)