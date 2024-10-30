# ## Level 1 - Basic RAG

# To get started, we'll set up a basic RAG pipeline using a bare bones approach. This is sometimes called 'Naive RAG' by many in the industry. A basic RAG pipeline includes the following 3 steps:

# 1) Chunk documents by heading - containing only the content from each subheading

# 2) Embed each document

# 3) Use Cosine similarity to retrieve documents in order to answer query

import json
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
import logging
from typing import Callable, List, Dict, Any, Tuple, Set
from Helpers.metric_helpers import evaluate_retrieval, evaluate_end_to_end
from Helpers.retrieve_query_helpers import answer_query_base
from Helpers.eval_helpers import save_xml_string_to_file, save_e2e_results_to_csv, print_and_save_avg_metrics, plot_performance

from Helpers.voyage_vector_db import VectorDB

import anthropic
import os

client = anthropic.Anthropic(
    # This is the default and can be omitted
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# Load the evaluation dataset
with open('evaluation/docs_evaluation_dataset.json', 'r') as f:
    eval_data = json.load(f)

# Load the Anthropic documentation
with open('data/anthropic_docs.json', 'r') as f:
    anthropic_docs = json.load(f)

# Initialize the VectorDB
db = VectorDB("anthropic_docs")
db.load_data(anthropic_docs)

def retrieve_base(query, db):
    results = db.search(query, k=3)
    context = ""
    for result in results:
        chunk = result['metadata']
        context += f"\n{chunk['text']}\n"
    return results, context

def answer_query_base(query, db):
    documents, context = retrieve_base(query, db)
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

def evaluate_basic_rag(eval_data_to_use, db, topK = None):
    if topK is not None:
        eval_data_to_use = eval_data[:topK]

    avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs = evaluate_retrieval(retrieve_base, eval_data_to_use, db)
    e2e_accuracy, e2e_results, detailed_responses = evaluate_end_to_end(answer_query_base, db, eval_data_to_use)

    detailed_responses_file_path = "evaluation/xmls/evaluation_results_detailed.xml"
    save_xml_string_to_file(detailed_responses, detailed_responses_file_path)
    print(f"Detailed LLM responses saved to: {detailed_responses_file_path}")

    evaluation_results_detailed_path = 'evaluation/csvs/evaluation_results_detailed.csv'
    save_e2e_results_to_csv(eval_data_to_use,
                        precisions,
                        recalls,
                        mrrs,
                        e2e_results,
                        evaluation_results_detailed_path)
    print(f"Detailed results saved to: {evaluation_results_detailed_path}")


    avg_metrics_path = 'evaluation/json_results/evaluation_results_one.json'
    print_and_save_avg_metrics(avg_precision, avg_recall, f1, avg_mrr, e2e_accuracy, avg_metrics_path)
    print(f"Avg Metrics saved to: {avg_metrics_path}")

    plot_performance('evaluation/json_results', ['Basic RAG'], colors=['skyblue'])

    print("Evaluation complete")