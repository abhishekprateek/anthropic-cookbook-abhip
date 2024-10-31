import json
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
import logging
from typing import Callable, List, Dict, Any, Tuple, Set
from Helpers.eval_helpers import print_debug_logs

import anthropic
import os

client = anthropic.Anthropic(
    # This is the default and can be omitted
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

def calculate_mrr(retrieved_links: List[str], correct_links: Set[str]) -> float:
    for i, link in enumerate(retrieved_links, 1):
        if link in correct_links:
            return 1 / i
    return 0

def evaluate_e2e_v2(search_function: Callable, eval_data):
    correct_answers = 0
    is_correct_flags = []
    detailed_responses = []
    total_questions = len(eval_data)
    precisions = []
    recalls = []
    mrrs = []    

    for i, item in enumerate(tqdm(eval_data, desc="Evaluating End-to-End V2")):
        query = item['question']
        correct_answer = item['correct_answer']
        correct_links = set(item['correct_chunks'])

        precision, recall, mrr, detailed_response, is_correct = evauluate_e2e_single_query(query, search_function, correct_answer, correct_links)
        print_debug_logs(f"detailed_response:\n {detailed_response}")

        precisions.append(precision)
        recalls.append(recall)
        mrrs.append(mrr)
        
        detailed_responses.append(detailed_response)
        if is_correct:
            correct_answers += 1
        is_correct_flags.append(is_correct)
        
        logging.info(f"Question {i + 1}/{total_questions}: {query}")
        logging.info(f"Correct: {is_correct}")
        logging.info("---")

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_questions} items. Current Avg Precision: {sum(precisions) / len(precisions):.4f}, Avg Recall: {sum(recalls) / len(recalls):.4f}, Avg MRR: {sum(mrrs) / len(mrrs):.4f}")
            current_accuracy = correct_answers / (i + 1)
            print(f"Processed {i + 1}/{total_questions} questions. Current Accuracy: {current_accuracy:.4f}")

        # time.sleep(2)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    accuracy = correct_answers / total_questions

    return avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs, accuracy, is_correct_flags, detailed_responses

def evauluate_e2e_single_query(query, search_function: Callable, correct_answer, correct_links):
    
    generated_answer, retrieved_links = search_function(query)

    precision, recall, mrr = calculate_retrieval_metrics(retrieved_links, correct_links)

    detailed_response, is_correct = calculate_answer_accuracy(query, correct_answer, generated_answer)

    return precision, recall, mrr, detailed_response, is_correct

def calculate_answer_accuracy(query, correct_answer, generated_answer):
    prompt = f"""
        You are an AI assistant tasked with evaluating the correctness of answers to questions about Anthropic's documentation.
        Question: {query}
        Correct Answer: {correct_answer}
        Generated Answer: {generated_answer}
        Is the Generated Answer correct based on the Correct Answer? You should pay attention to the substance of the answer, and ignore minute details that may differ. 
        Small differences or changes in wording don't matter. If the generated answer and correct answer are saying essentially the same thing then that generated answer should be marked correct. 
        However, if there is any critical piece of information which is missing from the generated answer in comparison to the correct answer, then we should mark this as incorrect. 
        Finally, if there are any direct contradictions between the correect answer and generated answer, we should deem the generated answer to be incorrect.
        Respond in the following XML format:
        <evaluation>
        <content>
        <explanation>Your explanation here</explanation>
        <is_correct>true/false</is_correct>
        </content>
        </evaluation>
        """

    detailed_response = ""
    is_correct = False
    try:
        response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "<evaluation>"}
                ],
                temperature=0,
                stop_sequences=["</evaluation>"]
            )
            
        response_text = response.content[0].text
        print_debug_logs(response_text)

        detailed_response = (
                f'<DetailedResponse>'
                f'<Query>{query}</Query>'
                f'<CorrectAnswer>{correct_answer}</CorrectAnswer>'
                f'<GeneratedAnswer>{generated_answer}</GeneratedAnswer>'
                f'<LLMEvaluation>{response_text}</LLMEvaluation>'
                f'</DetailedResponse>'
            )
        
        is_correct = '<is_correct>true' in response_text.lower()
        # print_debug_logs(f"Is Correct: {is_correct}")

    except Exception as e:
        logging.error(f"Unexpected error: ", exc_info=True)

    return detailed_response, is_correct

def calculate_retrieval_metrics(retrieved_links: List[str], correct_links: Set[str]) -> Tuple[float, float, float]:
    true_positives = len(set(retrieved_links) & correct_links)
    precision = true_positives / len(retrieved_links) if retrieved_links else 0
    recall = true_positives / len(correct_links) if correct_links else 0
    mrr = calculate_mrr(retrieved_links, correct_links)
    return precision, recall, mrr