# ## Eval Setup

# When evaluating RAG applications, it's critical to evaluate the performance of the retrieval system and end to end system separately.

# We synthetically generated an evaluation dataset consisting of 100 samples which include the following:
# - A question
# - Chunks from our docs which are relevant to that question. This is what we expect our retrieval system to retrieve when the question is asked
# - A correct answer to the question.

# This is a relatively challenging dataset. Some of our questions require synthesis between more than one chunk in order to be answered correctly, so it's important that our system can load in more than one chunk at a time. You can inspect the dataset by opening `evaluation/docs_evaluation_dataset.json`

# Run the next cell to see a preview of the dataset

#previewing our eval dataset
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
from tqdm import tqdm
import logging
from typing import Callable, List, Dict, Any, Tuple, Set
import anthropic

client = anthropic.Anthropic(
    # This is the default and can be omitted
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

def get_sublist(list, start = 0, num_items = None):
    if num_items is None:
        num_items = len(list) - start
    print(f'StartIndex: {start}, NumItems: {num_items}')
    return list[start:start + num_items]

def print_debug_logs(message):
  debug_logs = os.getenv("DEBUG_LOGS", 'false').lower() == 'true'
  if debug_logs:
    print(message)

def preview_json(file_path, num_items=3):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
        if isinstance(data, list):
            preview_data = data[:num_items]
        elif isinstance(data, dict):
            preview_data = dict(list(data.items())[:num_items])
        else:
            print(f"Unexpected data type: {type(data)}. Cannot preview.")
            return
        
        print(f"Preview of the first {num_items} items from {file_path}:")
        print(json.dumps(preview_data, indent=2))
        print(f"\nTotal number of items: {len(data)}")
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def save_xml_string_to_file(xml_strings, file_path):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        try:
            concatenated_xml = '\n'.join(xml_strings)
            f.write(concatenated_xml)
            # concatenated_xml = "<root>\n" + "\n".join(xml_strings) + "\n</root>"
            # dom = minidom.parseString(concatenated_xml)
            # pretty_xml_as_string = dom.toprettyxml()
            # f.write(pretty_xml_as_string)
        except Exception as e:
            print(f"save_xml_string_to_file error: {str(e)}, raw string: {concatenated_xml}")

def save_e2e_results_to_csv(eval_data, precisions, recalls, mrrs, e2e_results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Create a DataFrame
    df = pd.DataFrame({
        'question': [item['question'] for item in eval_data],
        'retrieval_precision': precisions,
        'retrieval_recall': recalls,
        'retrieval_mrr': mrrs,
        'e2e_correct': e2e_results
    })

    # Save to CSV
    df.to_csv(file_path, index=False)

def print_and_save_avg_metrics(scenario, avg_precision, avg_recall, f1, avg_mrr, e2e_accuracy, file_path):
    # Print the results
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average MRR: {avg_mrr:.4f}")
    print(f"Average F1: {f1:.4f}")
    print(f"End-to-End Accuracy: {e2e_accuracy:.4f}")

    # Save the results to a file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump({
            "name": f'{scenario}',
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": f1,
            "average_mrr": avg_mrr,
            "end_to_end_accuracy": e2e_accuracy
        }, f, indent=2)

def plot_performance(results_folder='evaluation/json_results', include_methods=None, colors=None):
    # Set default colors
    default_colors = ['skyblue', 'lightgreen', 'salmon']
    if colors is None:
        colors = default_colors
    
    # Load JSON files
    results = []
    for filename in os.listdir(results_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(results_folder, filename)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    if 'name' not in data:
                        print(f"Warning: {filename} does not contain a 'name' field. Skipping.")
                        continue
                    if include_methods is None or data['name'] in include_methods:
                        results.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: {filename} is not a valid JSON file. Skipping.")
    
    if not results:
        print("No JSON files found with matching 'name' fields.")
        return
    
    # Validate data
    required_metrics = ["average_precision", "average_recall", "average_f1", "average_mrr", "end_to_end_accuracy"]
    for result in results.copy():
        if not all(metric in result for metric in required_metrics):
            print(f"Warning: {result['name']} is missing some required metrics. Skipping.")
            results.remove(result)
    
    if not results:
        print("No valid results remaining after validation.")
        return
    
    # Sort results based on end-to-end accuracy
    results.sort(key=lambda x: x['end_to_end_accuracy'])
    
    # Prepare data for plotting
    methods = [result['name'] for result in results]
    metrics = required_metrics
    
    # Set up the plot
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")
    
    x = range(len(metrics))
    width = 0.8 / len(results)
    
    # Create color palette
    num_methods = len(results)
    color_palette = colors[:num_methods] + sns.color_palette("husl", num_methods - len(colors))
    
    # Plot bars for each method
    for i, (result, color) in enumerate(zip(results, color_palette)):
        values = [result[metric] for metric in metrics]
        offset = (i - len(results)/2 + 0.5) * width
        bars = plt.bar([xi + offset for xi in x], values, width, label=result['name'], color=color)
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Customize the plot
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.title('RAG Performance Metrics (Sorted by End-to-End Accuracy)', fontsize=16)
    plt.xticks(x, metrics, rotation=45, ha='right')
    plt.legend(title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def evaluate_e2e_internal(search_function: Callable, eval_data):
    correct_answers = 0
    is_correct_flags = []
    detailed_responses = []
    total_questions = len(eval_data)
    precisions = []
    recalls = []
    mrrs = []
    success_count = 0
    failure_count = 0

    for i, item in enumerate(tqdm(eval_data, desc="Evaluating End-to-End V2")):
        try:
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

            success_count += 1
            # time.sleep(2)
        except Exception as e:
            logging.error(f"Unexpected error while processing query #{i}, query {query}: ", exc_info=True)
            failure_count += 1

    print(f"Success count: {success_count}, Failure count: {failure_count}, Success rate: {success_count * 100 / (success_count + failure_count):.4f} %")
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    accuracy = correct_answers / total_questions

    return avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs, accuracy, is_correct_flags, detailed_responses

def calculate_answer_accuracy(query, correct_answer, generated_answer, retrieved_links, correct_links):
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
                f'<CorrectLinks>{correct_links}</CorrectCorrectLinks>'
                f'<GeneratedAnswer>{generated_answer}</GeneratedAnswer>'
                f'<RetrievedLinks>{retrieved_links}</RetrievedLinks>'
                f'<LLMEvaluation>{response_text}</LLMEvaluation>'
                f'</DetailedResponse>'
            )
        
        is_correct = '<is_correct>true' in response_text.lower()
        # print_debug_logs(f"Is Correct: {is_correct}")

    except Exception as e:
        logging.error(f"Unexpected error: ", exc_info=True)

    return detailed_response, is_correct

def calculate_mrr(retrieved_links: List[str], correct_links: Set[str]) -> float:
    for i, link in enumerate(retrieved_links, 1):
        if link in correct_links:
            return 1 / i
    return 0

def calculate_retrieval_metrics(retrieved_links: List[str], correct_links: Set[str]) -> Tuple[float, float, float]:
    true_positives = len(set(retrieved_links) & correct_links)
    precision = true_positives / len(retrieved_links) if retrieved_links else 0
    recall = true_positives / len(correct_links) if correct_links else 0
    mrr = calculate_mrr(retrieved_links, correct_links)
    return precision, recall, mrr

def evauluate_e2e_single_query(query, search_function: Callable, correct_answer, correct_links):
    
    generated_answer, retrieved_links = search_function(query)

    precision, recall, mrr = calculate_retrieval_metrics(retrieved_links, correct_links)

    detailed_response, is_correct = calculate_answer_accuracy(query, correct_answer, generated_answer, retrieved_links, correct_links)

    return precision, recall, mrr, detailed_response, is_correct

def evaluate_e2e_v3(scenario, model_query_function, eval_data, start_index = 0, num_items = None, ):
    eval_data_to_use = get_sublist(eval_data, start_index, num_items)

    avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs, accuracy, is_correct_flags, detailed_responses = evaluate_e2e_internal(model_query_function, eval_data_to_use)
    
    detailed_responses_file_path = f"evaluation/xmls/{scenario}_evaluation_results_detailed.xml"
    save_xml_string_to_file(detailed_responses, detailed_responses_file_path)
    print(f"Detailed LLM responses saved to: {detailed_responses_file_path}")

    evaluation_results_detailed_path = f'evaluation/csvs/{scenario}_evaluation_results_detailed.csv'
    save_e2e_results_to_csv(eval_data_to_use,
                        precisions,
                        recalls,
                        mrrs,
                        is_correct_flags,
                        evaluation_results_detailed_path)
    print(f"Detailed results saved to: {evaluation_results_detailed_path}")


    avg_metrics_path = f'evaluation/json_results/{scenario}_evaluation_results_one.json'
    print_and_save_avg_metrics(scenario, avg_precision, avg_recall, f1, avg_mrr, accuracy, avg_metrics_path)
    print(f"Avg Metrics saved to: {avg_metrics_path}")

    plot_performance('evaluation/json_results', [scenario], colors=['skyblue'])

    print(f"Evaluation complete: {scenario}")

def sample_end_to_end(answer_query_function, db, eval_data, num_queries):
    correct_answers = 0
    results = []
    total_questions = len(eval_data)
    
    for i, item in enumerate(tqdm(eval_data, desc=f"Evaluating End-to-End for {num_queries} queries.")):
        query = item['question']
        correct_answer = item['correct_answer']
        generated_answer = answer_query_function(query, db)
        
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
            print(f"QUERY:\n {query}\n")
            print(f"CORRECT ANSWER:\n {correct_answer}\n")
            print(f"GENERATED ANSWER:\n {generated_answer}\n")
            print(f"LLM EVALUATION:\n {response_text}\n")
            evaluation = ET.fromstring(response_text)
            is_correct = evaluation.find('is_correct').text.lower() == 'true'
            
            if is_correct:
                correct_answers += 1
            results.append(is_correct)
            
            logging.info(f"Question {i + 1}/{total_questions}: {query}")
            logging.info(f"Correct: {is_correct}")
            logging.info("---")
            
        except ET.ParseError as e:
            logging.error(f"XML parsing error: {e}")
            is_correct = 'true' in response_text.lower()
            results.append(is_correct)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            results.append(False)
        
        if (i + 1) % 10 == 0:
            current_accuracy = correct_answers / (i + 1)
            print(f"Processed {i + 1}/{total_questions} questions. Current Accuracy: {current_accuracy:.4f}")

        if i == (num_queries-1):
            print(f"Finished processing {num_queries}, exiting loop now. ")
            break
        # time.sleep(2)
    accuracy = correct_answers / total_questions
    return accuracy, results