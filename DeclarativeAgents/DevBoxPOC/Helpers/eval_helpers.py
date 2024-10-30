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
import xml.dom.minidom as minidom
import os
import pandas as pd

import os
import matplotlib.pyplot as plt
import seaborn as sns

def print_debug_logs(message, debug_logs):
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
        concatenated_xml = "<root>\n" + "\n".join(xml_strings) + "\n</root>"
        dom = minidom.parseString(concatenated_xml)
        pretty_xml_as_string = dom.toprettyxml()
        f.write(pretty_xml_as_string)

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
    df.to_csv('evaluation/csvs/evaluation_results_detailed.csv', index=False)

def print_and_save_avg_metrics(avg_precision, avg_recall, f1, avg_mrr, e2e_accuracy, file_path):
    # Print the results
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average MRR: {avg_mrr:.4f}")
    print(f"Average F1: {f1:.4f}")
    print(f"End-to-End Accuracy: {e2e_accuracy:.4f}")

    # Save the results to a file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open('evaluation/json_results/evaluation_results_one.json', 'w') as f:
        json.dump({
            "name": "Basic RAG",
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