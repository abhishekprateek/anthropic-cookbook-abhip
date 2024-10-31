import os
from openai import OpenAI
from Helpers.eval_helpers import print_debug_logs
import re
from typing import Callable, List, Dict, Any, Tuple, Set
from Helpers.metric_helpers import evaluate_e2e_v2
from Helpers.eval_helpers import save_xml_string_to_file, save_e2e_results_to_csv, print_and_save_avg_metrics, plot_performance
from Helpers.e2e_helpers import get_sublist

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

scenario_openAI = 'OpenAI'

def extract_response_and_links(value):
  
  ## Cleanup the value
  # Remove all occurrences of '【*source】'
  value = re.sub(r'【.*?source】', '', value)

  
  # Extract chunk_links
  chunk_links = re.findall(r'<chunk_link>(.*?)</chunk_link>', value)


  # Extract response
  
  # Find the first occurrence of <Response>
  response_start = value.find('<Response>')

  # Extract substring from <Response> till the end
  response = value[response_start:] if response_start != -1 else ""

  response = response.replace('<Response>', '')
  response = response.replace('</Response>', '')
  
  return chunk_links, response

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
anthropicV3_assistant_id = 'asst_Y08LrfVIXEzGTeFpKiW1vMeT'
anthropicV2_assistant_id = 'asst_IHAfVcQdhcZ5a9YffFhwutS6'

def create_thread_and_run(query):
  thread = client.beta.threads.create()
  print_debug_logs(f'threadId: {thread.id}')

  message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=query
  )

  run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=anthropicV2_assistant_id,
  )

  if run.status == 'completed': 
    messages = client.beta.threads.messages.list(
      thread_id=thread.id
    )

    value = messages.data[0].content[0].text.value
    print_debug_logs(f'Value2:\n {value}')

    chunk_links, response = extract_response_and_links(value)

  else:
    print(run.status)

  return chunk_links, response

def test_openAI_e2e(query):
  debug_logs = True
  query = "How can you create multiple test cases for an evaluation in the Anthropic Evaluation tool?"
  chunk_links, response = create_thread_and_run(query)

def openAI_gpt_query_function(query):
   chunk_links, response = create_thread_and_run(query)
   return response, chunk_links

def evaluate_opeAI_gpt(eval_data, start_index = 0, num_items = None):
    eval_data_to_use = get_sublist(eval_data, start_index, num_items)

    avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs, accuracy, is_correct_flags, detailed_responses = evaluate_e2e_v2(openAI_gpt_query_function, eval_data_to_use)
    
    scenario = scenario_openAI
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