import os
from openai import OpenAI
from Helpers.eval_helpers import print_debug_logs, evaluate_e2e_v3
import re

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
    evaluate_e2e_v3(scenario_openAI, openAI_gpt_query_function, eval_data, start_index, num_items)