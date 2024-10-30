import os
from openai import OpenAI
import re

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

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

def print_debug_logs(message, debug_logs):
  if debug_logs:
    print(message)

def create_thread_and_run(query, debug_logs = False):
  thread = client.beta.threads.create()
  print(f'threadId: {thread.id}')

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
    # print(messages)

    value = messages.data[0].content[0].text.value
    print_debug_logs(f'Value:\n {value}', debug_logs)

    chunk_links, response = extract_response_and_links(value)

  else:
    print(run.status)

  return chunk_links, response