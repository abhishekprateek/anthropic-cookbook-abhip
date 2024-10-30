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

preview_json('evaluation/docs_evaluation_dataset.json')

def save_xml_string_to_file(xml_strings, file_path):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        concatenated_xml = "<root>\n" + "\n".join(xml_strings) + "\n</root>"
        dom = minidom.parseString(concatenated_xml)
        pretty_xml_as_string = dom.toprettyxml()
        f.write(pretty_xml_as_string)
    print(f"XML string saved to: {file_path}")
