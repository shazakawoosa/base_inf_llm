import torch
import transformers
import os
import json
import re
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define the quantization configuration
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Define the quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # Set the quantization type
    use_nested_quant=False,           # Disable nested quantization
    bnb_4bit_compute_dtype=torch.float16  # Use float16 for computation
)

# Load the model with the new argument
#model_name = "meta-llama/Llama-3.1-8B"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Set up the text generation pipeline
model_id = "meta-llama/Llama-3.1-8B-Instruct"
#model_id = "meta-llama/Llama-3.1-8B"

print(model_id)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use FP16 for further optimizations
    device_map="auto",          # Automatically place layers on GPU(s)
    #load_in_4bit=True           # Use 8-bit quantization (set to `load_in_4bit=True` for 4-bit)
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
       
    },
    device_map="auto",
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    do_sample=True
)

def compare_responses(question_prompt, answer_a, answer_b):
    """Generate a comparison of responses using the given prompt."""
    prompt = f"""[System]
Please evaluate the two provided explanations for the given question. Your task is to choose the explanation that better addresses the question by providing clearer reasoning, evidence, and coherence. 

Compare both explanations objectively, avoiding biases due to their length or order. After comparing, provide your final verdict strictly in this format:
"[A]" if Explanation A is better, "[B]" if Explanation B is better, or "[C]" for a tie.

[New Evaluation]
[Question]
{question_prompt}
[The Start of Explanation A]
{answer_a}
[The End of Explanation A]
[The Start of Explanation B]
{answer_b}
[The End of Explanation B]
"""
# Assess the explanations based on the following criteria:
# - Helpfulness
# - Relevance
# - Accuracy
# - Depth
# - Clarity
# - Logical consistency

    # Generate the comparison result using the pipeline
    output = pipeline(prompt, max_new_tokens=500, num_return_sequences=1)

    # Extract and return only the generated continuation
    generated_text = output[0]['generated_text']
    continuation = generated_text[len(prompt):].strip()
    return continuation

def load_json(file_path):
    """Load the JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

import re

def extract_responses(data):
    """Extracts the question, response explanations, and ground truth explanations from the provided data."""
    extracted_explanations = []
    count = 0

    for item in data:
        # Get the 'prompt' field
        # if count>=5:
        #     break
        full_prompt = item.get("prompt", "")
        
        # Modify regex to extract the actual question from the prompt
        # The question starts after '</SYS>>\nQuestion:\n' and ends before 'Answer Choices:'
        question_match = re.search(r"</SYS>>\nQuestion:\n(.*?)\nAnswer Choices:", full_prompt, re.DOTALL)
        
        if not question_match:
            print(f"Skipping iteration: Unable to extract question from prompt:\n")
            continue

        question_prompt = question_match.group(1).strip()  # Extracted question
        
        # Extract explanation from the response and ground truth fields
        exp1 = item.get("response", {}).get("explanation", "")
        exp2 = item.get("ground_truth", {}).get("explanation", "")

        # Ensure explanations are present
        if not exp1 or not exp2:
            print(f"Skipping iteration: Missing explanations for extracted question:\n")
            continue

        # Extract the chosen label from the response
        chosen_label = item.get("response", {}).get("chosen_option_label", None)
        if chosen_label is None:
            print(f"Warning: Missing chosen label for question:\n")
            continue

        # Append the extracted data to the results
        extracted_explanations.append({
            "question": question_prompt,
            "answer_a": exp1,
            "answer_b": exp2,
            "chosen_label": chosen_label  # Include the chosen label
        })
        count += 1

    return extracted_explanations


def run_pipeline(json_file_path):
    # Load JSON data
    data = load_json(json_file_path)

    # Extract questions and responses
    extracted_data = extract_responses(data)

    results = []
    model_a_count = 0  # Counter for Assistant A
    model_b_count = 0  # Counter for Assistant B
    tie_count = 0      # Counter for ties
    count=0 
    # Loop through each extracted question-response pair
    for item in extracted_data:
        
        question_prompt = item["question"]
        answer_a = item["answer_a"]
        answer_b = item["answer_b"]

        # Generate comparison result
        comparison_result = compare_responses(question_prompt, answer_a, answer_b)
        #print(comparison_result)
        # Parse the selection from the result
        selection = ""
        if "[A]" in comparison_result:
            selection = "A"
            model_a_count += 1
        elif "[B]" in comparison_result:
            selection = "B"
            model_b_count += 1
        elif "[C]" in comparison_result:
            selection = "C"
            tie_count += 1
        else:
            print(f"Warning: Unable to determine selection for question:\n")
            
        # Append the result
        results.append({
            "question": question_prompt,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "comparison_result": comparison_result,
            "selection": selection
        })
        count += 1

    # Save the results
    with open("comparison_results_str_quan_inst_2.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to 'comparison_result_str_quan_inst_2.json'")
    print(f"Assistant A selected {model_a_count} times.")
    print(f"Assistant B selected {model_b_count} times.")
    print(f"Ties: {tie_count}")

# Path to your JSON file
json_file_path = "test_set_1.json"

# Run the pipeline
run_pipeline(json_file_path)
