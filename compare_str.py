import torch
import transformers
import os
import json
import re
import torch
import time
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


# # Set up the text generation pipeline
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# #model_id = "meta-llama/Llama-3.1-8B"

print(model_name)


# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,  # Use FP16 for further optimizations
#     device_map="auto",          # Automatically place layers on GPU(s)
#     #load_in_4bit=True           # Use 8-bit quantization (set to `load_in_4bit=True` for 4-bit)
# )

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
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

def extract_responses(data):
    """Extracts the question, response explanations, and ground truth explanations from the provided data."""
    extracted_explanations = []

    for item in data:
        # Get the 'prompt' field
        full_prompt = item.get("prompt", "")

        # Extract the question from the prompt
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
        correct_answer = item.get("ground_truth", {}).get("correct_answer", None)
        
        if chosen_label is None or correct_answer is None:
            print(f"Warning: Missing chosen label or correct answer for question:\n")
            continue

        # Append the extracted data to the results
        extracted_explanations.append({
            "question": question_prompt,
            "answer_a": exp1,
            "answer_b": exp2,
            "chosen_label": chosen_label,  # Include the chosen label
            "correct_answer": correct_answer  # Include the correct answer
        })

    return extracted_explanations

def run_pipeline(json_file_path):
    # Load JSON data
    data = load_json(json_file_path)

    # Extract questions and responses
    extracted_data = extract_responses(data)

    results = []
        # Initialize counts
    model_a_count = 0
    model_b_count = 0
    tie_count = 0
    incorrect_count = 0  # New counter for chosen_label != correct_answer
    start_time = time.time()     
    print("start time",start_time)   
    # Iterate through the dataset
    for item in extracted_data:
        question_prompt = item["question"]
        answer_a = item["answer_a"]
        answer_b = item["answer_b"]
        chosen_label = item["chosen_label"]
        correct_answer = item["correct_answer"]

        # Generate comparison result
        comparison_result = compare_responses(question_prompt, answer_a, answer_b)

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
            print(f"Warning: Unable to determine selection for question:\n{question_prompt}")

        # Check if the chosen_label is not equal to the correct_answer
        if chosen_label != correct_answer:
            incorrect_count += 1  # Increment the incorrect count
             # Append the result
        results.append({
            "question": question_prompt,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "comparison_result": comparison_result,
            "selection": selection,
            "chosen_label": chosen_label,
            "correct_answer": correct_answer
        })
    # End the timer
    end_time = time.time()

    # Calculate total execution time
    execution_time = end_time - start_time

    # Output the result
    print(f"Model A chosen count: {model_a_count}")
    print(f"Model B chosen count: {model_b_count}")
    print(f"Tie count: {tie_count}")
    print(f"Number of incorrect labels: {incorrect_count}")
    print(f"Execution time: {execution_time:.2f} seconds")


    # Save the results
    with open("comparison_results_inst_quan.json", "w") as f:
        json.dump(results, f, indent=4)

        

    # Path to your JSON file
json_file_path = "test_set_1.json"

    # Run the pipeline
run_pipeline(json_file_path)
