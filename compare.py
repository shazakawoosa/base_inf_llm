import torch
import transformers
import os
import json

# Set the Hugging Face access token if required
os.environ["HF_TOKEN"] = "your_huggingface_token"

# The model ID for LLaMA
model_id = "meta-llama/Llama-3.1-8B"

# Load the pipeline with the model, setting it to use GPU and torch.bfloat16 for better memory usage
pipeline = transformers.pipeline(
    "text-generation", 
    model=model_id, 
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="auto",  # Automatically selects the device (GPU if available)
)

def compare_explanations(comparison_prompt, exp1, exp2):
    """Generate a comparison of explanations (Model A vs Model B), first selecting the better one, then explaining why."""
    # Construct the full comparison prompt with explanations (Model A and Model B)
    full_prompt = f"""{comparison_prompt}
Model A: {exp1}
Model B: {exp2}
"""

    # Generate the comparison result using the pipeline
    output = pipeline(full_prompt, max_length=500, num_return_sequences=1)

    # Extract and return the generated text (which should include the choice and reasoning)
    return output[0]['generated_text']

def load_json(file_path):
    """Loads the JSON file and returns the data."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_explanations(data):
    """Extracts the response and ground truth explanations without labeling them as such."""
    extracted_explanations = []
    count = 0
    # Loop through each item in the data (iteration)
    for item in data:
        # if count >= 5:
        #     break
        question_prompt = item.get("prompt", "")
        exp1 = item.get("response", {}).get("explanation", "")
        exp2 = item.get("ground_truth", {}).get("explanation", "")

        # If explanations are missing, skip to the next iteration
        if not exp1 or not exp2:
            print(f"Skipping iteration: Missing explanations for prompt: {question_prompt}")
            continue

        extracted_explanations.append({
            "question_prompt": question_prompt,
            "exp1": exp1,
            "exp2": exp2
        })
        count += 1
    
    return extracted_explanations

def run_pipeline(json_file_path, comparison_prompt):
    # Load the JSON file with explanations
    data = load_json(json_file_path)

    # Extract explanations
    extracted_explanations = extract_explanations(data)

    results = []
    model_a_count = 0  # Counter for how many times Model A (response) is selected
    model_b_count = 0  # Counter for how many times Model B (ground_truth) is selected

    # Loop through each extracted explanation pair and compare them
    for item in extracted_explanations:
        question_prompt = item["question_prompt"]
        exp1 = item["exp1"]
        exp2 = item["exp2"]

        # Pass the extracted explanations to the comparison function
        comparison_result = compare_explanations(comparison_prompt, exp1, exp2)

        # Check if the result mentions Model A or Model B as the selected explanation
        if "Model A" in comparison_result:
            model_a_count += 1
        elif "Model B" in comparison_result:
            model_b_count += 1

        # Prepare the result to save in JSON format
        result = {
            "question_prompt": question_prompt,
            "exp1": exp1,
            "exp2": exp2,
            "comparison_result": comparison_result,
        }
        results.append(result)

    # Save the results to a JSON file
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to 'comparison_results.json'")
    print(f"Model A (response) selected {model_a_count} times.")
    print(f"Model B (ground_truth) selected {model_b_count} times.")

# Path to your JSON file
json_file_path = "test_set_1.json"

# Comparison prompt for selecting the better explanation (Model A vs Model B)
comparison_prompt = """
You are asked to compare two explanations for a legal question. 
Select the better explanation based on clarity, depth, and accuracy. 
First, choose either Model A or Model B as the better explanation. 
Then, explain why you selected the chosen explanation over the other one. 
Provide a score from 1 to 10 for each explanation and explain which explanation is better and why.
"""

# Run the pipeline on all iterations in the JSON file
run_pipeline(json_file_path, comparison_prompt)
