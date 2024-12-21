import torch
import transformers
import os
import json

# Set the Hugging Face access token if required
# os.environ["HF_TOKEN"] = "your_huggingface_token"

# The model ID for LLaMA
model_id = "meta-llama/Llama-3.1-8B"

# Load the pipeline with the model, setting it to use GPU and torch.bfloat16 for better memory usage
pipeline = transformers.pipeline(
    "text-generation", 
    model=model_id, 
    model_kwargs={"torch_dtype": torch.float16,
                  "pad_token_id": 128001  },# Use the eos_token_id for padding}, 
    device_map="auto",  # Automatically selects the device (GPU if available)
    num_return_sequences=1,
    temperature=0.7,  # Adjust for creativity
    top_k=5,         # Controls randomness by limiting token sampling
    do_sample=True    # Ensure random sampling instead of greedy decoding
)

def compare_explanations(comparison_prompt, exp1, exp2):
    """Generate a comparison of explanations (Model A vs Model B) in structured format."""
    # Construct the full comparison prompt with explanations (Model A and Model B)
    full_prompt = f"""{comparison_prompt}
Exp_A: {exp1}
Exp_B: {exp2}
"""
    # Generate the comparison result using the pipeline
    output = pipeline(full_prompt, max_new_tokens=500, num_return_sequences=1)

    # Extract and return only the generated continuation
    generated_text = output[0]['generated_text']
    continuation = generated_text[len(full_prompt):].strip()  # Remove the prompt from the result

    # Parse the output for structured data (Selection and Reasoning)
    structured_output = {
        "Selection": None,
        "Reasoning": None
    }

    # Split the output into "Selection" and "Reasoning" sections
    if "Selection:" in continuation and "Reasoning:" in continuation:
        parts = continuation.split("Reasoning:", 1)
        structured_output["Selection"] = parts[0].replace("Selection:", "").strip()
        structured_output["Reasoning"] = parts[1].strip()

    if "Selection:" not in continuation or "Reasoning:" not in continuation:
        structured_output["selection"] = "Unable to determine"
        structured_output["reasoning"] = continuation  # Include raw output for debugging


    return structured_output

def load_json(file_path):
    """Loads the JSON file and returns the data."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_explanations(data):
    """Extracts the response and ground truth explanations without labeling them as such."""
    extracted_explanations = []
    # Loop through each item in the data (iteration)
    for item in data:
        question_prompt = item.get("prompt", "")
        exp1 = item.get("response", {}).get("explanation", "")
        exp2 = item.get("ground_truth", {}).get("explanation", "")

        # If explanations are missing, skip to the next iteration
        if not exp1 or not exp2:
            print(f"Skipping iteration: Missing explanations for prompt: {question_prompt}")
            continue

        extracted_explanations.append({
            "Question_prompt": question_prompt,
            "Exp_A": exp1,
            "Exp_B": exp2
        })
    
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
        question_prompt = item["Question_prompt"]
        exp1 = item["Exp_A"]
        exp2 = item["Exp_B"]

        # Pass the extracted explanations to the comparison function
        comparison_result = compare_explanations(comparison_prompt, exp1, exp2)

        # Count selections
        selection = comparison_result.get("Selection", None)  # Use None as the default value

        if selection is None:
            # Handle the case where the selection is None
            print(f"Warning: Unable to determine selection for prompt:\n")
            #print(f"Model output: {comparison_result}")
        else:
            if "Exp_A" in selection:
                model_a_count += 1
            elif "Exp_B" in selection:
                model_b_count += 1
            else:
                # Handle unexpected cases where the selection is non-empty but unrecognized
                print(f"Warning: Unexpected selection format for prompt:\n")
               # print(f"Model output: {comparison_result}")


        # Prepare the result to save in JSON format
        result = {
            "Question_prompt": question_prompt,
            "Exp_A": exp1,
            "Exp_B": exp2,
            "Selection": comparison_result.get("Selection"),
            "Reasoning": comparison_result.get("Reasoning")
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
comparison_prompt = """You are given two explanations of the same question or concept. For each explanation, provide a detailed evaluation based on the following criteria:
1. **Clarity**: How clearly is the explanation written? Is it easy to understand?
2. **Accuracy**: Does the explanation accurately convey the correct information? Are there any factual errors or misleading statements?
3. **Depth**: How detailed is the explanation? Does it cover all necessary aspects of the topic, or is it overly simplistic?
4. **Relevance**: Is the explanation focused and relevant to the question or topic? Does it avoid unnecessary information?
5. **Conciseness**: Is the explanation concise without omitting essential details, or is it too wordy?

For each explanation, give a score from 1 to 10 on each criterion and explain your reasoning behind each score.

Finally, select the better explanation (Exp A or Exp B) and provide your reasoning in the following format:
Selection: [Exp_A/Exp_B]
Reasoning: [Explain why you selected this explanation, including strengths and weaknesses of both explanations based on the criteria.]

Exp A: [First explanation]
Exp B: [Second explanation]
"""

# Run the pipeline on all iterations in the JSON file
run_pipeline(json_file_path, comparison_prompt)
