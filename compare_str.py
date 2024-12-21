import torch
import transformers
import os
import json
import re
# Set the Hugging Face access token if required
# os.environ["HF_TOKEN"] = "your_huggingface_token"

# The model ID for LLaMA
model_id = "meta-llama/Llama-3.1-8B"

# Load the pipeline with the model, setting it to use GPU and torch.bfloat16 for better memory usage
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "pad_token_id": 128001,  # Set explicitly to suppress warnings
    },
    device_map="auto",  # Automatically selects the device (GPU if available)
    num_return_sequences=1,
    temperature=0.7,  # Adjust for creativity
    top_k=50,         # Controls randomness by limiting token sampling
    do_sample=True    # Ensure random sampling instead of greedy decoding
)

def compare_responses(question, answer_a, answer_b):
    """Generate a comparison of responses using the given prompt."""
    prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "A" if assistant A is better, "B" if assistant B is better, and "C" for a tie.

[Example Evaluation]
[User Question]
What is the capital of France?
[The Start of Assistant A’s Answer]
The capital of France is Paris. It is known for its history, art, and culture. Paris is home to iconic landmarks like the Eiffel Tower and the Louvre Museum.
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
France’s capital is Paris. It is a major European city and a global center for art, fashion, and culture.
[The End of Assistant B’s Answer]
[Evaluation Explanation]
Both assistants correctly identified Paris as the capital of France. However, Assistant A provided additional context about Paris, mentioning its historical and cultural significance, as well as iconic landmarks, which adds depth and detail to the response. Assistant B's answer is accurate but less detailed. Therefore, Assistant A's response is more helpful and comprehensive.

[Final Verdict]
A
Please do not treat this example evaluation as the baseline but only as an example. Therefore, form your own explanations and the final asnwer based on your reasoning.
[New Evaluation]
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
"""

    # Generate the comparison result using the pipeline
    output = pipeline(prompt, max_new_tokens=500, num_return_sequences=1)

    # Extract and return only the generated continuation
    generated_text = output[0]['generated_text']
    continuation = generated_text[len(prompt):].strip()  # Remove the prompt from the result
    return continuation

def load_json(file_path):
    """Loads the JSON file and returns the data."""
    with open(file_path, 'r') as f:
        return json.load(f)

import re

def extract_responses(data):
    """Extracts the question, response explanation, and ground truth explanation from the provided data."""
    extracted_explanations = []
    count = 0

    for item in data:
        # Get the 'prompt' field
        full_prompt = item.get("prompt", "")
        
        # Extract the question text between <<SYS>> and "Response:"
        question_match = re.search(r"<<SYS>>.*?Question:(.*?)Answer Choices:", full_prompt, re.DOTALL)
        if not question_match:
            print(f"Skipping iteration: Unable to extract question from prompt:\n{full_prompt}")
            continue

        question_prompt = question_match.group(1).strip()

        # Extract explanation from the response and ground truth fields
        exp1 = item.get("response", {}).get("explanation", "")
        exp2 = item.get("ground_truth", {}).get("explanation", "")

        # Ensure explanations are present
        if not exp1 or not exp2:
            print(f"Skipping iteration: Missing explanations for extracted question:\n")
            continue

        # Append the extracted data to the results
        extracted_explanations.append({
            "question": question_prompt,
            "answer_a": exp1,
            "answer_b": exp2
        })
        count += 1

    return extracted_explanations


def run_pipeline(json_file_path):
    # Load the JSON file with responses
    data = load_json(json_file_path)

    # Extract questions and responses
    extracted_data = extract_responses(data)

    results = []
    model_a_count = 0  # Counter for how many times Assistant A is selected
    model_b_count = 0  # Counter for how many times Assistant B is selected
    tie_count = 0      # Counter for ties
    count =0
    # Loop through each extracted question-response pair and compare them
    for item in extracted_data:
        # if count >=5:
        #     break
        question = item["question"]
        answer_a = item["answer_a"]
        answer_b = item["answer_b"]

        # Pass the question and responses to the comparison function
        comparison_result = compare_responses(question, answer_a, answer_b)
        #print("output",comparison_result)
        # test_prompt = "What is the capital of France?"
        # print(pipeline(test_prompt, max_new_tokens=50))

        # Parse the selection from the result
        selection = ""
        if "A" in comparison_result:
            selection = "A"
            model_a_count += 1
        elif "B" in comparison_result:
            selection = "B"
            model_b_count += 1
        elif "C" in comparison_result:
            selection = "C"
            tie_count += 1
        else:
            print(f"Warning: Unable to determine selection for question:\n")
            
        # Prepare the result to save in JSON format
        results.append({
            "question": question,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "comparison_result": comparison_result,
            "selection": selection
        })
        count+=1

    # Save the results to a JSON file
    with open("comparison_results_str.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to 'comparison_results_str.json'")
    print(f"Assistant A selected {model_a_count} times.")
    print(f"Assistant B selected {model_b_count} times.")
    print(f"Ties: {tie_count}")

# Path to your JSON file
json_file_path = "test_set_1.json"

# Run the pipeline on all iterations in the JSON file
run_pipeline(json_file_path)
