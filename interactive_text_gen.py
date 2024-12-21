import torch
import transformers
import argparse

def initialize_pipeline(model_id):
    """Initialize the text generation pipeline."""
    try:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto"  # Automatically selects GPU if available
        )
        print("Pipeline initialized successfully.")
        return pipeline
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        exit(1)

def interactive_generation(pipeline):
    """Allow interactive text generation using the pipeline."""
    print("\nInteractive Text Generation")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Enter your prompt: ")
            if user_input.lower() == "exit":
                print("Exiting interactive text generation. Goodbye!")
                break

            # Generate text with temperature and top_k sampling for diversity
            output = pipeline(
                user_input,
                max_new_tokens=200, 
                num_return_sequences=1,
                temperature=0.7,  # Adjust for creativity
                top_k=50,         # Controls randomness by limiting token sampling
                do_sample=True    # Ensure random sampling instead of greedy decoding
            )
            generated_text = output[0]["generated_text"]
            print(f"\nGenerated Text:\n{generated_text}\n")
        except Exception as e:
            print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Text Generation")
    parser.add_argument("--model_id", type=str, default="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
                        help="The Hugging Face model ID to use for text generation.")
    args = parser.parse_args()

    # Load the pipeline
    print(f"Loading model: {args.model_id}")
    pipeline = initialize_pipeline(args.model_id)

    # Start the interactive text generation loop
    interactive_generation(pipeline)
