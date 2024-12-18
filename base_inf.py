import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
  "text-generation",
  model="meta-llama/Meta-Llama-3.1-8B-Instruct",
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
)
prompt = "Explain the theory of relativity in simple terms."
output = pipeline(prompt, max_length=100, num_return_sequences=1)

print(output[0]["generated_text"])
