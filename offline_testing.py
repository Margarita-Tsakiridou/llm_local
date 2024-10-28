import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify model name (it will load from cache when offline)
model_name = "EleutherAI/gpt-neo-1.3B"

# Load the model and tokenizer from local cache
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the model runs on the appropriate device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define a prompt for "interrogation"
prompt = "Can I run llm inference locally without internet coonection."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text using sampling with temperature and max_length settings
output = model.generate(
    **inputs,
    max_length=100,
    temperature=0.7,  # adjust for creativity level
    do_sample=True,   # enable sampling for diverse output
    pad_token_id=tokenizer.eos_token_id  # set padding token to eos_token
)

# Decode the generated response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)

