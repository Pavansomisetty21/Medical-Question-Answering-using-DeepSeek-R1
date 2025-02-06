#Medical Question Answering using DeepSeek-R1 with prompt-based generation

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the prompt template
prompt_template = """Below is an condition that describes a task,  with an input that provides further context.
Write a response that accurately completes the task.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

condition:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning and you can answer the question on medical.
Please answer the following medical question.

Question:
{}

Response:
<think>{}"""

def generate_response(model, tokenizer, question):
    # Format the prompt using the structured template
    formatted_prompt = prompt_template.format(question, "")

    # Tokenize the input using the updated format
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

    # Generate a response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode the response and extract the output after ' Response:'
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response.split("Response:")[1].strip() if " Response:" in response else response.strip()

# Load Model & Tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
modelprompt = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example Question
question = "what medicies should we take for a severe headache and fever?"

print("\n\nQuestion: " + question + "\n\n")
print("Answer:\n" + "-"*50)
print(generate_response(modelprompt, tokenizer, question))



# Example Question 2
question = "How to save someone from eternal depression paired up with ADHD?"

print("\n\nQuestion: " + question + "\n\n")
print("Answer:\n" + "-"*50)
print(generate_response(modelprompt, tokenizer, question))