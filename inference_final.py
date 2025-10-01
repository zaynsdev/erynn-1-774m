import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Paths to model and adapter
MODEL_PATH = r"ORİGİNAL HUGGING FACE OPENAI/GPT-2 REPO MODEL PATH AND WEIGHTS"
ADAPTER_PATH = r"LORA-ADAPTER-PATH"

def load_model():
    """Load the model and tokenizer."""
    # Load model with low memory usage
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16
    )
    # Add LoRA adapter / fp4 FORMAT TENSOR
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_response(model, tokenizer, instruction, context=None):
    """
    Generate a response for the given instruction and optional context.
    Example: get_response(model, tokenizer, "Write an ad for a phone")
    """
    # Build simple prompt
    prompt = f"Instruction: {instruction}\n"
    if context and context.strip():
        prompt += f"Context: {context}\n"
    prompt += "Response: "
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,  # Short and focused responses
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,  # Added for warnings
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response_start = response.find("Response: ") + len("Response: ")
    return response[response_start:].strip()

def main():
    """Run example instructions to test the model."""
    print("Erynn is ready! Testing some examples...\n")
    
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Test 1: Short explanation
    print("Test 1: Explain AI briefly")
    response = get_response(model, tokenizer, "Explain artificial intelligence in 50 words or less.")
    print(response, "\n" + "-"*40)
    
    # Test 2: Summarization
    print("\nTest 2: Summarize this text")
    context = "Deep learning is a key AI technology. It excels in computer vision and natural language processing, driving advances in image recognition and speech synthesis."
    response = get_response(model, tokenizer, "Summarize this text in 30 words or less.", context)
    print(response, "\n" + "-"*40)
    
    # Test 3: Advertisement
    print("\nTest 3: Write a smartwatch ad")
    response = get_response(model, tokenizer, "Write a short advertisement for a smartwatch in 40 words.")
    print(response, "\n" + "-"*40)
    
    # Test 4: List
    print("\nTest 4: List Python advantages")
    response = get_response(model, tokenizer, "List three advantages of Python programming.")
    print(response)
    
    print("\nTry your own instruction: get_response(model, tokenizer, 'Your instruction here')")

if __name__ == "__main__":

    main()
