import torch

def get_binary_prediction(model, tokenizer, prompt, choice_a_token=' A', choice_b_token=' B'):
    """
    Extracts the next-token logits for specific choices to perform binary classification.
    Bypasses text generation to ensure deterministic evaluation and avoid decoding artifacts.
    """
    # Tokenize input and move to the same device as the model (e.g., CUDA)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Extract the specific token IDs for ' A' and ' B'. 
    # Note: Leading spaces are crucial as most LLM tokenizers treat 'A' and ' A' as different tokens.
    token_id_a = tokenizer.encode(choice_a_token, add_special_tokens=False)[-1]
    token_id_b = tokenizer.encode(choice_b_token, add_special_tokens=False)[-1]
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Focus on the last token's hidden state to predict the next token
        next_token_logits = outputs.logits[0, -1, :]
    
    # Extract raw logit values for the two target tokens
    logit_a = next_token_logits[token_id_a].item()
    logit_b = next_token_logits[token_id_b].item()
    
    # Determine prediction based on the higher logit value
    prediction = 'A' if logit_a > logit_b else 'B'
    
    return prediction, logit_a, logit_b
