import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model_and_tokenizer(model_string: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_string,
        device_map='auto'
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    
    with torch.no_grad():
        ids = tokenizer('pt')
        out = model(**ids)

    del ids, out

    return model, tokenizer


@torch.no_grad()
def run_baseline(target_model, target_tokenizer, inputs, max_new_tokens):

    input_len = inputs["input_ids"].size(1)
    outputs = target_model.generate(
        **inputs, max_new_tokens=max_new_tokens
    )
    return target_tokenizer.batch_decode(
        outputs[:, input_len:], skip_special_tokens=True
    )[0]


@torch.no_grad()
def generate_draft_tokens(draft_model, input_ids, max_draft_tokens):
    draft_tokens = []
    draft_probs  = []

    current_ids = input_ids.clone()

    for _ in range(max_draft_tokens):
        out = draft_model(current_ids)
        logits = out.logits[0, -1]
        probs = torch.softmax(logits, dim=0)

        next_token = torch.multinomial(probs, 1)              # [1]
        token_id = next_token.item()

        draft_tokens.append(token_id)
        draft_probs.append(probs[token_id].item())

        current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

    draft_tokens = torch.tensor(draft_tokens, device=input_ids.device).unsqueeze(0)
    return draft_tokens, draft_probs


@torch.no_grad()
def verify_draft_tokens(draft_tokens, draft_probs, input_ids, target_model):
    seq_len = input_ids.size(1)
    sequence = torch.cat([input_ids, draft_tokens], dim=1)

    accepted_tokens = []

    target_out = target_model(sequence)
    target_logits = target_out.logits[0]

    for i, token_id in enumerate(draft_tokens[0]):
        pos = seq_len - 1 + i

        target_probs = torch.softmax(target_logits[pos], dim=0)
        target_prob = target_probs[token_id].item()
        draft_prob  = draft_probs[i]

        ratio = min(1, target_prob / draft_prob)

        if torch.rand(1).item() < ratio:
            accepted_tokens.append(token_id)
        else:
            # Reject: sample from target distribution
            new_token = torch.multinomial(target_probs, 1).item()
            accepted_tokens.append(new_token)
            break

    accepted_tokens = torch.tensor(accepted_tokens, device=input_ids.device).unsqueeze(0)
    final_sequence = torch.cat([input_ids, accepted_tokens], dim=1)
    return final_sequence


def speculative_decoding(
    target_model, target_tokenizer,
    draft_model, draft_tokenizer,
    prompt: str,
    max_new_tokens: int,
    max_draft_tokens: int
):
    inputs = target_tokenizer([prompt], return_tensors="pt").to(target_model.device)
    base_out = run_baseline(target_model, target_tokenizer, inputs)

    draft_tokens, draft_probs = generate_draft_tokens(
        draft_model, inputs["input_ids"], max_draft_tokens
    )

    final_ids = verify_draft_tokens(
        draft_tokens, draft_probs, inputs["input_ids"], target_model
    )

    result = target_tokenizer.decode(final_ids[0], skip_special_tokens=True)
    return base_out, result

def run(
    target: str, 
    draft: str, 
    prompts: List[str], 
    max_new_tokens: int,
    max_draft_tokens: int
):
    print("SPECULATIVE DECODING RUN")
    print("==="*80)
    print(f"Target model: {target}")
    print(f"Draft model: {draft}")
    print(f"Maximum new tokens: {max_new_tokens}")
    print(f"Maximum draft tokens: {max_draft_tokens}")
    print("==="*80)

    target_model, target_tokenizer = get_model_and_tokenizer(target)
    draft_model, draft_tokenizer   = get_model_and_tokenizer(draft)

    for prompt in prompts:
        base_out, result = speculative_decoding(
            target_model, target_tokenizer,
            draft_model, draft_tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            max_draft_tokens=max_draft_tokens
        )
        print(base_out, result)

    print("Finished speculative decoding")
