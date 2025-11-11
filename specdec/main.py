import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from tqdm import tqdm


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

def get_model_and_tokenizer(model_string: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_string,
        device_map='auto'
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_string)
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
    draft_tokens = torch.empty(max_draft_tokens, dtype=torch.long, device=input_ids.device)
    draft_probs  = torch.empty(max_draft_tokens, device=input_ids.device)

    out = draft_model(input_ids, use_cache=True)
    past = out.past_key_values

    for i in range(max_draft_tokens):
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, 1)  # [1,1]
        token_id = next_token.item()

        draft_tokens[i] = token_id
        draft_probs[i] = probs[0, token_id]

        out = draft_model(next_token, use_cache=True, past_key_values=past)
        past = out.past_key_values

    draft_tokens = draft_tokens.unsqueeze(0)
    return draft_tokens, draft_probs


@torch.no_grad()
def verify_draft_tokens(draft_tokens, draft_probs, input_ids, target_model):
    seq_len = input_ids.size(1)
    sequence = torch.cat([input_ids, draft_tokens], dim=1)

    accepted = []

    target_out = target_model(sequence)
    target_logits = target_out.logits[0]

    num_accepted = 0
    num_draft = draft_tokens.shape[0]

    for i, token_id in enumerate(draft_tokens[0]):
        pos = seq_len - 1 + i

        target_probs = torch.softmax(target_logits[pos], dim=0)
        target_prob = target_probs[token_id]
        draft_prob  = draft_probs[i]

        ratio = min(1.0, (target_prob / draft_prob).item())

        if torch.rand(1).item() < ratio:
            accepted.append(token_id.item())
            num_accepted += 1
        else:
            new_token = torch.multinomial(target_probs, 1).item()
            accepted.append(new_token)
            break

    num_rejected = num_draft - num_accepted 

    accepted = torch.tensor(accepted, device=input_ids.device).unsqueeze(0)
    final_sequence = torch.cat([input_ids, accepted], dim=1)

    del target_out, target_logits
    return final_sequence, num_rejected, num_accepted


def speculative_decoding(
    target_model, target_tokenizer,
    draft_model,
    prompt: str,
    max_new_tokens: int,
    max_draft_tokens: int
):
    inputs = target_tokenizer([prompt], return_tensors="pt").to(target_model.device)

    with Timer() as base_t:
        base_out = run_baseline(target_model, target_tokenizer, inputs, max_new_tokens)

    with Timer() as spec_t:
        draft_tokens, draft_probs = generate_draft_tokens(
            draft_model, inputs["input_ids"], max_draft_tokens
        )

        final_ids, num_rejected, num_accepted = verify_draft_tokens(
            draft_tokens, draft_probs, inputs["input_ids"], target_model
        )

    results = {
        'num_accepted': num_accepted,
        'num_rejected': num_rejected,
        'base_time': base_t.time_elapsed,
        'spec_time': spec_t.time_elapsed,
    }
    

    spec_out = target_tokenizer.decode(final_ids[0], skip_special_tokens=True)
    return results, spec_out, base_out


def run(
    target: str,
    draft: str,
    prompts: List[str],
    max_new_tokens: int,
    max_draft_tokens: int,
    plot_times: bool = True
):
    print("SPECULATIVE DECODING RUN")
    print("==="*80)
    print(f"Target model: {target}")
    print(f"Draft model: {draft}")
    print(f"Maximum new tokens: {max_new_tokens}")
    print(f"Maximum draft tokens: {max_draft_tokens}")
    print("==="*80)

    target_model, target_tokenizer = get_model_and_tokenizer(target)
    draft_model, _ = get_model_and_tokenizer(draft)

    base_times, spec_times = [], []
    if prompts is [] or prompts is None:
        prompts = PROMPTS

    for prompt in tqdm(prompts, desc="SpecDec", ncols=80):
        results, spec_out, base_out = speculative_decoding(
            target_model, target_tokenizer,
            draft_model,
            prompt,
            max_new_tokens=max_new_tokens,
            max_draft_tokens=max_draft_tokens
        )

        base_times.append(results['base_time'])
        spec_times.append(results['spec_time'])

        cleanup()

    print("Finished speculative decoding")

    if plot_times:
        plot_timings(base_times, spec_times)

    del target_model, draft_model
    return base_times, spec_times
