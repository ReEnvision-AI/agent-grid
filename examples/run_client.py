import time

import torch
from rich.console import Console
from rich.markdown import Markdown
from transformers import AutoTokenizer

from agentgrid import AutoDistributedModelForCausalLM

MODELS = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
]


def selectmodel() -> str:
    print("Please select a model:")
    for i, model in enumerate(MODELS):
        print(f"   {i+1}: {model}")

    selection = -1

    while selection < 0 or selection > len(MODELS):
        i = input("Selection: ")
        try:
            selection = int(i)
        except:
            selection = -1

    return MODELS[selection - 1]


def generate(model_id: str) -> None:
    # Load the model
    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_name_or_path=model_id,
        torch_dtype=torch.bfloat16,
        initial_peers=["/ip4/127.0.0.1/tcp/31331/p2p/Qmbnu3pqyWvXaehb5si5mpyeVQh7RZMPcrrWBX2gnWcc3D"],
        #attn_implementation="flash_attention_2",
    ).to("cuda")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Get the messages from the user
    user_prompt = input("User prompt: ")
    messages = [{"role": "system", "content": "You are a very helpful assistant. detailed thinking off /no_think"}, {"role": "user", "content": user_prompt}]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    n_inputs = inputs["input_ids"].size(1)
    print(f"Number of tokens: {n_inputs}")

    with torch.inference_mode():
        start_time = time.perf_counter()
        response = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=512)
        end_time = time.perf_counter()

    n_outputs = len(response[0, n_inputs:])
    total_time = end_time-start_time
    tps = n_outputs / total_time

    output = tokenizer.decode(response[0, n_inputs:], skip_special_tokens=True)

    console = Console()
    console.print(Markdown(output))

    console.print(f"Tokens per second: {tps:.4f}")


def main():
    model_id = selectmodel()

    print(f"You have selected {model_id}")

    generate(model_id)


if __name__ == "__main__":
    main()
