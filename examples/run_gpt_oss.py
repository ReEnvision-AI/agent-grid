import time
import torch
from rich.console import Console
from rich.markdown import Markdown
from transformers import AutoTokenizer, TextStreamer, StoppingCriteria, StoppingCriteriaList, EosTokenCriteria

from agentgrid import AutoDistributedModelForCausalLM


class SmartTextStreamer(TextStreamer):
    """Simple streamer for clean output display."""

    def __init__(self, tokenizer, console: Console):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=False)
        self.console = console
        self.tokens_generated = 0
        self.full_text = ""

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.full_text += text
        self.tokens_generated += 1
        self.console.print(text, end="")

def generate_smart(model_id: str) -> None:
    """Generate with smart prompt classification and optimal formatting."""

    # Load the model
    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_name_or_path=model_id,
        torch_dtype=torch.bfloat16,
        initial_peers=["/ip4/127.0.0.1/tcp/31331/p2p/Qmbnu3pqyWvXaehb5si5mpyeVQh7RZMPcrrWBX2gnWcc3D"],
        device_map="auto"
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    console = Console()

    while True:
        # Get the messages from the user
        user_prompt = input("\nYour prompt (or 'quit' to exit): ").strip()

        if user_prompt.lower() in ['quit', 'exit', 'q']:
            break

        if not user_prompt:
            continue

        console.print(f"\n[bold]Model:[/bold] {model_id}")
        console.print(f"[bold]Your prompt:[/bold] {user_prompt}")

        # Classify prompt and get optimal format
        #prompt_type = classify_prompt(user_prompt)
        #formatted_prompt = get_best_prompt_format(user_prompt, prompt_type)
        formatted_prompt = user_prompt

        #console.print(f"[dim]Prompt type: {prompt_type}[/dim]")
        #console.print(f"[dim]Formatted: {formatted_prompt[:50]}...[/dim]")

        # Choose optimal parameters based on prompt type
        #if prompt_type == 'creative':
        #    params = {
        #        "max_new_tokens": 150,
        #        "do_sample": True,
        #        "temperature": 0.8,
        #        "top_p": 0.9,
        #        "repetition_penalty": 1.2,
        #    }
        #elif prompt_type == 'question':
        #    params = {
        #        "max_new_tokens": 200,
        #        "do_sample": False,  # Greedy for factual answers
        #        "repetition_penalty": 1.1,
        #    }
        #elif prompt_type == 'technical':
        #    params = {
        #        "max_new_tokens": 100,  # Shorter for technical
        #        "do_sample": True,
        #        "temperature": 0.5,
        #        "repetition_penalty": 1.2,
        #    }
        #else:  # factual, general
        #eos_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
        params = {
                "max_new_tokens": 8192,
                "do_sample": False,
                "temperature": 0.6,
                "repetition_penalty": 1.2,
                #"eos_token_id": eos_ids,
                #"pad_token_id": tokenizer.pad_token_id
            }

        # Try generation
        try:
            # Tokenize
            messages = [{"role": "user", "content": user_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,               # makes sure we get a str, not token ids
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                return_attention_mask=True,
                add_special_tokens=False,     # chat template already inserted them
            ).to(model.device)

            console.print("\n[bold]Response:[/bold] ", end="")

            # Create streamer
            streamer = SmartTextStreamer(tokenizer, console)

            start_time = time.perf_counter()

            # Generate
            model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                #pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
                **params
            )

            end_time = time.perf_counter()

            console.print()  # New line

            # Clean up the response
            response = streamer.full_text.strip()

            # Remove any weird artifacts
            #if response.endswith(('and', 'the', 'This', 'We can', 'Sure')):
            #    response = response.rsplit('.', 1)[0] + '.' if '.' in response else response

            # Display stats
            console.print(f"\n[dim]Stats: {streamer.tokens_generated} tokens in {end_time - start_time:.1f}s ({streamer.tokens_generated / (end_time - start_time):.1f} t/s)[/dim]")
            
            # Show final cleaned response
            if response:
                console.print("\n[bold green]Final Answer:[/bold green]")
                console.print(Markdown(response))
            else:
                console.print("[bold red]No valid response generated[/bold red]")

        except Exception as e:
            print(e)
            console.print(f"\n[bold red]Error: {e}[/bold red]")

        console.print("\n" + "="*60)


def main():
    model_id = "unsloth/gpt-oss-20b-BF16"
    console = Console()
    console.print("[bold green]Smart GPT-OSS Client[/bold green]")
    console.print("This client automatically optimizes prompts based on content type.")
    console.print("Type 'quit' to exit.\n")

    generate_smart(model_id)


if __name__ == "__main__":
    main()