# -*- coding: utf-8 -*-
"""Mistral_finetune_inference

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RM8gwEu9Ft0UEo2qqLfH-zZejx31DPLc
"""

import random

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# # Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes

"""# Functions"""

if __name__ == "__main__":
    from transformers import GenerationConfig, TextStreamer

    def generate(prompt, model, tokenizer, max_new_tokens=1024):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        model.eval()
        with torch.no_grad():
            generation_config = GenerationConfig(
                repetition_penalty=1.0,
                max_new_tokens=max_new_tokens,
                # temperature=0.4,
                # top_p=0.95,
                # top_k=20,
                # bos_token_id=tokenizer.bos_token_id,
                # eos_token_id=tokenizer.eos_token_id,
                # eos_token_id=0, # for open-end generation.
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
            # streamer = TextStreamer(tokenizer, skip_prompt=True)
            streamer = None
            generated = model.generate(
                inputs=input_ids,
                generation_config=generation_config,
                # stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
        gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]) :]
        output = tokenizer.batch_decode(gen_tokens)[0]
        output = output.split(tokenizer.eos_token)[0]
        return output.strip()

    """# Alpaca"""

    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = (
        torch.float16  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    )
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    alpaca_model, alpaca_tokenizer = FastLanguageModel.from_pretrained(
        model_name="siddankthep/Alpaca-10K-Mistral-7B-QA",  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    FastLanguageModel.for_inference(alpaca_model)  # Enable native 2x faster inference

    """## Q&A functions"""

    from transformers import TextStreamer

    def generate_question_alpaca(model, tokenizer, context):
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
        ### Instruction:
        {}
    
        ### Input:
        {}
    
        ### Response:
        {}"""
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    "Generate 10 simple questions that can be answered from the following context.",  # instruction
                    context,  # input
                    "",  # output - leave this blank for generation!
                )
            ],
            return_tensors="pt",
        ).to("cuda")

        text_streamer = TextStreamer(tokenizer)
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

    def answer_alpaca(model, tokenizer, context, question):
        from transformers import TextStreamer

        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
        ### Instruction:
        {}
    
        ### Input:
        Question: {}
        
        Context: {}
    
        ### Response:
        {}"""

        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    "Answer the following question strictly based on the given context.",  # instruction
                    question,
                    context,  # input
                    "",  # output - leave this blank for generation!
                )
            ],
            return_tensors="pt",
        ).to("cuda")

        text_streamer = TextStreamer(tokenizer)
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

    def get_paragraphs(file_path):
        with open("text_data.txt", "r") as f:
            lines = f.readlines()

        paragraphs = []
        for line in lines:
            if line != "\n":
                paragraphs.append(line.strip("\n"))

        return paragraphs

    """## Inference"""

    paragraphs = get_paragraphs(
        "/home/quannguyen57/Sid/text_data.txt"
    )  # Change the path to the text_data.txt file

    task = int(
        input(
            "Choose what task you would like to test:\n[1] Question generation\n[2] Question answering\nYour choice: "
        )
    )

    if task == 1:
        content = (
            input("Enter a context: ")
            or paragraphs[random.randint(0, len(paragraphs) - 1)]
        )

        print(content)

        print("Question generation:\n __________________________________________")
        print(generate_question_alpaca(alpaca_model, alpaca_tokenizer, content))

    if task == 2:
        content = (
            input("Enter a context: ")
            or paragraphs[random.randint(0, len(paragraphs) - 1)]
        )

        print(content)
        question = input("Enter a question: ") or (
            "How do weights affect the strength of signals in an artificial neural network?"
        )

        print("Question answering:\n __________________________________________")
        print(answer_alpaca(alpaca_model, alpaca_tokenizer, content, question))
