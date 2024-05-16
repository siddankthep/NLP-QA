# NLP-QA

Question-Answering with MistralAI

# Dependency installation (if run unsloth on local machine)

1. First you'll need miniconda3 to set up environments
   ````mkdir -p ~/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm -rf ~/miniconda3/miniconda.sh```
   ````
2. Install dependencies for using unsloth

   ````
   conda create --name unsloth_env python=3.10
   conda activate unsloth_env
   conda install pytorch-cuda=<12.1/11.8> pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   pip install --no-deps trl peft accelerate bitsandbytes```
   ````

# Running the finetune scripts

## With unsloth

Make sure that you are in your unsloth_env

```
conda activate unsloth_env
```

Run the fine_tune scripts

```
python3 mistral_finetune_alpaca_unsloth.py
```

If gpu cannot run unsloth, you can run the follwing notebook on Google Colab using the free Tesla T4 GPU

[Try the Google Colab finetune version](https://colab.research.google.com/drive/1MoY4jCCFHNtQfdJTui3x2fBEOhg3mgRS?usp=sharing){:target="\_blank"}

## Without unsloth

If you do not use unsloth, use HuggingFace's transformers library instead, which will be slower

```
python3 mistral_finetune_alpaca_hf.py
```

# Inference

To run inference, run

```
python3 mistral_inference_alpaca.py
```
