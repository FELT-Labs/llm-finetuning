# LLM Fine-tuning
Algorithms and instructions for fine-tuning LLMs with FELT Labs

Full tutorial: <https://medium.com/@breta.hajek/fine-tuning-large-language-models-with-felt-labs-full-guide-3f1c3fcc9af6>

_Does your project have a specific need for fine-tuning LLMs? Contact us at [support@feltlabs.ai](mailto:support@feltlabs.ai), and our team will help you with that!_


## Publishing Algorithm
Provider: `https://provider.feltlabs.ai/`  
File URL: `https://raw.githubusercontent.com/FELT-Labs/llm-finetuning/main/algorithm.py`  
Entry point: `python3 $ALGO`  
Docker image: `feltlabs/llm-compute:latest`  

## Publishing Dataset
Access type: `Compute`
Provider: `https://provider.feltlabs.ai/`  
File URL: `https://raw.githubusercontent.com/FELT-Labs/llm-finetuning/main/dataset.json`

## Run Fine-Tuning
Go to <https://app.feltlabs.ai/learning/single> and do following steps:

1. Select your dataset
2. Select fine-tuning algorithm
3. Pick hyperparameters
4. Start training

## Inference
Use [inference.ipynb](./inference.ipynb) notebook for running the inference. You can use it through Google Colab: <https://colab.research.google.com/github/FELT-Labs/llm-finetuning/blob/main/inference.ipynb>

