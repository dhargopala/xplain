# XPLAIN: XAI for Interpretable LLMs through Perturbation Analysis and Normalized Vector Similarity

## Introduction
![XPLAIN architecture](assets/architecture.png?raw=true)

##### XPLAIN is a word level score that quantifies the importance of each word in the given input prompt.
We propose an approach that perturbs the prompt via masking words to generate several outputs through the LLM and then compares the outputs with the generated output of the unperturbed prompt through vector-based similarity at a semantic level. After this, the word level importance of each masked word in the perturbed sentences is mathematically derived and referred to as the XPLAIN metric.

## Instructions
This repository contains code to run generate the XPLAIN metric for a given input prompt. 

Currently models from the family of `VERTEX_AI` and `GOOGLE_AI_STUDIO` are supported for LLMs and the Embddeing Models, these include the latest Gemini models aswell.

In order to use the `GOOGLE_AI_STUDIO` models please ensure that [API Key](https://aistudio.google.com/app/apikey) is set as an environment variable.

```sh
export GOOGLE_AI_STUDIO_API_KEY="YOUR_API_KEY"
```

## Installation
To install the package simply run: 
```sh
pip install git+https://github.com/dhargopala/xplain.git
```

## Usage

`examples/google.py` contains the code needed to generate the XPLAIN score for the input prompt.
This examples utilizes Google AI Studio models.

We first import the package and define the constants.

```python
from xplain.executor import XPLAINMetricCalculator

LLM_MODEL_NAME="gemini-2.5-flash"
EMBEDDER_MODEL_NAME="gemini-embedding-001"
```

After setting the global values we define the LLM and Embedding model's arguments.

```python
llm_args = {
    "MODEL_NAME": LLM_MODEL_NAME,
    "output_tokens": 65535,   
    "top_k": 40,
    "top_p": 0.95,
    "temperature": 0
}
embedder_args = {
    "MODEL_NAME": EMBEDDER_MODEL_NAME
}
```

After setting the relevant API arguments, the execution just requires a few lines of code:


```python
xplain_calculator = XPLAINMetricCalculator()

xplain_calculator.select_llm("GOOGLE_AI_STUDIO", **llm_args)
xplain_calculator.select_embedder("GOOGLE_AI_STUDIO", **embedder_args)

scores = xplain_calculator.compute_score("Why do we need sleep?")
```

Sample Output:

```
XPLAIN Scores for the prompt: 'Why do we need sleep?' :
-------------------------
Why        | Score: 0.1613
do         | Score: 0.0188
we         | Score: 0.0000
need       | Score: 0.0785
sleep?     | Score: 1.0000
-------------------------
```