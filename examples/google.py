from xplain.executor import XPLAINMetricCalculator

LLM_MODEL_NAME="gemini-2.5-flash"
EMBEDDER_MODEL_NAME="gemini-embedding-001"

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

xplain_calculator = XPLAINMetricCalculator()

xplain_calculator.select_llm("GOOGLE_AI_STUDIO", **llm_args)
xplain_calculator.select_embedder("GOOGLE_AI_STUDIO", **embedder_args)

prompt = "Why do we need sleep?"
scores = xplain_calculator.compute_score(prompt)

print(f"XPLAIN Scores for the prompt: '{prompt}' :")
print("-" * 25)
for word, score in scores:
  print(f"{word:<10} | Score: {score:.4f}")
print("-" * 25)