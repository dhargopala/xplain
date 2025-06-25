import typing as t

import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from xplain.llms.vertexai import VertexAILLMFactory
from xplain.llms.google import GoogleAIStudioLLMFactory
from xplain.embeddings.vertexai import VertexAIEmbedderFactory

llm_factory_mapping = {
        "VERTEX_AI": VertexAILLMFactory,
        "GOOGLE_AI_STUDIO": GoogleAIStudioLLMFactory
    }

embedder_factory_mapping = {
        "VERTEX_AI": VertexAIEmbedderFactory
    }

class XPLAINMetricCalculator:
    def select_llm(self, name, **kwargs):
        llm_factory_class = llm_factory_mapping.get(name)
        if not llm_factory_class:
            raise ValueError(f"Unsupported LLM type: {name}, Supported LLM families: {list(llm_factory_mapping.keys())}")
        self.llm = llm_factory_class.create_llm(**kwargs)

    def select_embedder(self, name, **kwargs):
        embedder_factory_class = embedder_factory_mapping.get(name)
        if not embedder_factory_class:
            raise ValueError(f"Unsupported Embedder type: {name}, Supported Embedder types: {list(embedder_factory_mapping.keys())}")
        self.embedder = embedder_factory_class.create_embedder(**kwargs)

    def _create_perturbations(self, sentence:str, n:int=1) -> t.List[str]:
        words = sentence.split()
        perturbations = []
        masked_words = []
        for i in range(len(words)):
            perturbation = words[:i] + words[i+n:]
            perturbations.append(' '.join(perturbation))
            masked_words.append(words[i])
        return perturbations, masked_words
    
    def _original_prompt_outputs(self, sentence:str):
        self.original_prompt_output = self.llm.generate_text(sentence)
        self.original_prompt_output_embeddings = [self.embedder.generate_embeddings(
                                                        self.original_prompt_output)]

    def _compute_output_and_embeddings(self, sentence: str,
                                       perturbations: t.List[str], max_workers:int
                                       ) -> t.Dict[str, t.Tuple[str, t.List[float]]]:
        generated_outputs = {}
        generated_embeddings = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exec:
            original_prompt_future = exec.submit(self._original_prompt_outputs, sentence) 
            futures_perturbation_tuple = [(
                prompt,exec.submit(self.llm.generate_text, prompt)
                ) for prompt in perturbations]
            
        original_prompt_future.result()

        for perturbation, future in futures_perturbation_tuple:
            result = future.result()
            generated_outputs[perturbation] = result

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exec:
            futures_output_tuple = [(
                perturbation,output,exec.submit(self.embedder.generate_embeddings, output)
                ) for perturbation, output in generated_outputs.items()]
            
        for perturbation, output, future in futures_output_tuple:
            embedding = future.result()
            generated_embeddings[perturbation] = [output, embedding]

        return generated_embeddings

    def compute_score(self, sentence:str, max_workers: int=10) -> t.Tuple[str,float]:
        perturbations, masked_words = self._create_perturbations(sentence)
        max_workers = min(len(perturbations)+1,max_workers)

        output_and_embeddings = self._compute_output_and_embeddings(sentence,perturbations,max_workers)
        embeddings_list = [output_and_embeddings[perturbation][1] for perturbation in perturbations]

        similarities = cosine_similarity(embeddings_list, self.original_prompt_output_embeddings)
        similarities = similarities.flatten()

        scaler = MinMaxScaler()
        normalized_similarities = scaler.fit_transform(similarities.reshape(-1,1))
        xplain_scores = 1 - normalized_similarities.flatten()
        xplain_score_tuple = tuple(zip(masked_words,xplain_scores))

        return xplain_score_tuple