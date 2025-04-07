import openai
from typing import List, Dict
import time
from prompts import prompt
import numpy as np
import csv, json
from tqdm import tqdm, trange
import pickle as pkl
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import faiss
import copy
import os
import argparse
from torch import Tensor


def decompose_question(question: str, prompt: str, llm_model: str = "gpt-4o") -> List[Dict]:
    call_prompt = prompt.replace("${question}", question)
    response = call_llm(call_prompt, model=llm_model)

    decomposed_questions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("### Q"):
            question_part, context_part = line.split("## Need Context? ##")
            question_text = question_part.split(":", 1)[1].strip()
            needs_context = context_part.strip().lower().startswith("yes")
            q_label = question_part.split(":")[0].strip("#").strip()
            decomposed_questions.append({"label": q_label, "text": question_text, "needs_context": needs_context})

    return decomposed_questions


def call_llm(prompt, model) -> str:
    if model == "gpt-4o":
        openai.api_type = "azure"
        openai.api_base = "https://your-azure-instance.openai.azure.com/"
        openai.api_version = "2024-05-01-preview"
        openai.api_key = "<your-api-key>"
        engine = "gpt-4o"
    elif model == "gpt-4o-mini":
        openai.api_type = "azure"
        openai.api_base = "https://your-azure-instance.openai.azure.com/"
        openai.api_version = "2024-02-15-preview"
        openai.api_key = "<your-api-key>"
        engine = "gpt-4o-mini"

    try_times = 0
    success = False
    while not success and try_times < 4:
        try:
            response = openai.ChatCompletion.create(
                engine=engine,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )
            output = response.choices[0].message["content"]
            success = True
        except Exception:
            time.sleep(5)
            try_times += 1
    return output if success else "Failed"


def load_embedding(dataset: str, embedding_model: str):
    sentences, passage_embeddings = [], []

    with open(f"{dataset}/corpus.tsv", "r") as f:
        reader = csv.reader(f, delimiter='\t')
        for lines in tqdm(reader):
            if lines[0] == "id":
                continue
            sentences.append(lines[1])

    for i in trange(4):
        path = f"{dataset}/{embedding_model}/embeddings-{i}-of-4.pkl"
        with open(path, "rb") as f:
            passage_embeddings.append(pkl.load(f))

    passage_embeddings = np.concatenate(passage_embeddings, axis=0)
    print("Passage Size:", len(sentences), passage_embeddings.shape)
    return sentences, passage_embeddings


def retrieve_context(query, cpu_index, corpus, embedding_tokenizer, embedding_model, embedding_model_name, top_k=3) -> List[str]:
    query_embedding = embed_text(query, embedding_tokenizer, embedding_model, embedding_model_name)
    _, dev_I = cpu_index.search(query_embedding, top_k)
    return [corpus[r] for r in dev_I[0]]


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed_text(text, tokenizer, model, embedding_model_name) -> List[float]:
    if "e5-large" in embedding_model_name:
        encoded_input = tokenizer("query " + text, max_length=100, padding=True, truncation=True, return_tensors='pt')
    else:
        encoded_input = tokenizer(text, max_length=100, padding=True, truncation=True, return_tensors='pt')

    for x in encoded_input:
        encoded_input[x] = encoded_input[x].cuda()

    with torch.no_grad():
        if embedding_model_name == "gte-base":
            sentence_embeddings = model(**encoded_input).last_hidden_state[:, 0].detach().cpu()
        elif embedding_model_name == "e5-large":
            outputs = model(**encoded_input)
            embeddings = average_pool(outputs.last_hidden_state, encoded_input['attention_mask'])
            sentence_embeddings = embeddings.detach().cpu()
        else:
            embeddings = model(**encoded_input, output_hidden_states=True, return_dict=True).last_hidden_state[:, 0, :]
            sentence_embeddings = embeddings.detach().cpu()

    return F.normalize(sentence_embeddings, p=2, dim=1).squeeze(1)


def zigzag_visit(lst):
    n = len(lst)
    result = [None] * n
    i, j = 0, 0
    while j < (n + 1) // 2:
        result[j] = lst[i]
        i += 2
        j += 1
    i = 1
    j = n - 1
    while j >= (n + 1) // 2:
        result[j] = lst[i]
        i += 2
        j -= 1
    return result


def answer_sub_question(sub_q: str, context_passages: List[str], model: str) -> str:
    reorder_passages = zigzag_visit(context_passages)
    context_text = "\n\n".join(reorder_passages)
    prompt = f"""You have the following context passages:\n{context_text}

Question: {sub_q}

Please Answer the above question with one or a list of entities with the given context as the reference. 
If you do not find an answer in the context, please use your own knowledge to answer it.
Do not give any explanation. Your answer needs to be as short as possible."""
    return call_llm(prompt, model=model).strip()


def generate_final_answer(original_question: str, sub_questions: Dict[str, str], sub_answers: Dict[str, str], model: str, dataset: str) -> str:
    sub_answer_text = "\n".join([f"{k}: {sub_questions[k]}, Answer for {k}: {v}" for k, v in sub_answers.items()])
    if dataset == "strategyqa":
        final_prompt = "Your answer needs to be True or False only."
    else:
        final_prompt = "Your answer needs to be as short as possible and should be with one or a list of entities."
    prompt = f"""For the question: {original_question}\n\nWe have the following decomposed subquestions and sub-answers:\n{sub_answer_text}\n
Based on these sub-answers, provide the final concise answer to the original question: \"{original_question}\".\nDo not give an explanation. {final_prompt}"""
    return call_llm(prompt, model=model).strip()


def replace_placeholders(question_text: str, answers_so_far: Dict[str, str]) -> str:
    import re
    matches = re.findall(r"#(\d+)", question_text)
    for m in matches:
        place_holder = f"#{m}"
        q_key = f"Q{m}"
        if q_key in answers_so_far:
            question_text = question_text.replace(place_holder, answers_so_far[q_key])
    return question_text
