import os
import re
import time
import json
import csv
import copy
import pickle as pkl
import argparse
from typing import List, Dict

import torch
import numpy as np
import torch.nn.functional as F
import faiss
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel
import openai

from prompts import prompt

# ------------------------
# Configurable constants
# ------------------------
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "./hf_cache")
DATA_DIR = os.getenv("DATA_DIR", "./data")
AZURE_API_BASE_GPT4O = os.getenv("AZURE_API_BASE_GPT4O", "https://your-azure-gpt4o-instance.openai.azure.com/")
AZURE_API_BASE_GPT4O_MINI = os.getenv("AZURE_API_BASE_GPT4O_MINI", "https://your-azure-gpt4o-mini-instance.openai.azure.com/")
AZURE_API_KEY_GPT4O = os.getenv("AZURE_API_KEY_GPT4O", "your-gpt4o-key")
AZURE_API_KEY_GPT4O_MINI = os.getenv("AZURE_API_KEY_GPT4O_MINI", "your-gpt4o-mini-key")
ENGINE_NAME = "your-engine-name"
# ------------------------
# Utility functions
# ------------------------
def prRed(s): print("\033[91m {}\033[00m".format(s))
def prPurple(s): print("\033[95m {}\033[00m".format(s))
def prYellow(s): print("\033[93m {}\033[00m".format(s))
def prLightPurple(s): print("\033[94m {}\033[00m".format(s))

# ------------------------
# LLM API Wrapper
# ------------------------
def call_llm(prompt, model) -> str:
    if model == "gpt-4o":
        openai.api_type = "azure"
        openai.api_base = AZURE_API_BASE_GPT4O
        openai.api_version = "2024-05-01-preview"
        openai.api_key = AZURE_API_KEY_GPT4O
        engine = ENGINE_NAME
    elif model == "gpt-4o-mini":
        openai.api_type = "azure"
        openai.api_base = AZURE_API_BASE_GPT4O_MINI
        openai.api_version = "2024-02-15-preview"
        openai.api_key = AZURE_API_KEY_GPT4O_MINI
        engine = ENGINE_NAME

    try_times, success = 0, False
    while not success and try_times < 3:
        try:
            response = openai.ChatCompletion.create(
                engine=engine,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )
            return response.choices[0].message["content"].strip()
        except (openai.error.Timeout,
                openai.error.RateLimitError,
                openai.error.APIConnectionError,
                openai.error.APIError,
                openai.error.InvalidRequestError,
                Exception):
            print("Retrying due to API error...")
            time.sleep(5)
            try_times += 1
    return "Failed"

# ------------------------
# Question Decomposition
# ------------------------
def decompose_question(question: str, prompt: str, llm_model: str = "gpt-4o") -> List[Dict]:
    call_prompt = prompt.replace("${question}", question)
    response = call_llm(call_prompt, model=llm_model)
    decomposed_questions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("### Q"):
            q_part, ctx_part = line.split("## Need Context? ##")
            q_text = q_part.split(":", 1)[1].strip()
            needs_context = ctx_part.strip().lower().startswith("yes")
            q_label = q_part.split(":")[0].strip("#").strip()
            decomposed_questions.append({
                "label": q_label,
                "text": q_text,
                "needs_context": needs_context
            })
    return decomposed_questions

# ------------------------
# Embedding + Retrieval
# ------------------------
def load_embedding(dataset: str):
    sentences, passage_embeddings = [], []
    with open(os.path.join(DATA_DIR, dataset, "corpus.tsv"), "r") as f:
        reader = csv.reader(f, delimiter='\t')
        for lines in tqdm(reader):
            if lines[0] != "id":
                sentences.append(lines[1])
    for i in trange(4):
        with open(os.path.join(DATA_DIR, dataset, f"embeddings-{i}-of-4.pkl"), "rb") as f:
            passage_embeddings.append(pkl.load(f))
    return sentences, np.concatenate(passage_embeddings, axis=0)

def embed_text(text, tokenizer, model, normalize=True) -> List[float]:
    encoded = tokenizer(text, max_length=100, padding=True, truncation=True, return_tensors='pt')
    encoded = {k: v.cuda() for k, v in encoded.items()}
    with torch.no_grad():
        output = model(**encoded).last_hidden_state[:, 0].detach().cpu()
    return F.normalize(output, p=2, dim=1).squeeze(1) if normalize else output

def retrieve_context(query, cpu_index, corpus, tokenizer, model, top_k=3):
    q_embedding = embed_text(query, tokenizer, model, False)
    _, indices = cpu_index.search(q_embedding, top_k)
    return [corpus[i] for i in indices[0]]

# ------------------------
# Sub-question Answering
# ------------------------
def zigzag_visit(lst):
    return [lst[i] for i in range(0, len(lst), 2)] + [lst[i] for i in range(len(lst)-1 if len(lst)%2==0 else len(lst)-2, 0, -2)]

def answer_sub_question(sub_q, passages, model):
    context = "\n\n".join(zigzag_visit(passages))
    prompt = f"""You have the following context passages:
{context}

Question: {sub_q}

Answer to the original question should be with one or a list of entities with the given context as the reference. 
If you do not find an answer in the context, please use your own knowledge to answer it.
Do not give any explanation. Your answer needs to be as short as possible with one or a list of entities."""
    return call_llm(prompt, model=model)

# ------------------------
# Final Answer Generation
# ------------------------
def generate_final_answer(original_q, sub_qs, sub_as, model):
    text = "\n".join([f"{k}: {sub_qs[k]}, Answer for {k}: {v}" for k, v in sub_as.items()])
    prompt = f"""Original question: {original_q}

We have the following decomposed subquestions and sub-answers:
{text}

Based on these sub-answers, provide the final concise answer to the original question: \"{original_q}\".
Do not give an explanation. Answer needs to be as short as possible and should be with one or a list of entities."""
    return call_llm(prompt, model=model)

def multi_turn_qa(question, sub_questions, gold_answer, tokenizer, embedding_model, llm_model):
    subq_dict = {q["label"]: q["text"] for q in sub_questions}
    answer_dict = {}
    passage_dict = {}

    for subq in sub_questions:
        label = subq["label"]
        text = subq["text"]
        needs_context = subq.get("needs_context", True)

        # Replace #n placeholders with prior answers
        resolved_text = replace_placeholders(text, answer_dict)

        passages = []
        if needs_context:
            passages = retrieve_context(resolved_text, index, corpus, tokenizer, embedding_model, top_k=9)

        answer = answer_sub_question(resolved_text, passages, llm_model)
        answer_dict[label] = answer
        passage_dict[label] = passages

    final_answer = generate_final_answer(question, subq_dict, answer_dict, llm_model)
    print(f"-------\nquestion: {question}\nsubq: {sub_questions}\nanswer: {answer_dict}\nPred: {final_answer} gold: {gold_answer}")
    return final_answer, answer_dict, passage_dict

def replace_placeholders(text: str, answers_so_far: Dict[str, str]) -> str:
    matches = re.findall(r"#(\\d+)", text)
    for m in matches:
        placeholder = f"#{m}"
        q_key = f"Q{m}"
        if q_key in answers_so_far:
            text = text.replace(placeholder, answers_so_far[q_key])
    return text

# ------------------------
# Main Function
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="test")
    args = parser.parse_args()

    sentence_embedding_model = "facebook/dragon-plus-query-encoder"
    tokenizer = AutoTokenizer.from_pretrained(sentence_embedding_model, cache_dir=HF_CACHE_DIR, trust_remote_code=True)
    embedding_model = AutoModel.from_pretrained(sentence_embedding_model, cache_dir=HF_CACHE_DIR, trust_remote_code=True).cuda()
    embedding_model.eval()

    questions_path = os.path.join(args.save_dir, args.dataset, f"prompts_decompose_test_t{args.temperature}_{args.expname}/generate.jsonl")
    with open(questions_path, "r") as f:
        questions = [json.loads(line) for line in f]

    corpus, embeddings = load_embedding(args.dataset)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.omp_set_num_threads(32)
    index.add(embeddings)

    saved = []
    for idx, item in enumerate(tqdm(questions)):
        try:
            final, inter_ans, inter_pass = multi_turn_qa(item["question"], item["decomposed"], item["answer"], tokenizer, embedding_model, args.llm_model)
            item.update({"index": idx, "final_answer": final, "intermediate_answers": inter_ans, "intermediate_passages": inter_pass})
            saved.append(item)
        except Exception as e:
            print(f"Error for item {idx}: {e}")

    if saved:
        out_path = os.path.join(args.save_dir, "output", args.dataset, f"prompts_{args.llm_model}-{args.expname}.jsonl")
        with open(out_path, "w") as f:
            for ex in saved:
                f.write(json.dumps(ex) + '\n')
