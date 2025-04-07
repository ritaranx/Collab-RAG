from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import os
from tqdm import trange
import copy
import json

parser = argparse.ArgumentParser("")
parser.add_argument("--llm_model", type=str, default="gpt-4o")
parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--model_path", type=str, default="models/llama-3.1-8b-instruct")
parser.add_argument("--expname", type=str, default="")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_p", type=float, default=0.99)
parser.add_argument("--tensor_parallel_size", type=int, default=1)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

sampling_params = SamplingParams(
    temperature=args.temperature,
    top_p=args.top_p,
    repetition_penalty=1.05,
    max_tokens=2048
)

llm = LLM(
    model=args.model_path,
    tensor_parallel_size=args.tensor_parallel_size,
    gpu_memory_utilization=0.88,
    trust_remote_code=True
)

datasets = ["bamboogle", "2wikimultihopqa", "hotpotqa", "musique"]
model_name = args.expname

prompt = """Please break down the given question into multiple specific sub-questions that address individual components of the original question.
Please generate the decomposed sub-questions for the below question. The sub-question should be labeled with a reference to previous answers (e.g., #1) when needed. For example, #1 means the answer for decomposed question 1. 
The token after the question `## Need Context` stands for whether the decomposed question needs external corpus to answer. 'Yes' means it needs external corpus to answer, 'No' means it can be directly answered without retrieval.
Here are four examples:

[[Begin of the Example 1]]
## Question: 
What is the average winter daytime temperature in the region containing Richmond, in the state where WXBX is located?

## Decomposed Question:
### Q1: Which state is WXBX located? ## Need Context? ## Yes
### Q2: In which of #1 's regions is Richmond? ## Need Context? ## Yes
### Q3: What is the average winter daytime temperature in #2? ## Need Context? ## Yes
[[End of the Example 1]]

[[Begin of the Example 2]]
## Question: 
How long was the place where the Yongle Emperor greeted the person to whom the edict was addressed the capitol of the area where Guangling District was located?

## Decomposed Question:
### Q1: Who was the edict addressed to? ## Need Context? ## Yes
### Q2: Where did the Yongle Emperor greet #1 ?  ## Need Context? ## Yes
### Q3: Where does Guangling District locate? ## Need Context? ## Yes
### Q4: How long had #2 been the capital city of #3 ?  ## Need Context? ## Yes
[[End of the Example 2]]

Now, decompose the following question:
## Question: 
${question}

## Decomposed Question:"""

for dataset in datasets:
    prompts = []
    contexts = []

    with open(f"./processed_data/{dataset}/test.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            question = example["question_text"]
            answer_spans = [span for obj in example["answers_objects"] for span in obj["spans"]]
            item = {"question": question, "answer": answer_spans}
            prompt_text = prompt.replace("${question}", question)
            prompts.append([{"role": "user", "content": prompt_text.strip()}])
            contexts.append(item)

    print(dataset, len(prompts), len(contexts))
    examples = []
    output_dir = f"test/{dataset}/prompts_decompose_test_t{args.temperature}_{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    for i in trange(len(prompts)):
        text = tokenizer.apply_chat_template(prompts[i], tokenize=False, add_generation_prompt=True)
        print(text)

        N_samples = 1 if args.temperature == 0 else 3
        for j in range(N_samples):
            ctx = copy.deepcopy(contexts[i])
            outputs = llm.generate([text], sampling_params)
            generated_text = outputs[0].outputs[0].text
            if j == 0:
                print(len(outputs))
                print('======\n', generated_text, '\n======')

            decomposed_questions = []
            for line in generated_text.strip().split("\n"):
                line = line.strip()
                if line.startswith("### Q"):
                    try:
                        question_part, context_part = line.split("## Need Context? ##")
                        question_text = question_part.split(":", 1)[1].strip()
                        needs_context = context_part.strip().lower().startswith("yes")
                        q_label = "Q" + question_part.split(":")[0].split("Q")[-1].strip()
                        decomposed_questions.append({
                            "label": q_label,
                            "text": question_text,
                            "needs_context": needs_context
                        })
                    except:
                        decomposed_questions = "Error"
                        print(f"Error in decomposing \n\n {generated_text} \n\n")
                        break

            ctx["question_id"] = i
            ctx["decompose_id"] = j
            ctx["decomposed"] = decomposed_questions
            examples.append(ctx)

    with open(f"{output_dir}/generate.jsonl", "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

