import json 
import pickle as pkl
import torch.nn.functional as F
from tqdm import tqdm, trange
import argparse
import csv
from transformers import AutoTokenizer, AutoModel
import torch, os
from torch import Tensor


# step 1: download wikipedia corpora from `https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz`
# wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
# then you will get the file named `psgs_w100.tsv`
# run the command python CUDA_VISIBLE_DEVICES=0 process_wiki.py --shard_id=1 --shards=8  --sentence_embedding_model facebook/dragon-plus-context-encoder

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def gen_embedding(args, sentences, tokenizer, model, batch_size = 512, normalize = False):
    # Tokenize sentences
    sentence_embeddings = []
    num_iter = len(sentences)//batch_size if len(sentences) % batch_size == 0 else (len(sentences)//batch_size + 1)

    for i in trange(num_iter):
        if args.sentence_embedding_model in ["intfloat/e5-large-v2"]:
            passages = ["passage: " + x for x in sentences[i*batch_size:(i+1)*batch_size]]
            encoded_input = tokenizer(passages, max_length = 200, padding=True, truncation=True, return_tensors='pt') #.to(f"cuda:{gpuid}")
            if i == 0:
                print("Example input:\t\t", passages[0])
        else:
            encoded_input = tokenizer(sentences[i*batch_size:(i+1)*batch_size], max_length = 200, padding=True, truncation=True, return_tensors='pt') #.to(f"cuda:{gpuid}")
            if i == 0:
                print("Example input:\t\t", sentences[i*batch_size])
        for x in encoded_input:
            encoded_input[x] = encoded_input[x].cuda()

        # Compute token embeddings
        if args.sentence_embedding_model in ['bert', 'simcse']:
            with torch.no_grad():
                embeddings = model(**encoded_input, output_hidden_states=True, return_dict=True).pooler_output
                sentence_embeddings.append(embeddings.detach().cpu())
        elif args.sentence_embedding_model in ['BAAI/bge-large-en-v1.5', "Alibaba-NLP/gte-base-en-v1.5"]: 
            with torch.no_grad():
                model_output = model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                sentence_embeddings.append(model_output.last_hidden_state[:, 0].detach().cpu())
        elif args.sentence_embedding_model in ["intfloat/e5-large-v2"]:
            with torch.no_grad():
                outputs = model(**encoded_input)
                embeddings = average_pool(outputs.last_hidden_state, encoded_input['attention_mask'])
                sentence_embeddings.append(embeddings.detach().cpu())
        elif args.sentence_embedding_model in ['facebook/dragon-plus-context-encoder', "OpenMatch/cocodr-base-msmarco", "OpenMatch/cocodr-large-msmarco"]:# dragon
            with torch.no_grad():
                embeddings = model(**encoded_input, output_hidden_states=True, return_dict=True).last_hidden_state[:, 0, :] # the embedding of the [CLS] token after the final layer
                sentence_embeddings.append(embeddings.detach().cpu())
        
    sentence_embeddings = torch.cat(sentence_embeddings, dim = 0)
    print("shape:", sentence_embeddings.shape)
    if args.sentence_embedding_model in ["intfloat/e5-large-v2", "Alibaba-NLP/gte-base-en-v1.5"]: # set true for E5 and GTE, no need to use this for dragon!!
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1).squeeze(1)
        print(f"{args.sentence_embedding_model}, {args.sentence_embedding_model_save_name}, {args.dataset}, With Normalization!")
    else:
        print(f"{args.sentence_embedding_model}, {args.sentence_embedding_model_save_name}, {args.dataset}, NO Normalization!")
    print("Sentence embeddings Shape:", sentence_embeddings.shape)
    return sentence_embeddings.detach().cpu().numpy()

# Sentences we want sentence embeddings for
if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--dataset", type=str, default=0)
    parser.add_argument("--total_data", type=int, default=22000000)
    parser.add_argument("--sentence_embedding_model", type=str, default="facebook/dragon-plus-context-encoder") #facebook/dragon-plus-context-encoder,  "OpenMatch/cocodr-base-msmarco"
    parser.add_argument("--sentence_embedding_model_save_name", type=str, default="dragon", choices = ["coco_base", "coco_large", "dragon", "gte-base", "e5-large"])
    args = parser.parse_args()

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(args.sentence_embedding_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.sentence_embedding_model, trust_remote_code=True).cuda()
    model.eval()
    cnt = 0
    sentences = []
    idxs = []
    print("start", args.total_data // args.shards * args.shard_id, "end", args.total_data // args.shards * (1+args.shard_id))
    if args.dataset in ["wiki"]:
        target_file = f"{args.dataset}/psgs_w100.tsv" if args.dataset in ["wiki"] else f"{args.dataset}/corpus.tsv"
        with open(target_file, "r") as f:
            reader = csv.reader(f, delimiter = '\t')
            for lines in tqdm(reader):
                if lines[0] == "id":
                    continue
                idx = int(lines[0])
                text = lines[1]
                title = lines[2]
                if idx <= (args.total_data // args.shards * (args.shard_id)):
                    continue
                elif idx > (args.total_data // args.shards * (1 + args.shard_id)):
                    break
                else:
                    sentences.append(text)
                    idxs.append(idx)
    elif args.dataset == "hpqa":
        with open(f"corpus.jsonl", "r") as f:
            idx = 0
            for lines in tqdm(f):
                data = json.loads(lines)
                text = data["text"]
                title = data["text"] + " " + data["title"]
                if idx <= (args.total_data // args.shards * (args.shard_id)):
                    idx += 1
                    continue
                elif idx > (args.total_data // args.shards * (1 + args.shard_id)):
                    break
                else:
                    sentences.append(text)
                    idxs.append(idx)
                    idx += 1
   
    embeddings = gen_embedding(args, sentences, tokenizer, model, batch_size = 256)
    os.makedirs(f"{args.dataset}/{args.sentence_embedding_model_save_name}/", exist_ok = True)
    with open(f"{args.dataset}/{args.sentence_embedding_model_save_name}/embeddings-{args.shard_id}-of-{args.shards}.pkl", "wb") as f:
        pkl.dump(embeddings, f)