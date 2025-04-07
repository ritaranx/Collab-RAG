# Collab-RAG

Here is the code repo of our paper Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration.

## Dataset and Indexing
The raw dataset we use is stored in `datasets.zip`, following the instruction in this link: [link](https://github.com/sunnynexus/Search-o1). 
The corpus can be found at [this link](https://huggingface.co/datasets/BeIR/hotpotqa/blob/main/corpus.jsonl.gz). Please use the following command to generate the embeddings for each dataset (suppose the corpus file is named `corpus.jsonl`).
```
CUDA_VISIBLE_DEVICES=0 process_wiki.py \
--shard_id={0,1,..,7} \
--shards=8  \
--sentence_embedding_model facebook/dragon-plus-context-encoder \
--total_data 6000000 \
--sentence_embedding_model_save_name dragon \
--dataset hpqa
```

## White Box SLM Decomposition
Please use the command for using SLM for break down questions:
```
CUDA_VISIBLE_DEVICES=0,1 python slm_decompose.py \
--model_path meta-llama/Llama-3.1-8B-Instruct (you can change to your trained model later) \
--tokenizer meta-llama/Llama-3.1-8B-Instruct \
--temperature 1.0 \
--tensor_parallel_size 2 \
--expname [Your Experiment Name]
```
After this step, the decomposed question will be saved at 
`f"test/{dataset}/prompts_decompose_test_t{args.temperature}_{model_name}/generate.jsonl`

## Black Box LLM Reader
Please use the command for using SLM for break down questions:
```
CUDA_VISIBLE_DEVICES=0 python llm_reader.py \
--llm_model gpt-4o-mini \

```

## Using Llama Factory for Finetuning

## Citation
If you find this repository helpful, please kindly consider citing the corresponding paper. Thanks!
```
@article{xu2025collab,
  title={Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration},
  author={Xu, Ran and Shi, Wenqi and Zhuang, Yuchen and Yu, Yue and Ho, Joyce C and Wang, Haoyu and Yang, Carl},
  journal={arXiv preprint},
  year={2025}
}
```
