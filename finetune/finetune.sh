
#"python,
#            " -m", "FlagEmbedding.baai_general_embedding.finetune.hn_mine",
#            "--model_name_or_path", MODEL_PATH["embed_model"][EMBEDDING_MODEL],
#            "--input_file", "./data/toy_finetune_data.jsonl",
#            "--output_file", "./data/toy_finetune_data_minedHN.jsonl",sa
#            "--range_for_sampling", "20-200",
#            "--use_gpu_for_searching"]

python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path /data/lrq/llm/llm_data/text/bge-large-zh-v1.5 \
--input_file ./data/toy_finetune_data.jsonl \
--output_file ./data/toy_finetune_data_minedHN.jsonl \
--range_for_sampling 20-220 \
--use_gpu_for_searching