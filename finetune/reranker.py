import subprocess
from configs import MODEL_PATH, EMBEDDING_MODEL


class EmbeddingFinetune:

    def __init__(self, ):
        conda_env_name = 'chatglm'
        self.activate_command = f'conda activate {conda_env_name} && '

    # def finetune_reranker(self):
    #     output_dir = '/'
    #
    #     # 定义命令
    #     command = [
    #         'torchrun',
    #         '--nproc_per_node', str(number_of_gpus),
    #         '-m', 'FlagEmbedding.reranker.run',
    #         '--output_dir', output_dir,
    #         '--model_name_or_path', 'BAAI/bge-reranker-base',
    #         '--train_data', './toy_finetune_data.jsonl',
    #         '--learning_rate', '6e-5',
    #         '--fp16',
    #         '--num_train_epochs', '5',
    #         '--per_device_train_batch_size', str(batch_size),
    #         '--gradient_accumulation_steps', '4',
    #         '--dataloader_drop_last', 'True',
    #         '--train_group_size', '16',
    #         '--max_len', '512',
    #         '--weight_decay', '0.01',
    #         '--logging_steps', '10'
    #     ]
    #
    #     subprocess.run(self.activate_command + ' '.join(command), shell=True)

    def mine_hard_neg(self):
        command = [
            "python",
            " -m", "FlagEmbedding.baai_general_embedding.finetune.hn_mine",
            "--model_name_or_path", MODEL_PATH["embed_model"][EMBEDDING_MODEL],
            "--input_file", "./data/toy_finetune_data.jsonl",
            "--output_file", "./data/toy_finetune_data_minedHN.jsonl",
            "--range_for_sampling", "20-200",
            "--use_gpu_for_searching"]
        result = subprocess.run(self.activate_command + ' '.join(command), shell=True)

        # 输出命令执行结果
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # 检查命令是否成功执行
        if result.returncode == 0:
            print("Command executed successfully.")
        else:
            print("Command failed with return code", result.returncode)

if __name__ == "__main__":
    embedding_finetune = EmbeddingFinetune()
    embedding_finetune.mine_hard_neg()