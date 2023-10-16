import time


def downloadDir(repo_id, local_dir):
    from huggingface_hub import snapshot_download
    cache_dir = local_dir + "/cache"
    while True:
        try:
            snapshot_download(cache_dir=cache_dir,
                              local_dir=local_dir,
                              repo_id=repo_id,
                              local_dir_use_symlinks=False,
                              resume_download=True,
                              allow_patterns=["*.model", "*.json", "*.bin",
                                              "*.py", "*.md", "*.txt"],
                              ignore_patterns=["*.safetensors", "*.msgpack",
                                               "*.h5", "*.ot", ],
                              )
        except Exception as e:
            print(e)
            # time.sleep(5)
        else:
            print('下载完成')
            break


def downloadFile(repo_id, local_dir, filename):
    from huggingface_hub import hf_hub_download
    cache_dir = local_dir + "/cache"
    while True:
        try:
            hf_hub_download(cache_dir=cache_dir,
                            local_dir=local_dir,
                            repo_id=repo_id,
                            filename=filename,
                            local_dir_use_symlinks=False,
                            resume_download=True,
                            etag_timeout=100
                            )
        except Exception as e:
            print(e)
            # time.sleep(5)
        else:
            print('下载完成')
            break

if __name__=="__main__":
    repo_id = "gpt2"
    local_dir = './gpt2'
    downloadDir(repo_id, local_dir)
    # repo_id = "BlinkDL/rwkv-4-pile-7b"  # 仓库ID
    # local_dir = 'd:/ai/models2'
    # filename = "RWKV-4-Pile-7B-Chn-testNovel-done-ctx2048-20230404.pth"
    # downloadFile(repo_id, local_dir, filename)