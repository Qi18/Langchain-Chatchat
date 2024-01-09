# 🤗Huggingface Model Downloader

Considering the lack of multi-threaded download support in the official [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/guides/download#download-from-the-cli), and the inadequate error handling in [`hf_transfer`](https://github.com/huggingface/hf_transfer), this command-line tool smartly utilizes `wget` or `aria2` for LFS files and `git clone` for the rest.

## Features
- ⏯️ **Resume from breakpoint**: You can re-run it or Ctrl+C anytime.
- 🚀 **Multi-threaded Download**: Utilize multiple threads to speed up the download process.
- 🚫 **File Exclusion**: Use `--exclude` or `--include` to skip or specify files, save time for models with **duplicate formats** (e.g., .bin and .safetensors).
- 🔐 **Auth Support**: For gated models that require Huggingface login, use `--hf_username` and `--hf_token` to authenticate.
- 🪞 **Mirror Site Support**: Set up with `HF_ENDPOINT` environment variable.
- 🌍 **Proxy Support**: Set up with `HTTPS_PROXY` environment variable.
- 📦 **Simple**: No dependencies & No installation required.

## Usage
First, Download [`hfd.sh`](#file-hfd-sh) or clone this repo, and then grant execution permission to the script.
```bash
chmod a+x hfd.sh
```
**Usage Instructions:**
```
$ ./hfd.sh -h
Usage:
  hfd <model_id> [--exclude exclude_pattern] [--hf_username username] [--hf_token token] [--tool wget|aria2c] [-x threads] [--dataset]

Description:
  Downloads a model or dataset from Hugging Face using the provided model ID.

Parameters:
  model_id        The Hugging Face model ID in the format 'repo/model_name'.
  --exclude       (Optional) Flag to specify a string pattern to exclude files from downloading.
  exclude_pattern The pattern to match against filenames for exclusion.
  --hf_username   (Optional) Hugging Face username for authentication.
  --hf_token      (Optional) Hugging Face token for authentication.
  --tool          (Optional) Download tool to use. Can be wget (default) or aria2c.
  -x              (Optional) Number of download threads for aria2c.
  --dataset       (Optional) Flag to indicate downloading a dataset.

Example:
  hfd bigscience/bloom-560m --exclude safetensors
  hfd meta-llama/Llama-2-7b --hf_username myuser --hf_token mytoken --tool aria2c -x 8
  hfd lavita/medical-qa-shared-task-v1-toy --dataset
```
**Download a model:**
```
./hfd.sh bigscience/bloom-560m
```

**Download a model need login**

Get huggingface token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then
```bash
hfd meta-llama/Llama-2-7b --hf_username YOUR_HF_USERNAME --hf_token YOUR_HF_TOKEN
```
**Download a model and exclude certain files (e.g., .safetensors):**


```bash
./hfd.sh bigscience/bloom-560m --exclude safetensors
```

**Download with aria2c and multiple threads:**
 ```bash
 ./hfd.sh bigscience/bloom-560m --tool aria2c -x 4
 ```

*Output*:
During the download, the file URLs will be displayed:

```console
$ ./hfd.sh bigscience/bloom-560m --exclude safetensors
...
Start Downloading lfs files, bash script:

wget -c https://huggingface.co/bigscience/bloom-560m/resolve/main/flax_model.msgpack
# wget -c https://huggingface.co/bigscience/bloom-560m/resolve/main/model.safetensors
wget -c https://huggingface.co/bigscience/bloom-560m/resolve/main/onnx/decoder_model.onnx
...
```

### Create an Alias for Convenience
For easier access, you can create an alias for the script:
```bash
alias hfd="$PWD/hfd.sh"
```
