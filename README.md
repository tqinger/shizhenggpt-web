# ShizhenGPT-7B-LLM：可复现的单卡 Web 部署

本目录部署的是 **`FreedomIntelligence/ShizhenGPT-7B-LLM` 文本版**，而不是四诊 Omni 版。文本版的架构与 Qwen2.5 对齐，可直接使用 Transformers；官方模型卡也明确列出 vLLM/SGLang 兼容性。页面以 Gradio 提供多轮对话、系统提示词、采样参数和流式输出。

## 本次服务器的部署配置

- 系统：Ubuntu 22.04
- GPU：NVIDIA RTX A4000，16 GiB 显存
- Python：3.11，PyTorch 2.2.2 + CUDA 12.1
- 推理精度：bitsandbytes NF4 4-bit；适合 16 GiB 显存的单用户交互
- 服务端口：`7860`
- 项目目录：`/home/featurize/work/shizhenggpt-web`
- 模型目录：`/home/featurize/data/models/ShizhenGPT-7B-LLM`

> `work/` 是 Featurize 的云端同步目录，适合代码；`data/` 是本地高速盘，适合体积较大的模型，但实例销毁后会清除。

## 一、前置条件

1. Linux + NVIDIA GPU。若使用本项目默认的 4-bit 推理，推荐至少 12 GiB 显存；本次已经在 16 GiB A4000 上部署。
2. Python 3.10+，并已安装与 GPU/CUDA 匹配的 PyTorch。在 Featurize 上使用预装的 `base` CUDA 环境验证：

   ```bash
   /environment/miniconda3/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   ```

3. 磁盘至少预留约 20 GiB 给原始 BF16 权重和 Hugging Face 缓存。

## 二、安装与下载

```bash
git clone git@github.com:tqinger/shizhenggpt-web.git shizhenggpt-web
cd shizhenggpt-web

# Featurize：不要创建或激活 .venv；使用预装的 CUDA base 环境。
export PYTHON_BIN=/environment/miniconda3/bin/python
"$PYTHON_BIN" -m pip install --user --upgrade-strategy only-if-needed -r requirements.txt

# Featurize 实例：使用账号可写的本地高速盘；不要使用 /data（通常无写权限）。
export MODEL_DIR=/home/featurize/data/models/ShizhenGPT-7B-LLM
mkdir -p "$MODEL_DIR"
huggingface-cli download FreedomIntelligence/ShizhenGPT-7B-LLM \
  --local-dir "$MODEL_DIR"
```

其他 Linux 服务器可将 `MODEL_DIR` 改为任一当前用户可写、且有至少 20 GiB 空间的目录，例如 `$HOME/models/ShizhenGPT-7B-LLM`。

如果 Hugging Face 出现临时 `429 Too Many Requests`，下载并未成功；先稍后重试。登录自己的 Hugging Face 账号可使用个人访问令牌并通常获得更稳定的下载额度：

```bash
huggingface-cli login
huggingface-cli download FreedomIntelligence/ShizhenGPT-7B-LLM \
  --local-dir "$MODEL_DIR"
```

如网络环境允许使用镜像，可临时设置镜像端点后重试：

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  FreedomIntelligence/ShizhenGPT-7B-LLM --local-dir "$MODEL_DIR"
```

如目标服务器尚未安装 CUDA 版 PyTorch，请先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择与 CUDA 驱动匹配的命令，再安装本项目 `requirements.txt`。

下载完成后确认模型完整：

```bash
test -f "$MODEL_DIR/config.json" && echo "模型文件已就绪"
```

## 三、启动页面

前台启动（便于首次排错）：

```bash
PYTHON_BIN=/environment/miniconda3/bin/python \
  MODEL_PATH="$MODEL_DIR" PORT=7860 ./start.sh
```

后台启动前，先检查服务是否已经在运行。`HTTP/1.1 200 OK` 表示网页已经可用，**不要再启动第二个进程**：

```bash
curl -I http://127.0.0.1:7860
```

仅在服务未运行时后台启动：

```bash
mkdir -p logs
PYTHON_BIN=/environment/miniconda3/bin/python \
  MODEL_PATH="$MODEL_DIR" PORT=7860 \
  nohup ./start.sh > logs/app.log 2>&1 < /dev/null &
echo $! > app.pid
tail -f logs/app.log
```

`nohup: ignoring input` 是正常提示。`tail -f` 会持续显示日志；按 `Ctrl + C` 只会退出日志查看，**不会停止网页服务**。如果出现 `Cannot find empty port ... 7860`，说明已有服务正在使用该端口；运行上述 `curl` 检查并直接访问页面，或按下面的“停止服务”步骤先停止旧服务。



服务监听 `0.0.0.0:7860`。先在服务器上验证：

```bash
curl -I http://127.0.0.1:7860
```

在 Featurize 平台上，将本地服务映射到公网：

```bash
featurize port export 7860
```

命令会返回实际的公网端口；请访问：点击网页连接就可以使用前端网页测试模型功能

```text
http://workspace.featurize.cn:<命令返回的公网端口>
```

可用 `featurize port list` 查询已经创建的映射。其他云平台则需要在安全组、防火墙或端口映射中显式放行对应端口。

停止服务：

```bash
cd /home/featurize/work/shizhenggpt-web
kill "$(cat app.pid)"

# 验证：连接失败即代表服务已停止
curl -I http://127.0.0.1:7860
```

若最后一条命令仍返回 `200 OK`，说明该端口上还有服务在运行；执行 `ss -ltnp '( sport = :7860 )'` 查找进程，而不要重复执行启动命令。
