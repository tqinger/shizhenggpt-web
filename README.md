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
2. Python 3.10+，并已安装与 GPU/CUDA 匹配的 PyTorch。例如先验证：

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

3. 磁盘至少预留约 20 GiB 给原始 BF16 权重和 Hugging Face 缓存。

## 二、安装与下载

```bash
git clone git@github.com:tqinger/shizhenggpt-web.git shizhenggpt-web
cd shizhenggpt-web

# 推荐使用虚拟环境；Featurize 也可按平台建议直接使用 pip install --user。
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade-strategy only-if-needed -r requirements.txt

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
MODEL_PATH="$MODEL_DIR" PORT=7860 ./start.sh
```

后台启动前，先检查服务是否已经在运行。`HTTP/1.1 200 OK` 表示网页已经可用，**不要再启动第二个进程**：

```bash
curl -I http://127.0.0.1:7860
```

仅在服务未运行时后台启动：

```bash
mkdir -p logs
MODEL_PATH="$MODEL_DIR" PORT=7860 \
  nohup ./start.sh > logs/app.log 2>&1 < /dev/null &
echo $! > app.pid
tail -f logs/app.log
```

`nohup: ignoring input` 是正常提示。如果出现 `Cannot find empty port ... 7860`，说明已有服务正在使用该端口；运行上述 `curl` 检查并直接访问页面，或按下面的“停止服务”步骤先停止旧服务。

如需向外网映射端口，请务必在启动时设置认证信息（不要把密码写进代码或提交到仓库）：

```bash
APP_USERNAME=your-name APP_PASSWORD='use-a-strong-secret' \
  MODEL_PATH="$MODEL_DIR" PORT=7860 ./start.sh
```

服务监听 `0.0.0.0:7860`。先在服务器上验证：

```bash
curl -I http://127.0.0.1:7860
```

在 Featurize 平台上，将本地服务映射到公网：

```bash
featurize port export 7860
```

命令会返回实际的公网端口；请访问：

```text
http://workspace.featurize.cn:<命令返回的公网端口>
```

可用 `featurize port list` 查询已经创建的映射。其他云平台则需要在安全组、防火墙或端口映射中显式放行对应端口。

停止服务：

```bash
kill "$(cat app.pid)"
```

## 四、在 Featurize 平台启动并访问网页

以下命令在 **Featurize 实例的 Web 终端** 或 SSH 终端中执行；不要在本机 Windows 终端中执行。

### 1. 进入项目目录并启动服务

首次部署先按第二节安装依赖、下载模型。后续启动只需执行：

```bash
cd /home/featurize/work/shizhenggpt-web
mkdir -p logs
nohup ./start.sh > logs/app.log 2>&1 < /dev/null &
echo $! > app.pid
```

确认服务已正常运行并监听本机的 `7860` 端口：

```bash
ps -p "$(cat app.pid)" -o pid=,stat=,cmd=
curl -I http://127.0.0.1:7860
```

预期 `curl` 显示 `HTTP/1.1 200 OK`。若失败，实时查看日志：

```bash
tail -f logs/app.log
```

> 本项目的 `start.sh` 已设置 `SERVER_NAME=0.0.0.0`，符合 Featurize 对公网端口映射的监听要求。

### 2. 将服务端口映射为可访问的网址

在同一台 Featurize 实例中执行：

```bash
featurize port export 7860
```

平台会返回一个动态公网端口，例如：

```text
Local port 7860 has been exported to 47657
You can visit http://workspace.featurize.cn:47657
```

复制命令返回的完整 `http://workspace.featurize.cn:<公网端口>` 到浏览器即可打开网页。这里的公网端口由平台分配，**不要直接访问 `workspace.featurize.cn:7860`**。

查询映射、或取消不用的映射：

```bash
featurize port list
featurize port unexport 7860
```

端口映射不创建第二个 Web 服务；它只是将浏览器访问的公网端口转发到本机的 `7860`。因此先启动应用、再映射端口，是排查问题最简单的顺序。Featurize 单实例最多可映射 10 个端口。

### 3. 停止或重启网页服务

```bash
cd /home/featurize/work/shizhenggpt-web
kill "$(cat app.pid)"             # 停止

# 重新启动
nohup ./start.sh > logs/app.log 2>&1 < /dev/null &
echo $! > app.pid
```

如果部署了更新后的代码，先停止旧进程，再执行上述重启命令。端口映射仍然指向 `7860`，通常无需重新执行 `featurize port export 7860`；用 `featurize port list` 确认即可。

### 4. 公开访问前的账号保护（推荐）

Featurize 映射后的端口可被任何知道网址的人访问。需要保护页面时，先停止旧服务，再在终端中交互式设置账号和密码后启动，避免把密码写入代码或命令历史：

```bash
cd /home/featurize/work/shizhenggpt-web
kill "$(cat app.pid)"

read -rp "页面账号: " APP_USERNAME
read -rsp "页面密码: " APP_PASSWORD; echo
export APP_USERNAME APP_PASSWORD
nohup ./start.sh > logs/app.log 2>&1 < /dev/null &
echo $! > app.pid
unset APP_PASSWORD
```

刷新网页后将出现浏览器的基本认证登录框。不要复用 SSH 密码。

## 五、运行方式与显存说明

本项目默认使用 NF4 4-bit 量化和单请求队列，因此适合演示、评估和低并发内部使用。首次消息才会加载模型，通常需要几十秒；后续消息无需重复加载。回复以流式方式显示；没有人为设定“最大生成长度”，只会在模型本身的上下文窗口边界处停止，以避免超出模型可处理的总 token 数。

- **显存不足**：较长的上下文和较长的回复都会增加显存占用。请清空对话后重试，确保没有其他 GPU 进程，并确认 `bitsandbytes` 已成功安装。
- **模型下载到错误位置**：通过 `MODEL_PATH=/实际目录 python app.py` 显式指定。
- **外网打不开**：先确认 `curl 127.0.0.1:7860` 成功，再检查安全组/平台端口映射；不要为了临时测试使用 Gradio 的公共 `share=True` 链接。
- **多人/生产负载**：模型卡支持 vLLM 或 SGLang。建议把模型服务化为受鉴权的内部 API，再将这个 Gradio 页面改为 API 客户端；不要直接把单进程演示服务暴露给公网。

## 六、安全与医疗使用边界

页面内置了“研究与学习”提示词和页面警示，但这不能替代产品安全治理。公开访问前至少应添加账号认证、HTTPS、访问日志脱敏、限流和输出审核。模型输出不能作为诊断、处方或紧急医疗建议；涉及真实患者的使用应经过医疗专业人员审核，并完成适用的合规、隐私和临床验证工作。

## 参考

- [ShizhenGPT-7B-LLM 官方模型卡](https://huggingface.co/FreedomIntelligence/ShizhenGPT-7B-LLM)
- [ShizhenGPT 论文](https://arxiv.org/abs/2508.14706)
- [Featurize：将服务（端口）暴露到公网](https://docs.featurize.cn/docs/manual/port-exporting)
