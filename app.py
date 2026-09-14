"""A lightweight Gradio UI for ShizhenGPT-7B-LLM.

The application defaults to 4-bit inference so it can run on a 16 GiB GPU.
It intentionally uses the text-only model; the Omni model has extra audio and
vision dependencies that are unnecessary for a basic chat service.
"""

import logging
import os
from threading import Thread
from functools import lru_cache

# Disable Gradio telemetry by default. A deployment can still override it with
# GRADIO_ANALYTICS_ENABLED=True if desired.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TextIteratorStreamer


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("shizhenggpt-web")

MODEL_ID = os.getenv(
    "MODEL_ID", "FreedomIntelligence/ShizhenGPT-7B-LLM"
)
MODEL_PATH = os.getenv("MODEL_PATH", MODEL_ID)
APP_USERNAME = os.getenv("APP_USERNAME")
APP_PASSWORD = os.getenv("APP_PASSWORD")
DEFAULT_SYSTEM_PROMPT = (
    "你是一个中医药知识学习助手。请用清晰、审慎的中文回答，说明信息仅供学习和"
    "研究参考；不要把内容表述为个体化诊断或处方。遇到紧急或严重症状时，建议用户"
    "尽快向合格的医疗专业人士求助。"
)


@lru_cache(maxsize=1)
def load_model():
    """Load the tokenizer and model once, on the first request."""
    LOGGER.info("Loading model from %s", MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def _context_window(tokenizer, model):
    """Return the model's real context limit, ignoring tokenizer sentinel values."""
    candidates = []
    for value in (
        getattr(tokenizer, "model_max_length", None),
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "n_positions", None),
    ):
        if isinstance(value, int) and 0 < value < 1_000_000:
            candidates.append(value)
    if not candidates:
        raise RuntimeError("无法识别模型上下文长度，不能安全地开始生成。")
    return min(candidates)


def generate_answer(history, system_prompt, temperature, top_p):
    """Stream an assistant response; generation is limited only by model context."""
    tokenizer, model = load_model()

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(
        {
            "role": item["role"],
            "content": item["content"],
        }
        for item in history
        if item.get("role") in {"user", "assistant"} and item.get("content")
    )

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    available_new_tokens = _context_window(tokenizer, model) - model_inputs.input_ids.shape[1]
    if available_new_tokens <= 0:
        raise ValueError("当前对话已达到模型上下文上限，请清空对话后重试。")

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300.0
    )

    generation_kwargs = {
        **model_inputs,
        # No arbitrary UI cap: this only prevents exceeding the model context window.
        "max_new_tokens": available_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    if float(temperature) > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    def run_generation():
        with torch.inference_mode():
            model.generate(**generation_kwargs)

    worker = Thread(target=run_generation, daemon=True)
    worker.start()
    yield from streamer


def add_user_message(message, history):
    history = history or []
    message = (message or "").strip()
    if not message:
        return "", history, history
    updated_history = history + [{"role": "user", "content": message}]
    return "", updated_history, updated_history


def add_assistant_message(history, system_prompt, temperature, top_p):
    if not history:
        yield history, history
        return
    try:
        answer = ""
        for text in generate_answer(history, system_prompt, temperature, top_p):
            answer += text
            updated_history = history + [{"role": "assistant", "content": answer}]
            yield updated_history, updated_history
    except Exception as exc:  # Keep a UI-visible error while preserving server logs.
        LOGGER.exception("Generation failed")
        answer = f"生成失败：{type(exc).__name__}: {exc}"
        updated_history = history + [{"role": "assistant", "content": answer}]
        yield updated_history, updated_history


with gr.Blocks(title="ShizhenGPT 中医知识助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# ShizhenGPT 中医知识助手\n"
        "文本版 `ShizhenGPT-7B-LLM` · 4-bit 本地推理 · 首次提问会加载模型，需稍候。"
    )

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(type="messages", height=520, label="对话")
            message = gr.Textbox(
                label="问题",
                placeholder="例如：从中医基础理论角度，如何理解脾胃虚弱？",
                lines=2,
            )
            with gr.Row():
                send = gr.Button("发送", variant="primary")
                clear = gr.Button("清空对话")
        with gr.Column(scale=2):
            system_prompt = gr.Textbox(
                label="系统提示词",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=7,
            )
            temperature = gr.Slider(
                label="温度（0 为确定性输出）",
                minimum=0,
                maximum=1.5,
                value=0.7,
                step=0.1,
            )
            top_p = gr.Slider(
                label="Top-p", minimum=0.1, maximum=1.0, value=0.9, step=0.05
            )

    gr.Markdown(
        "**重要提示：** 该页面仅供研究与学习。模型输出不构成医疗诊断、处方或紧急医疗建议。"
    )
    history_state = gr.State([])

    submit_inputs = [message, history_state]
    submit_outputs = [message, chatbot, history_state]
    generate_inputs = [history_state, system_prompt, temperature, top_p]
    generate_outputs = [chatbot, history_state]

    send_event = send.click(add_user_message, submit_inputs, submit_outputs)
    send_event.then(add_assistant_message, generate_inputs, generate_outputs)
    enter_event = message.submit(add_user_message, submit_inputs, submit_outputs)
    enter_event.then(add_assistant_message, generate_inputs, generate_outputs)
    clear.click(lambda: ("", [], []), outputs=[message, chatbot, history_state])


if __name__ == "__main__":
    auth = (APP_USERNAME, APP_PASSWORD) if APP_USERNAME and APP_PASSWORD else None
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.getenv("SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        auth=auth,
        auth_message="受保护的内部演示服务" if auth else None,
    )
