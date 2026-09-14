"""Gradio 前端界面"""

import traceback
import time
from pathlib import Path
from typing import Any

import gradio as gr
import torch

from qwen_tts_webui.task_manager.call_queue import (
    wrap_gradio_call,
    wrap_queued_call,
)
from qwen_tts_webui.config_manager.config import (
    CONFIG_PATH,
    OUTPUT_PATH,
    QWEN_TTS_BASE_MODEL_LIST,
    QWEN_TTS_CUSTOM_VOICE_MODEL_LIST,
    QWEN_TTS_VOICE_DESIGN_MODEL_LIST,
    ATTN_IMPL_LIST,
)
from qwen_tts_webui.backend.memory_manager import (
    MODEL_PRECISION_LIST,
    get_available_devices,
    OutOfMemoryError,
)
from qwen_tts_webui.config_manager.shared import (
    opts,
    state,
    get_backend,
)
from qwen_tts_webui.silence_ui import add_silence_tab, apply_head_tail_silence
from qwen_tts_webui.version import VERSION


def format_memory_status() -> str:
    """格式化当前 TTS 模型与显存状态"""
    status = get_backend().get_memory_status()
    if status["loaded"]:
        model_line = f"TTS 模型已驻留显存: <code>{status['model_name']}</code>"
    else:
        model_line = "TTS 模型未加载 (显存已释放, 下次生成会重新加载)"
    if status["cuda_available"]:
        vram_line = (
            f"GPU 显存: 已分配 {status['allocated_mb']} MB · "
            f"已保留 {status['reserved_mb']} MB · "
            f"空闲 {status['free_mb']} MB / 总计 {status['total_mb']} MB"
        )
    else:
        vram_line = "未检测到 CUDA"
    return (
        "<div style='padding: 10px 12px; margin: 8px 0 16px; border: 1px solid #d0d7de; "
        "border-radius: 8px; background: #f6f8fa; font-size: 0.95em; line-height: 1.6;'>"
        f"<div>{model_line}</div><div>{vram_line}</div>"
        "<div style='color: #666;'>去跑 FlashTalk 等其他 GPU 任务前, 请先点「卸载模型并释放显存」。</div>"
        "</div>"
    )


def unload_tts_model_fn() -> str:
    """手动卸载 TTS 模型并回收显存"""
    backend = get_backend()
    if backend.model is None:
        gr.Info("当前没有驻留在显存中的 TTS 模型")
        return format_memory_status()
    backend.unload_model()
    gr.Info("已卸载 TTS 模型并回收显存, 可以启动 FlashTalk 等其他 GPU 任务")
    return format_memory_status()


def update_token_counter(
    text: str,
) -> str:
    """更新 token 计数器

    Args:
        text (str): 要计算 token 数量的文本

    Returns:
        str: HTML 格式的 token 数量信息
    """
    if not text or text.strip() == "":
        return "<div style='color: #666; font-size: 0.9em; margin-bottom: 4px;'>Token 数量: 0</div>"

    try:
        backend = get_backend()
        if backend.model is None:
            return "<div style='color: #666; font-size: 0.9em; margin-bottom: 4px;'>Token 数量: NaN (模型未加载, 可点击开始生成按钮加载模型)</div>"

        token_count = backend.count_tokens(text)
        return f"<div style='color: #666; font-size: 0.9em; margin-bottom: 4px;'>Token 数量: {token_count}</div>"
    except Exception as e:
        return f"<div style='color: #666; font-size: 0.9em; margin-bottom: 4px;'>Token 数量: 计算失败 ({str(e)})</div>"


def create_ui() -> gr.Blocks:
    """创建 Gradio 界面

    Returns:
        gr.Blocks: Gradio 界面实例
    """
    with gr.Blocks(title=f"Qwen TTS WebUI v{VERSION}") as demo:
        gr.Markdown(f"# Qwen TTS WebUI v{VERSION}")
        with gr.Row():
            unload_model_btn = gr.Button("卸载模型并释放显存", variant="stop")
            refresh_memory_btn = gr.Button("刷新显存状态", variant="secondary")
        memory_status = gr.HTML(value=format_memory_status())
        unload_model_btn.click(  # pylint: disable=no-member
            fn=wrap_queued_call(unload_tts_model_fn),
            outputs=[memory_status],
        )
        refresh_memory_btn.click(  # pylint: disable=no-member
            fn=format_memory_status,
            outputs=[memory_status],
        )
        demo.load(
            fn=format_memory_status,
            outputs=[memory_status],
            queue=False,
        )

        with gr.Tabs():
            with gr.Tab("声音生成", id="voice_generation"):
                with gr.Row():
                    with gr.Column():
                        gen_model_choices = QWEN_TTS_CUSTOM_VOICE_MODEL_LIST + opts.extra_custom_voice_models
                        gen_model = gr.Dropdown(label="模型选择", choices=gen_model_choices, value=gen_model_choices[0], interactive=True)
                        gen_token_counter = gr.HTML(value="<div style='color: #666; font-size: 0.9em; margin-bottom: 4px;'>Token 数量: 0</div>")
                        gen_text = gr.Textbox(label="合成文本", placeholder="请输入要合成的文本...", lines=5, show_label=False)
                        gen_instruct = gr.Textbox(label="声音特征描述", placeholder="例如: 用温柔的语气说。", lines=3)
                        with gr.Row():
                            gen_speaker = gr.Dropdown(label="发言人", choices=[], value=None, interactive=True)
                            gen_language = gr.Dropdown(label="语言", choices=["auto"], value="auto", interactive=True)
                        gen_segment = gr.Checkbox(label="分段生成 (按行切分文本)", value=False)
                    with gr.Column():
                        with gr.Row():
                            gen_button = gr.Button("开始生成", variant="primary")
                            stop_gen_button = gr.Button("终止", variant="stop", visible=False)

                        gen_audio_list = gr.State([])

                        @gr.render(inputs=gen_audio_list)
                        def render_gen_audios(paths: list[str]) -> None:
                            if not paths:
                                gr.Markdown("暂无生成的音频，请点击左侧按钮。")
                            else:
                                with gr.Column():
                                    for i, path in enumerate(paths):
                                        with gr.Group():
                                            gr.Audio(value=path, label=f"生成的音频 #{i + 1}", show_label=True)

                        gr.Markdown(
                            "## 使用说明\n"
                            "1. 在`合成文本`中输入要合成的文本, `声音特征描述`描述合成出来的声音是什么特征的, 即可点击`开始生成`.\n"
                            "2. 初次生成需要下载模型, 请耐心等待模型下载完成, 可在控制台中查看模型的下载进度.\n"
                            "3. `发言人`和`语言`选项默认分别只有`default`和`auto`, 当点击`开始生成`并且生成结束后, 这两个选项将会刷新, 刷新后即可选择其他选项.\n"
                            "4. 如果需要分段生成音频, 可启用`分段生成 (按行切分文本)`选项.\n"
                            "5. 生成的音频可在`音频浏览`中查看, 或者在 Qwen TTS WebUI 的 `outputs` 文件夹中查看.\n"
                            "6. 如需使用本地路径的模型 (使用自己手动下载下来的模型), 请先在`设置`页面的`额外模型列表`中添加本地模型路径, 并将`下载模型的 API 类型`设置为 `local`, 保存设置后即可在模型选择中使用本地模型. 通常情况下不需要修改该设置, 默认设置下会将模型自动从 HuggingFace / ModelScope 下载到本地并使用.\n"
                            "7. 运行时如果提示模型文件损坏, 可尝试将报错中提示的路径进行删除.\n"
                            "8. 推理设备会自动进行选择, 如果需要手动指定推理设备, 可修改设置中的`推理设备`选项.\n"
                            "9. 如果因修改设置后导致的报错, 请重置设置并重新启动 Qwen TTS WebUI."
                        )

            with gr.Tab("声音设计", id="voice_design"):
                with gr.Row():
                    with gr.Column():
                        design_model_choices = QWEN_TTS_VOICE_DESIGN_MODEL_LIST + opts.extra_voice_design_models
                        design_model = gr.Dropdown(label="模型选择", choices=design_model_choices, value=design_model_choices[0], interactive=True)
                        design_token_counter = gr.HTML(value="<div style='color: #666; font-size: 0.9em; margin-bottom: 4px;'>Token 数量: 0</div>")
                        design_text = gr.Textbox(label="合成文本", placeholder="请输入要合成的文本...", lines=5, show_label=False)
                        design_instruct = gr.Textbox(
                            label="声音特征描述", placeholder="例如：体现撒娇稚嫩的女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。", lines=3
                        )
                        design_language = gr.Dropdown(label="语言", choices=["auto"], value="auto", interactive=True)
                        design_segment = gr.Checkbox(label="分段生成 (按行切分文本)", value=False)
                    with gr.Column():
                        with gr.Row():
                            design_button = gr.Button("开始生成", variant="primary")
                            stop_design_button = gr.Button("终止", variant="stop", visible=False)

                        design_audio_list = gr.State([])

                        @gr.render(inputs=design_audio_list)
                        def render_design_audios(paths: list[str]) -> None:
                            if not paths:
                                gr.Markdown("暂无生成的音频，请点击左侧按钮。")
                            else:
                                with gr.Column():
                                    for i, path in enumerate(paths):
                                        with gr.Group():
                                            gr.Audio(value=path, label=f"生成的音频样本 #{i + 1}", show_label=True)

                        gr.Markdown(
                            "## 使用说明\n"
                            "1. 在`合成文本`中输入要合成的文本, `声音特征描述`描述合成出来的声音是什么特征的, 即可点击`开始生成`.\n"
                            "2. 初次生成需要下载模型, 请耐心等待模型下载完成, 可在控制台中查看模型的下载进度.\n"
                            "3. `语言`选项默认只有`auto`, 当点击`开始生成`并且生成结束后, 这个选项将会刷新, 刷新后即可选择其他选项.\n"
                            "4. 如果需要分段生成音频, 可启用`分段生成 (按行切分文本)`选项.\n"
                            "5. 生成的音频可在`音频浏览`中查看, 或者在 Qwen TTS WebUI 的 `outputs` 文件夹中查看.\n"
                            "6. 如需使用本地路径的模型 (使用自己手动下载下来的模型), 请先在`设置`页面的`额外模型列表`中添加本地模型路径, 并将`下载模型的 API 类型`设置为 `local`, 保存设置后即可在模型选择中使用本地模型. 通常情况下不需要修改该设置, 默认设置下会将模型自动从 HuggingFace / ModelScope 下载到本地并使用.\n"
                            "7. 运行时如果提示模型文件损坏, 可尝试将报错中提示的路径进行删除.\n"
                            "8. 推理设备会自动进行选择, 如果需要手动指定推理设备, 可修改设置中的`推理设备`选项.\n"
                            "9. 如果因修改设置后导致的报错, 请重置设置并重新启动 Qwen TTS WebUI."
                        )

            with gr.Tab("声音克隆", id="voice_clone"):
                with gr.Row():
                    with gr.Column():
                        clone_model_choices = QWEN_TTS_BASE_MODEL_LIST + opts.extra_voice_clone_models
                        clone_model = gr.Dropdown(label="模型选择", choices=clone_model_choices, value=clone_model_choices[0], interactive=True)
                        clone_token_counter = gr.HTML(value="<div style='color: #666; font-size: 0.9em; margin-bottom: 4px;'>Token 数量: 0</div>")
                        clone_text = gr.Textbox(label="合成文本", placeholder="请输入要合成的文本...", lines=5, show_label=False)
                        clone_language = gr.Dropdown(label="语言", choices=["auto"], value="auto", interactive=True)
                        clone_audio = gr.Audio(label="参考音频文件", type="filepath")
                        clone_ref_text = gr.Textbox(label="参考音频文本描述", placeholder="请输入参考音频对应的文本内容 (描述参考音频文件中说话的内容, 可不填)", lines=2)
                        clone_segment = gr.Checkbox(label="分段生成 (按行切分文本)", value=False)
                    with gr.Column():
                        gr.Markdown("### 第一阶段：语音克隆")
                        with gr.Row():
                            clone_button = gr.Button("开始生成", variant="primary")
                            stop_clone_button = gr.Button("终止", variant="stop", visible=False)

                        clone_source_list = gr.State([])
                        clone_audio_list = gr.State([])

                        @gr.render(inputs=clone_audio_list)
                        def render_clone_audios(paths: list[str]) -> None:
                            if not paths:
                                gr.Markdown("暂无生成的音频，请先完成第一阶段。")
                            else:
                                with gr.Column():
                                    for i, path in enumerate(paths):
                                        with gr.Group():
                                            gr.Audio(value=path, label=f"说话人音频 #{i + 1}", show_label=True)

                        gr.Markdown("### 第二阶段：给生成音频加静音（可选）")
                        gr.Markdown("基于第一阶段的原始克隆结果加静音，无需下载再上传。可改时长后再次点击。")
                        with gr.Row():
                            clone_head_silence = gr.Number(label="开头静音（秒）", value=0, minimum=0, maximum=3600)
                            clone_tail_silence = gr.Number(label="末尾静音（秒）", value=3, minimum=0, maximum=3600)
                        clone_pad_btn = gr.Button("给生成音频加静音", variant="secondary")
                        clone_pad_msg = gr.Markdown("完成第一阶段后，可在此给说话人音频加首尾静音。")

                        gr.Markdown(
                            "## 使用说明\n"
                            "1. 第一阶段: 在`合成文本`中输入要合成的文本, 再将一段音频导入`参考音频文件`中, 点击`开始生成`.\n"
                            "2. 第二阶段(可选): 生成完成后, 在右侧设置开头/末尾静音秒数 (默认末尾 3 秒), 点击`给生成音频加静音`. 无需下载再上传.\n"
                            "3. 初次生成需要下载模型, 请耐心等待模型下载完成, 可在控制台中查看模型的下载进度.\n"
                            "4. 导入`参考音频文件`的音频只需要几秒的长度即可, 当然音频长度可以更长, 可自行尝试.\n"
                            "5. `参考音频文本描述 `填写导入的音频中, 音频内说话的内容, 这个选项可不填.\n"
                            "6. `语言`选项默认只有`auto`, 当点击`开始生成`并且生成结束后, 这个选项将会刷新, 刷新后即可选择其他选项.\n"
                            "7. 如果需要分段生成音频, 可启用`分段生成 (按行切分文本)`选项.\n"
                            "8. 生成的音频可在`音频浏览`中查看, 或者在 Qwen TTS WebUI 的 `outputs` 文件夹中查看.\n"
                            "9. 如需使用本地路径的模型 (使用自己手动下载下来的模型), 请先在`设置`页面的`额外模型列表`中添加本地模型路径, 并将`下载模型的 API 类型`设置为 `local`, 保存设置后即可在模型选择中使用本地模型. 通常情况下不需要修改该设置, 默认设置下会将模型自动从 HuggingFace / ModelScope 下载到本地并使用.\n"
                            "10. 运行时如果提示模型文件损坏, 可尝试将报错中提示的路径进行删除.\n"
                            "11. 推理设备会自动进行选择, 如果需要手动指定推理设备, 可修改设置中的`推理设备`选项.\n"
                            "12. 如果因修改设置后导致的报错, 请重置设置并重新启动 Qwen TTS WebUI."
                        )

            add_silence_tab()

            with gr.Tab("音频浏览", id="audio_browser"):
                with gr.Row():
                    with gr.Column():
                        refresh_btn = gr.Button("刷新文件列表", variant="secondary")
                        explorer = gr.FileExplorer(root_dir=OUTPUT_PATH, glob="**/*.wav", file_count="single", label="选择音频文件")
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                audio_player = gr.Audio(label="播放器", type="filepath")
                                gr.Markdown(f"当前音频存储路径: `{OUTPUT_PATH.as_posix()}`")

                refresh_btn.click(  # pylint: disable=no-member
                    fn=gr.update,
                    outputs=explorer,
                )
                explorer.change(  # pylint: disable=no-member
                    fn=lambda x: x,
                    inputs=explorer,
                    outputs=audio_player,
                )

            with gr.Tab("设置", id="settings"):
                with gr.Row():
                    save_settings_btn = gr.Button("保存设置", variant="primary")
                    reset_settings_btn = gr.Button("重置设置", variant="secondary")

                gr.Markdown("**修改设置前请三思而后行, 如果遇到修改设置后导致报错, 请点击重置设置后再重新启动.**")
                gr.Markdown("### 模型加载设置")
                api_type = gr.Dropdown(label="下载模型的 API 类型", choices=["huggingface", "modelscope", "local"], value=opts.api_type)
                gr.Markdown(
                    "**提示**: 当使用本地路径的模型时, 需要将 API 类型设置为 `local`. 如果您未手动下载任何模型并且未在**额外模型列表**选项填写本地已下载好的模型路径, 请不要修改该设置!\n"
                    "- `huggingface`: 从 HuggingFace 下载模型 (优先使用本地缓存)\n"
                    "- `modelscope`: 从 ModelScope 下载模型 (优先使用本地缓存)\n"
                    "- `local`: 直接从本地路径加载模型,不进行任何网络请求"
                )

                available_devices = ["auto"] + [str(d) for d in get_available_devices()]
                device_map = gr.Dropdown(label="推理设备", choices=available_devices, value=str(opts.device_map))
                dtype = gr.Dropdown(label="推理精度", choices=[str(p) for p in MODEL_PRECISION_LIST], value=str(opts.dtype))
                attn_implementation = gr.Dropdown(label="加速方案", choices=[str(None)] + ATTN_IMPL_LIST, value=str(opts.attn_implementation))

                gr.Markdown("### 额外模型列表")
                gr.Markdown("在下方输入框中每行输入一个模型名称或路径, 这些模型将会添加到对应功能的模型选择列表中")
                extra_custom_voice_models = gr.Textbox(
                    label="声音生成额外模型",
                    placeholder="每行一个模型名称或路径\n例如:\nQwen/Qwen-TTS-Custom\n/path/to/local/model",
                    lines=4,
                    value="\n".join(opts.extra_custom_voice_models),
                )
                extra_voice_design_models = gr.Textbox(
                    label="声音设计额外模型",
                    placeholder="每行一个模型名称或路径\n例如:\nQwen/Qwen-TTS-Design\n/path/to/local/model",
                    lines=4,
                    value="\n".join(opts.extra_voice_design_models),
                )
                extra_voice_clone_models = gr.Textbox(
                    label="声音克隆额外模型",
                    placeholder="每行一个模型名称或路径\n例如:\nQwen/Qwen-TTS-Clone\n/path/to/local/model",
                    lines=4,
                    value="\n".join(opts.extra_voice_clone_models),
                )

                gr.Markdown("### 采样参数")
                do_sample = gr.Checkbox(label="是否使用采样", value=opts.do_sample)
                top_k = gr.Slider(label="Top-k 采样参数", minimum=1, maximum=100, step=1, value=opts.top_k)
                top_p = gr.Slider(label="Top-p 采样参数", minimum=0.0, maximum=1.0, step=0.01, value=opts.top_p)
                temperature = gr.Slider(label="采样温度", minimum=0.0, maximum=2.0, step=0.01, value=opts.temperature)
                repetition_penalty = gr.Slider(label="重复惩罚系数", minimum=1.0, maximum=2.0, step=0.01, value=opts.repetition_penalty)
                subtalker_dosample = gr.Checkbox(label="子说话者采样开关", value=opts.subtalker_dosample)
                subtalker_top_k = gr.Slider(label="子说话者 Top-k 采样", minimum=1, maximum=100, step=1, value=opts.subtalker_top_k)
                subtalker_top_p = gr.Slider(label="子说话者 Top-p 采样", minimum=0.0, maximum=1.0, step=0.01, value=opts.subtalker_top_p)
                subtalker_temperature = gr.Slider(label="子说话者采样温度", minimum=0.0, maximum=2.0, step=0.01, value=opts.subtalker_temperature)
                max_new_tokens = gr.Slider(label="最大生成 Token 数", minimum=1, maximum=4096, step=1, value=opts.max_new_tokens)

                def save_settings(
                    api_type_val: str,
                    device_map_val: str,
                    dtype_val: str,
                    attn_impl_val: str,
                    extra_custom_voice_models_val: str,
                    extra_voice_design_models_val: str,
                    extra_voice_clone_models_val: str,
                    do_sample_val: bool,
                    top_k_val: int,
                    top_p_val: float,
                    temperature_val: float,
                    repetition_penalty_val: float,
                    subtalker_dosample_val: bool,
                    subtalker_top_k_val: int,
                    subtalker_top_p_val: float,
                    subtalker_temperature_val: float,
                    max_new_tokens_val: int,
                ) -> tuple[Any, Any, Any]:
                    """保存设置

                    Args:
                        api_type_val (str): API 类型
                        device_map_val (str): 推理设备
                        dtype_val (str): 推理精度
                        attn_impl_val (str): 加速方案
                        extra_custom_voice_models_val (str): 声音生成额外模型列表
                        extra_voice_design_models_val (str): 声音设计额外模型列表
                        extra_voice_clone_models_val (str): 声音克隆额外模型列表
                        do_sample_val (bool): 是否使用采样
                        top_k_val (int): Top-k 采样参数
                        top_p_val (float): Top-p 采样参数
                        temperature_val (float): 采样温度
                        repetition_penalty_val (float): 重复惩罚系数
                        subtalker_dosample_val (bool): 子说话者采样开关
                        subtalker_top_k_val (int): 子说话者 Top-k 采样
                        subtalker_top_p_val (float): 子说话者 Top-p 采样
                        subtalker_temperature_val (float): 子说话者采样温度
                        max_new_tokens_val (int): 最大生成 Token 数

                    Returns:
                        tuple[Any, Any, Any]: 更新后的三个模型选择下拉框
                    """
                    opts.api_type = api_type_val
                    opts.device_map = device_map_val
                    opts.dtype = dtype_val
                    opts.attn_implementation = None if attn_impl_val == str(None) else attn_impl_val
                    # 处理额外模型列表: 按行分割并过滤空行
                    opts.extra_custom_voice_models = [x.strip() for x in extra_custom_voice_models_val.splitlines() if x.strip()]
                    opts.extra_voice_design_models = [x.strip() for x in extra_voice_design_models_val.splitlines() if x.strip()]
                    opts.extra_voice_clone_models = [x.strip() for x in extra_voice_clone_models_val.splitlines() if x.strip()]
                    opts.do_sample = do_sample_val
                    opts.top_k = top_k_val
                    opts.top_p = top_p_val
                    opts.temperature = temperature_val
                    opts.repetition_penalty = repetition_penalty_val
                    opts.subtalker_dosample = subtalker_dosample_val
                    opts.subtalker_top_k = subtalker_top_k_val
                    opts.subtalker_top_p = subtalker_top_p_val
                    opts.subtalker_temperature = subtalker_temperature_val
                    opts.max_new_tokens = max_new_tokens_val
                    opts.save(CONFIG_PATH)
                    gr.Info("设置已保存")
                    # 更新模型选择下拉框
                    gen_choices = QWEN_TTS_CUSTOM_VOICE_MODEL_LIST + opts.extra_custom_voice_models
                    design_choices = QWEN_TTS_VOICE_DESIGN_MODEL_LIST + opts.extra_voice_design_models
                    clone_choices = QWEN_TTS_BASE_MODEL_LIST + opts.extra_voice_clone_models
                    return (
                        gr.update(choices=gen_choices),
                        gr.update(choices=design_choices),
                        gr.update(choices=clone_choices),
                    )

                save_settings_btn.click(  # pylint: disable=no-member
                    fn=save_settings,
                    inputs=[
                        api_type,
                        device_map,
                        dtype,
                        attn_implementation,
                        extra_custom_voice_models,
                        extra_voice_design_models,
                        extra_voice_clone_models,
                        do_sample,
                        top_k,
                        top_p,
                        temperature,
                        repetition_penalty,
                        subtalker_dosample,
                        subtalker_top_k,
                        subtalker_top_p,
                        subtalker_temperature,
                        max_new_tokens,
                    ],
                    outputs=[gen_model, design_model, clone_model],
                )

                def reset_settings() -> list[Any]:
                    """重置设置

                    Returns:
                        list[Any]: 包含所有重置后设置项值的列表, 用于更新 Gradio 组件
                    """
                    opts.reset()
                    opts.save(CONFIG_PATH)
                    gr.Info("设置已重置为默认值")
                    # 更新模型选择下拉框
                    gen_choices = QWEN_TTS_CUSTOM_VOICE_MODEL_LIST + opts.extra_custom_voice_models
                    design_choices = QWEN_TTS_VOICE_DESIGN_MODEL_LIST + opts.extra_voice_design_models
                    clone_choices = QWEN_TTS_BASE_MODEL_LIST + opts.extra_voice_clone_models
                    return [
                        opts.api_type,
                        str(opts.device_map),
                        str(opts.dtype),
                        str(opts.attn_implementation),
                        "\n".join(opts.extra_custom_voice_models),
                        "\n".join(opts.extra_voice_design_models),
                        "\n".join(opts.extra_voice_clone_models),
                        opts.do_sample,
                        opts.top_k,
                        opts.top_p,
                        opts.temperature,
                        opts.repetition_penalty,
                        opts.subtalker_dosample,
                        opts.subtalker_top_k,
                        opts.subtalker_top_p,
                        opts.subtalker_temperature,
                        opts.max_new_tokens,
                        gr.update(choices=gen_choices),
                        gr.update(choices=design_choices),
                        gr.update(choices=clone_choices),
                    ]

                reset_settings_btn.click(  # pylint: disable=no-member
                    fn=reset_settings,
                    inputs=[],
                    outputs=[
                        api_type,
                        device_map,
                        dtype,
                        attn_implementation,
                        extra_custom_voice_models,
                        extra_voice_design_models,
                        extra_voice_clone_models,
                        do_sample,
                        top_k,
                        top_p,
                        temperature,
                        repetition_penalty,
                        subtalker_dosample,
                        subtalker_top_k,
                        subtalker_top_p,
                        subtalker_temperature,
                        max_new_tokens,
                        gen_model,
                        design_model,
                        clone_model,
                    ],
                )

        def update_metadata(
            current_speaker: str,
            current_language: str,
            model_name: str | None = None,
        ) -> tuple[str, str | None, Any, Any]:
            """更新模型元数据

            Args:
                current_speaker (str):
                    当前选择的发言人
                current_language (str):
                    当前选择的语言
                model_name (str | None):
                    目标模型名; 模型未加载时用于从 config 枚举发言人/语言

            Returns:
                (tuple[str, str | None, Any, Any]): 实际发言人, 实际语言, 发言人组件更新, 语言组件更新
            """
            speakers = get_backend().get_supported_speakers(model_name) or []
            languages = get_backend().get_supported_languages(model_name) or []

            speaker_choices = list(speakers)
            if "auto" in languages:
                language_choices = languages
            else:
                language_choices = ["auto"] + languages

            actual_speaker = current_speaker
            if (current_speaker is None or current_speaker == "default") and speakers:
                actual_speaker = speakers[0]

            actual_language = None if current_language == "auto" else current_language

            return (
                actual_speaker,
                actual_language,
                gr.update(choices=speaker_choices, value=actual_speaker),
                gr.update(choices=language_choices, value=current_language),
            )

        def refresh_metadata(
            current_speaker: str,
            current_language: str,
            model_name: str | None = None,
        ) -> tuple[Any, Any]:
            """刷新发言人/语言下拉框选项 (不生成音频)"""
            _, _, speaker_update, language_update = update_metadata(
                current_speaker, current_language, model_name
            )
            return speaker_update, language_update

        def update_metadata_simple(
            current_language: str,
        ) -> tuple[str | None, Any]:
            """更新模型元数据 (仅语言)

            Args:
                current_language (str): 当前选择的语言

            Returns:
                tuple[str | None, Any]: 实际语言, 语言组件更新
            """
            languages = get_backend().get_supported_languages() or []
            if "auto" in languages:
                language_choices = languages
            else:
                language_choices = ["auto"] + languages
            actual_language = None if current_language == "auto" else current_language
            return actual_language, gr.update(choices=language_choices, value=current_language)

        def generate_voice_fn(
            model_name: str,
            text: str,
            instruct: str,
            speaker: str,
            language: str,
            segment_gen: bool,
        ) -> tuple[list[str] | None, Any, Any]:
            """声音生成处理函数

            Args:
                model_name (str): 所选模型名称
                text (str): 待合成文本
                instruct (str): 声音特征描述指令
                speaker (str): 发言人标识
                language (str): 语言标识
                segment_gen (bool): 分段生成, 设置为 True 时将文本按行分段并分别生成

            Returns:
                (tuple[list[str] | None, Any, Any]): 生成的音频路径, 发言人组件更新, 语言组件更新
            """
            output_paths: list[str] = []
            if segment_gen:
                text_list = [x.strip() for x in text.splitlines() if x.strip() != ""]
            else:
                text_list = [text]
            try:
                start_time = time.perf_counter()
                gr.Info(f"加载 {model_name} 模型中")
                get_backend().load_model(
                    model_name=model_name,
                    api_type=opts.api_type,
                    device_map=opts.device_map,
                    dtype=getattr(torch, opts.dtype.split(".")[-1]),
                    attn_implementation=opts.attn_implementation,
                )
                actual_speaker, actual_language, speaker_update, language_update = update_metadata(speaker, language, model_name)
                for i, t in enumerate(text_list, start=1):
                    gr.Info(f"生成音频中 ({i}/{len(text_list)})")
                    output_path = get_backend().generate_custom_voice(
                        text=t,
                        speaker=actual_speaker,
                        language=actual_language,
                        instruct=instruct,
                        do_sample=opts.do_sample,
                        top_k=opts.top_k,
                        top_p=opts.top_p,
                        temperature=opts.temperature,
                        repetition_penalty=opts.repetition_penalty,
                        subtalker_dosample=opts.subtalker_dosample,
                        subtalker_top_k=opts.subtalker_top_k,
                        subtalker_top_p=opts.subtalker_top_p,
                        subtalker_temperature=opts.subtalker_temperature,
                        max_new_tokens=opts.max_new_tokens,
                    )
                    if state.interrupted:
                        return None, speaker_update, language_update
                    output_paths.append(str(output_path))

                gr.Info(f"音频生成完成, 耗时: {(time.perf_counter() - start_time):.2f}s")
                return output_paths, speaker_update, language_update
            except OutOfMemoryError as e:
                traceback.print_exc()
                gr.Warning(f"生成音频所需的显存不足: {str(e)}\n\n详细错误信息请在控制台中查看")
                return None, gr.update(), gr.update()
            except Exception as e:  # pylint: disable=duplicate-except
                traceback.print_exc()
                gr.Warning(f"生成失败: {str(e)}\n\n详细错误信息请在控制台中查看")
                return None, gr.update(), gr.update()

        def generate_design_fn(
            model_name: str,
            text: str,
            instruct: str,
            language: str,
            segment_gen: bool,
        ) -> tuple[list[str] | None, Any]:
            """声音设计处理函数

            Args:
                model_name (str): 所选模型名称
                text (str): 待合成文本
                instruct (str): 声音描述指令
                language (str): 语言标识
                segment_gen (bool): 分段生成, 设置为 True 时将文本按行分段并分别生成

            Returns:
                (tuple[list[str] | None, Any]): 生成的音频路径, 语言组件更新
            """
            output_paths: list[str] = []
            if segment_gen:
                text_list = [x.strip() for x in text.splitlines() if x.strip() != ""]
            else:
                text_list = [text]
            try:
                start_time = time.perf_counter()
                gr.Info(f"加载 {model_name} 模型中")
                get_backend().load_model(
                    model_name=model_name,
                    api_type=opts.api_type,
                    device_map=opts.device_map,
                    dtype=getattr(torch, opts.dtype.split(".")[-1]),
                    attn_implementation=opts.attn_implementation,
                )
                actual_language, language_update = update_metadata_simple(language)
                for i, t in enumerate(text_list, start=1):
                    gr.Info(f"生成音频中 ({i}/{len(text_list)})")
                    output_path = get_backend().generate_voice_design(
                        text=t,
                        instruct=instruct,
                        language=actual_language,
                        do_sample=opts.do_sample,
                        top_k=opts.top_k,
                        top_p=opts.top_p,
                        temperature=opts.temperature,
                        repetition_penalty=opts.repetition_penalty,
                        subtalker_dosample=opts.subtalker_dosample,
                        subtalker_top_k=opts.subtalker_top_k,
                        subtalker_top_p=opts.subtalker_top_p,
                        subtalker_temperature=opts.subtalker_temperature,
                        max_new_tokens=opts.max_new_tokens,
                    )
                    if state.interrupted:
                        return None, language_update
                    output_paths.append(str(output_path))

                gr.Info(f"音频生成完成, 耗时: {(time.perf_counter() - start_time):.2f}s")
                return output_paths, language_update
            except OutOfMemoryError as e:
                traceback.print_exc()
                gr.Warning(f"生成音频所需的显存不足: {str(e)}\n\n详细错误信息请在控制台中查看")
                return None, gr.update()
            except Exception as e:  # pylint: disable=duplicate-except
                traceback.print_exc()
                gr.Warning(f"生成失败: {str(e)}\n\n详细错误信息请在控制台中查看")
                return None, gr.update()

        def generate_clone_fn(
            model_name: str,
            text: str,
            language: str,
            ref_audio: str,
            ref_text: str,
            segment_gen: bool,
        ) -> tuple[list[str] | None, list[str] | None, Any]:
            """声音克隆处理函数

            Args:
                model_name (str): 所选模型名称
                text (str): 待合成文本
                language (str): 语言标识
                ref_audio (str): 参考音频路径
                ref_text (str): 参考音频的文本描述内容
                segment_gen (bool): 分段生成, 设置为 True 时将文本按行分段并分别生成

            Returns:
                (tuple[list[str] | None, list[str] | None, Any]): 展示音频路径, 原始克隆路径, 语言组件更新
            """
            output_paths: list[str] = []
            if segment_gen:
                text_list = [x.strip() for x in text.splitlines() if x.strip() != ""]
            else:
                text_list = [text]

            if not ref_audio:
                gr.Warning("请先上传参考音频文件")
                return None, None, gr.update()
            try:
                start_time = time.perf_counter()
                gr.Info(f"加载 {model_name} 模型中")
                get_backend().load_model(
                    model_name=model_name,
                    api_type=opts.api_type,
                    device_map=opts.device_map,
                    dtype=getattr(torch, opts.dtype.split(".")[-1]),
                    attn_implementation=opts.attn_implementation,
                )
                actual_language, language_update = update_metadata_simple(language)
                for i, t in enumerate(text_list, start=1):
                    gr.Info(f"生成音频中 ({i}/{len(text_list)})")
                    output_path = get_backend().generate_voice_clone(
                        text=t,
                        language=actual_language,
                        ref_audio=Path(ref_audio),
                        ref_text=ref_text,
                        x_vector_only_mode=(ref_text is None or ref_text.strip() == ""),
                        do_sample=opts.do_sample,
                        top_k=opts.top_k,
                        top_p=opts.top_p,
                        temperature=opts.temperature,
                        repetition_penalty=opts.repetition_penalty,
                        subtalker_dosample=opts.subtalker_dosample,
                        subtalker_top_k=opts.subtalker_top_k,
                        subtalker_top_p=opts.subtalker_top_p,
                        subtalker_temperature=opts.subtalker_temperature,
                        max_new_tokens=opts.max_new_tokens,
                    )
                    if state.interrupted:
                        return None, None, language_update
                    output_paths.append(str(output_path))

                gr.Info(f"第一阶段完成, 耗时: {(time.perf_counter() - start_time):.2f}s。如需加静音，请继续第二阶段。")
                return output_paths, output_paths, language_update
            except OutOfMemoryError as e:
                traceback.print_exc()
                gr.Warning(f"生成音频所需的显存不足: {str(e)}\n\n详细错误信息请在控制台中查看")
                return None, None, gr.update()
            except Exception as e:  # pylint: disable=duplicate-except
                traceback.print_exc()
                gr.Warning(f"生成失败: {str(e)}\n\n详细错误信息请在控制台中查看")
                return None, None, gr.update()

        def interrupt_fn() -> None:
            """中断任务"""
            state.interrupt()

        gen_event = gen_button.click(  # pylint: disable=no-member
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[gen_button, stop_gen_button],
            queue=False,
        ).then(
            fn=wrap_queued_call(wrap_gradio_call(generate_voice_fn)),
            inputs=[gen_model, gen_text, gen_instruct, gen_speaker, gen_language, gen_segment],
            outputs=[gen_audio_list, gen_speaker, gen_language],
        )
        gen_event.then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[gen_button, stop_gen_button],
            queue=False,
        )
        gen_event.then(
            fn=format_memory_status,
            outputs=[memory_status],
            queue=False,
        )
        # 页面加载及切换模型时, 立即刷新发言人/语言下拉框 (无需先生成一次)
        gen_model.change(  # pylint: disable=no-member
            fn=refresh_metadata,
            inputs=[gen_speaker, gen_language, gen_model],
            outputs=[gen_speaker, gen_language],
            queue=False,
        )
        demo.load(
            fn=refresh_metadata,
            inputs=[gen_speaker, gen_language, gen_model],
            outputs=[gen_speaker, gen_language],
            queue=False,
        )
        stop_gen_button.click(  # pylint: disable=no-member
            fn=lambda: (interrupt_fn(), gr.update(visible=True), gr.update(visible=False))[1:],
            cancels=[gen_event],
            outputs=[gen_button, stop_gen_button],
            queue=False,
        )

        design_event = design_button.click(  # pylint: disable=no-member
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[design_button, stop_design_button],
            queue=False,
        ).then(
            fn=wrap_queued_call(wrap_gradio_call(generate_design_fn)),
            inputs=[design_model, design_text, design_instruct, design_language, design_segment],
            outputs=[design_audio_list, design_language],
        )
        design_event.then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[design_button, stop_design_button],
            queue=False,
        )
        design_event.then(
            fn=format_memory_status,
            outputs=[memory_status],
            queue=False,
        )
        stop_design_button.click(  # pylint: disable=no-member
            fn=lambda: (interrupt_fn(), gr.update(visible=True), gr.update(visible=False))[1:],
            cancels=[design_event],
            outputs=[design_button, stop_design_button],
            queue=False,
        )

        clone_event = clone_button.click(  # pylint: disable=no-member
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[clone_button, stop_clone_button],
            queue=False,
        ).then(
            fn=wrap_queued_call(wrap_gradio_call(generate_clone_fn)),
            inputs=[clone_model, clone_text, clone_language, clone_audio, clone_ref_text, clone_segment],
            outputs=[clone_audio_list, clone_source_list, clone_language],
        )
        clone_event.then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[clone_button, stop_clone_button],
            queue=False,
        )
        clone_event.then(
            fn=format_memory_status,
            outputs=[memory_status],
            queue=False,
        )
        clone_pad_btn.click(  # pylint: disable=no-member
            fn=apply_head_tail_silence,
            inputs=[clone_source_list, clone_head_silence, clone_tail_silence],
            outputs=[clone_audio_list, clone_pad_msg],
        )
        stop_clone_button.click(  # pylint: disable=no-member
            fn=lambda: (interrupt_fn(), gr.update(visible=True), gr.update(visible=False))[1:],
            cancels=[clone_event],
            outputs=[clone_button, stop_clone_button],
            queue=False,
        )

        gen_text.change(  # pylint: disable=no-member
            fn=wrap_queued_call(update_token_counter),
            inputs=[gen_text],
            outputs=[gen_token_counter],
        )

        design_text.change(  # pylint: disable=no-member
            fn=wrap_queued_call(update_token_counter),
            inputs=[design_text],
            outputs=[design_token_counter],
        )

        clone_text.change(  # pylint: disable=no-member
            fn=wrap_queued_call(update_token_counter),
            inputs=[clone_text],
            outputs=[clone_token_counter],
        )

    return demo
