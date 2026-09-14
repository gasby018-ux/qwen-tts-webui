"""静音拼接 Gradio 页面"""

from pathlib import Path
from typing import Any

import gradio as gr

from qwen_tts_webui.config_manager.config import OUTPUT_PATH
from qwen_tts_webui.ffmpeg_utils import (
    FFmpegError,
    concat_with_silence,
    generate_silence,
    pad_head_tail,
    pad_silence,
)
from qwen_tts_webui.utils import generate_filename

SILENCE_OUTPUT_PATH = OUTPUT_PATH / "silence"
SILENCE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

AUDIO_FILE_TYPES = [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".wma"]


def as_audio_path(value: Any) -> Path | None:
    """把 Gradio 上传组件的返回值转成本地路径"""
    if value is None or value == "":
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, dict):
        path = value.get("path") or value.get("name")
        return Path(path) if path else None
    path = getattr(value, "path", None) or getattr(value, "name", None)
    if path:
        return Path(str(path))
    return Path(str(value))


def _format_info(info: dict) -> str:
    return (
        f"处理完成：时长 {round(float(info.get('duration') or 0), 3)}s · "
        f"{info.get('sample_rate')} Hz · {info.get('channels')} 声道"
    )


def _output_path(prefix: str, fmt: str) -> Path:
    fmt = (fmt or "wav").lower()
    if fmt not in {"wav", "mp3"}:
        raise ValueError("输出格式仅支持 wav 或 mp3。")
    return SILENCE_OUTPUT_PATH / f"{prefix}_{generate_filename()}.{fmt}"


def _fail(message: str) -> tuple[None, None, str]:
    gr.Warning(message)
    return None, None, message


def generate_silence_fn(
    duration: float,
    sample_rate: str,
    channels: str,
    fmt: str,
) -> tuple[str | None, str | None, str]:
    """生成指定时长的静音音频"""
    try:
        output = _output_path("silence", fmt)
        info = generate_silence(
            output_path=output,
            duration=duration,
            sample_rate=int(sample_rate),
            channels=1 if channels == "单声道" else 2,
            fmt=fmt,
        )
        path = str(output)
        return path, path, _format_info(info)
    except (ValueError, FFmpegError) as exc:
        return _fail(str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return _fail(f"生成失败: {exc}")


def concat_silence_fn(
    audio_a: Any,
    audio_b: Any,
    duration: float,
    fmt: str,
) -> tuple[str | None, str | None, str]:
    """将两段音频用指定时长静音拼接"""
    path_a = as_audio_path(audio_a)
    path_b = as_audio_path(audio_b)
    if path_a is None or not path_a.is_file():
        return _fail("请先上传第一段音频")
    if path_b is None or not path_b.is_file():
        return _fail("请先上传第二段音频")
    try:
        output = _output_path("concat", fmt)
        info = concat_with_silence(
            audio_a=path_a,
            audio_b=path_b,
            output_path=output,
            silence_duration=duration,
            fmt=fmt,
        )
        path = str(output)
        return path, path, _format_info(info)
    except (ValueError, FFmpegError) as exc:
        return _fail(str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return _fail(f"拼接失败: {exc}")


def pad_silence_fn(
    audio: Any,
    duration: float,
    position: str,
    fmt: str,
) -> tuple[str | None, str | None, str]:
    """在音频开头或末尾追加指定时长静音"""
    src = as_audio_path(audio)
    if src is None or not src.is_file():
        return _fail("请先上传音频")
    pos = "start" if position in {"开头", "首部", "start"} else "end"
    prefix = "prepend" if pos == "start" else "append"
    try:
        output = _output_path(prefix, fmt)
        info = pad_silence(
            audio=src,
            output_path=output,
            silence_duration=duration,
            position=pos,
            fmt=fmt,
        )
        path = str(output)
        return path, path, _format_info(info)
    except (ValueError, FFmpegError) as exc:
        return _fail(str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return _fail(f"处理失败: {exc}")


def apply_head_tail_silence(
    source_paths: list[str] | None,
    head_duration: float,
    tail_duration: float,
) -> tuple[Any, str]:
    """对已生成的说话人音频做第二阶段首尾静音。始终基于原始克隆结果，可重复调整时长。"""
    if not source_paths:
        gr.Warning("请先完成第一阶段：语音克隆生成音频")
        return gr.update(), "请先完成第一阶段：语音克隆生成音频"
    try:
        padded: list[str] = []
        last_info: dict = {}
        for src in source_paths:
            src_path = as_audio_path(src)
            if src_path is None or not src_path.is_file():
                raise ValueError("找不到已生成的说话人音频，请先重新克隆。")
            output = _output_path("clone_pad", src_path.suffix.lstrip(".") or "wav")
            last_info = pad_head_tail(
                audio=src_path,
                output_path=output,
                head_duration=head_duration,
                tail_duration=tail_duration,
            )
            padded.append(str(output))
        msg = f"第二阶段完成，共处理 {len(padded)} 条。"
        if last_info:
            msg += " " + _format_info(last_info)
        gr.Info(msg)
        return padded, msg
    except (ValueError, FFmpegError) as exc:
        gr.Warning(str(exc))
        return gr.update(), str(exc)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        message = f"加静音失败: {exc}"
        gr.Warning(message)
        return gr.update(), message


def add_silence_tab() -> None:
    """在当前 Tabs 中添加静音拼接页"""
    with gr.Tab("静音拼接", id="silence_tools"):
        gr.Markdown(
            "用 ffmpeg 生成静音、把两段音频用静音拼接，或在任意音频开头/末尾补静音。"
            "结果会保存到 `outputs/silence`，也可在「音频浏览」中查看。"
        )
        with gr.Tabs():
            with gr.Tab("生成静音音频"):
                with gr.Row():
                    with gr.Column():
                        silence_duration = gr.Number(label="静音时长（秒）", value=8, minimum=0.01, maximum=3600)
                        silence_sr = gr.Dropdown(
                            label="采样率",
                            choices=["8000", "16000", "22050", "24000", "32000", "44100", "48000"],
                            value="44100",
                        )
                        silence_ch = gr.Radio(label="声道", choices=["单声道", "立体声"], value="立体声")
                        silence_fmt = gr.Radio(label="输出格式", choices=["wav", "mp3"], value="wav")
                        silence_btn = gr.Button("生成静音", variant="primary")
                    with gr.Column():
                        silence_audio = gr.Audio(label="预览", type="filepath")
                        silence_file = gr.File(label="下载")
                        silence_msg = gr.Markdown("等待生成")
                silence_btn.click(  # pylint: disable=no-member
                    fn=generate_silence_fn,
                    inputs=[silence_duration, silence_sr, silence_ch, silence_fmt],
                    outputs=[silence_audio, silence_file, silence_msg],
                )

            with gr.Tab("两段音频中间插静音"):
                with gr.Row():
                    with gr.Column():
                        concat_a = gr.File(label="第一段音频", file_types=AUDIO_FILE_TYPES, type="filepath")
                        concat_b = gr.File(label="第二段音频", file_types=AUDIO_FILE_TYPES, type="filepath")
                        concat_duration = gr.Number(label="中间静音时长（秒）", value=1, minimum=0.01, maximum=3600)
                        concat_fmt = gr.Radio(label="输出格式", choices=["wav", "mp3"], value="wav")
                        concat_btn = gr.Button("拼接音频", variant="primary")
                    with gr.Column():
                        concat_audio = gr.Audio(label="预览", type="filepath")
                        concat_file = gr.File(label="下载")
                        concat_msg = gr.Markdown("等待拼接")
                concat_btn.click(  # pylint: disable=no-member
                    fn=concat_silence_fn,
                    inputs=[concat_a, concat_b, concat_duration, concat_fmt],
                    outputs=[concat_audio, concat_file, concat_msg],
                )

            with gr.Tab("音频首尾加静音"):
                with gr.Row():
                    with gr.Column():
                        pad_src = gr.File(label="上传音频", file_types=AUDIO_FILE_TYPES, type="filepath")
                        pad_position = gr.Radio(label="静音位置", choices=["开头", "末尾"], value="开头")
                        pad_duration = gr.Number(label="静音时长（秒）", value=5, minimum=0.01, maximum=3600)
                        pad_fmt = gr.Radio(label="输出格式", choices=["wav", "mp3"], value="wav")
                        pad_btn = gr.Button("添加静音", variant="primary")
                    with gr.Column():
                        pad_audio = gr.Audio(label="预览", type="filepath")
                        pad_file = gr.File(label="下载")
                        pad_msg = gr.Markdown("等待处理")
                pad_btn.click(  # pylint: disable=no-member
                    fn=pad_silence_fn,
                    inputs=[pad_src, pad_duration, pad_position, pad_fmt],
                    outputs=[pad_audio, pad_file, pad_msg],
                )

        gr.Markdown(
            "## 使用说明\n"
            "1. `生成静音音频`: 指定秒数后生成一段纯静音，可预览和下载。\n"
            "2. `两段音频中间插静音`: 上传两段音频，按指定秒数插入静音后拼接。采样率会按第一段音频对齐。\n"
            "3. `音频首尾加静音`: 上传任意音频，选择在开头或末尾补上指定时长的静音。\n"
            "4. 支持 wav / mp3 / m4a / aac / flac / ogg / opus / webm / wma。\n"
            "5. 可把 TTS 生成的 wav 再上传到这里做拼接或补静音。处理后的 wav 会出现在「音频浏览」。"
        )
