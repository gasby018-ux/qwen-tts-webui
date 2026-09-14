"""ffmpeg 静音生成与音频拼接工具"""

import json
import shutil
import subprocess
from pathlib import Path


FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

ALLOWED_OUTPUT_FORMATS = {"wav", "mp3"}
MAX_DURATION_SEC = 3600.0
MIN_DURATION_SEC = 0.01


class FFmpegError(RuntimeError):
    """ffmpeg 执行失败"""


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise FFmpegError("未找到 ffmpeg，请先安装后再使用本工具。")
    if not shutil.which("ffprobe"):
        raise FFmpegError("未找到 ffprobe，请先安装 ffmpeg 完整套件。")


def validate_duration(duration: float) -> float:
    try:
        duration = float(duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("静音时长必须是数字（秒）。") from exc
    if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
        raise ValueError(f"静音时长需在 {MIN_DURATION_SEC} ~ {MAX_DURATION_SEC} 秒之间。")
    return duration


def _run(cmd: list[str]) -> None:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise FFmpegError(err[-2000:] if err else "ffmpeg 执行失败。")


def probe_audio(path: Path) -> dict:
    cmd = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,channel_layout,codec_name",
        "-show_entries",
        "format=duration,format_name",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError("无法读取音频信息，请确认上传的是有效音频文件。")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise FFmpegError("文件中没有检测到音频流。")
    stream = streams[0]
    fmt = data.get("format") or {}
    sample_rate = int(stream.get("sample_rate") or 44100)
    channels = int(stream.get("channels") or 2)
    try:
        duration = float(fmt.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "channel_layout": stream.get("channel_layout") or ("stereo" if channels >= 2 else "mono"),
        "codec": stream.get("codec_name"),
        "duration": duration,
        "format_name": fmt.get("format_name"),
    }


def _layout(channels: int) -> str:
    return "stereo" if channels >= 2 else "mono"


def _output_args(fmt: str) -> list[str]:
    fmt = fmt.lower()
    if fmt not in ALLOWED_OUTPUT_FORMATS:
        raise ValueError("输出格式仅支持 wav 或 mp3。")
    if fmt == "mp3":
        return ["-c:a", "libmp3lame", "-q:a", "2"]
    return ["-c:a", "pcm_s16le"]


def generate_silence(
    output_path: Path,
    duration: float,
    sample_rate: int = 44100,
    channels: int = 2,
    fmt: str = "wav",
) -> dict:
    ensure_ffmpeg()
    duration = validate_duration(duration)
    sample_rate = int(sample_rate)
    channels = 2 if int(channels) >= 2 else 1
    if sample_rate not in {8000, 16000, 22050, 24000, 32000, 44100, 48000}:
        raise ValueError("采样率不支持，请选择 8000/16000/22050/24000/32000/44100/48000。")

    layout = _layout(channels)
    cmd = [
        FFMPEG,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl={layout}",
        "-t",
        f"{duration:.3f}",
        *_output_args(fmt),
        str(output_path),
    ]
    _run(cmd)
    return probe_audio(output_path)


def concat_with_silence(
    audio_a: Path,
    audio_b: Path,
    output_path: Path,
    silence_duration: float,
    fmt: str = "wav",
) -> dict:
    ensure_ffmpeg()
    silence_duration = validate_duration(silence_duration)
    info = probe_audio(audio_a)
    probe_audio(audio_b)
    sr = info["sample_rate"]
    ch = 2 if info["channels"] >= 2 else 1
    layout = _layout(ch)
    filter_complex = (
        f"[0:a]aresample={sr},aformat=sample_fmts=fltp:channel_layouts={layout}[a0];"
        f"[1:a]aresample={sr},aformat=sample_fmts=fltp:channel_layouts={layout}[a1];"
        f"anullsrc=r={sr}:cl={layout}:d={silence_duration:.3f}[silence];"
        f"[a0][silence][a1]concat=n=3:v=0:a=1[out]"
    )
    cmd = [
        FFMPEG,
        "-y",
        "-i",
        str(audio_a),
        "-i",
        str(audio_b),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        *_output_args(fmt),
        str(output_path),
    ]
    _run(cmd)
    return probe_audio(output_path)


def pad_silence(
    audio: Path,
    output_path: Path,
    silence_duration: float,
    position: str = "end",
    fmt: str = "wav",
) -> dict:
    """在音频开头或末尾拼接指定时长静音。"""
    ensure_ffmpeg()
    silence_duration = validate_duration(silence_duration)
    info = probe_audio(audio)
    sr = info["sample_rate"]
    ch = 2 if info["channels"] >= 2 else 1
    layout = _layout(ch)
    if position in {"start", "head", "开头", "首部"}:
        concat_graph = "[silence][a0]concat=n=2:v=0:a=1[out]"
    else:
        concat_graph = "[a0][silence]concat=n=2:v=0:a=1[out]"
    filter_complex = (
        f"[0:a]aresample={sr},aformat=sample_fmts=fltp:channel_layouts={layout}[a0];"
        f"anullsrc=r={sr}:cl={layout}:d={silence_duration:.3f}[silence];"
        f"{concat_graph}"
    )
    cmd = [
        FFMPEG,
        "-y",
        "-i",
        str(audio),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        *_output_args(fmt),
        str(output_path),
    ]
    _run(cmd)
    return probe_audio(output_path)


def append_silence(
    audio: Path,
    output_path: Path,
    silence_duration: float,
    fmt: str = "wav",
) -> dict:
    return pad_silence(audio, output_path, silence_duration, position="end", fmt=fmt)


def prepend_silence(
    audio: Path,
    output_path: Path,
    silence_duration: float,
    fmt: str = "wav",
) -> dict:
    return pad_silence(audio, output_path, silence_duration, position="start", fmt=fmt)


def _optional_duration(duration: float | str | None, label: str) -> float:
    if duration is None or duration == "":
        return 0.0
    try:
        value = float(duration)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是数字（秒）。") from exc
    if value < 0:
        raise ValueError(f"{label}不能为负数。")
    if value == 0:
        return 0.0
    return validate_duration(value)


def pad_head_tail(
    audio: Path,
    output_path: Path,
    head_duration: float = 0,
    tail_duration: float = 0,
    fmt: str = "wav",
) -> dict:
    """在音频开头和/或末尾拼接静音。时长为 0 表示该侧不加。"""
    ensure_ffmpeg()
    head = _optional_duration(head_duration, "开头静音时长")
    tail = _optional_duration(tail_duration, "末尾静音时长")
    if head <= 0 and tail <= 0:
        raise ValueError("请至少设置开头或末尾静音时长。")

    info = probe_audio(audio)
    sr = info["sample_rate"]
    ch = 2 if info["channels"] >= 2 else 1
    layout = _layout(ch)
    parts = [f"[0:a]aresample={sr},aformat=sample_fmts=fltp:channel_layouts={layout}[a0]"]
    labels: list[str] = []
    if head > 0:
        parts.append(f"anullsrc=r={sr}:cl={layout}:d={head:.3f}[head]")
        labels.append("[head]")
    labels.append("[a0]")
    if tail > 0:
        parts.append(f"anullsrc=r={sr}:cl={layout}:d={tail:.3f}[tail]")
        labels.append("[tail]")
    parts.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[out]")
    cmd = [
        FFMPEG,
        "-y",
        "-i",
        str(audio),
        "-filter_complex",
        ";".join(parts),
        "-map",
        "[out]",
        *_output_args(fmt),
        str(output_path),
    ]
    _run(cmd)
    return probe_audio(output_path)
