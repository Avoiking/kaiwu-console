"""从 Ollama 模型目录导入 GGUF：

- 解析 manifests/registry.ollama.ai 下所有标签清单
- 按 model 层 digest 去重（qwen3-8b 三标签共用一份权重 → 只导入一次）
- 硬链接 blob 成 .gguf（同盘零拷贝，秒完成）；跨盘回退复制
- 带 projector（mmproj）的视觉模型照常列出并导入主权重，
  但标注「视觉能力需 mmproj，二期支持」，导入的纯文本推理可用
- params 层读出 num_ctx / temperature / stop，展示用
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MEDIA_MODEL = "application/vnd.ollama.image.model"
MEDIA_PROJECTOR = "application/vnd.ollama.image.projector"
MEDIA_PARAMS = "application/vnd.ollama.image.params"


def _blob_path(blobs_dir: Path, digest: str) -> Path:
    return blobs_dir / digest.replace(":", "-")


def _read_params(blobs_dir: Path, digest: str) -> dict:
    p = _blob_path(blobs_dir, digest)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def scan(ollama_dir: str | Path) -> list[dict]:
    """扫描全部标签，返回按权重去重后的条目列表：

    [{
      "tags": ["qwen3-8b-32k", "qwen3-8b-16k", ...],   # 共用同一权重的标签
      "tag": "qwen3-8b-32k",                            # 首选标签名
      "digest": "sha256:...",
      "size": 5225374496,
      "num_ctx": 32768, "temperature": 0.6, "top_k": ..., "top_p": ...,
      "has_projector": False,        # 视觉模型 True（VL）
      "blob": Path, "params_digest": str|None,
    }]
    """
    root = Path(ollama_dir)
    manifests_root = root / "manifests" / "registry.ollama.ai"
    blobs = root / "blobs"
    if not manifests_root.exists():
        return []

    entries: dict[str, dict] = {}   # digest -> entry

    for mf in manifests_root.rglob("*"):
        if not mf.is_file():
            continue
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        layers = data.get("layers", [])
        model_layer = next((l for l in layers if l.get("mediaType") == MEDIA_MODEL), None)
        if not model_layer:
            continue
        digest = model_layer.get("digest", "")
        if not digest:
            continue
        # manifest 相对路径：library/<name>/<tag> → 标签名用 <name>:<tag> 的 <name> 部分
        rel = mf.relative_to(manifests_root)
        tag = rel.parts[-1]
        name = rel.parts[-2] if len(rel.parts) >= 2 else tag
        params_digest = next(
            (l.get("digest") for l in layers if l.get("mediaType") == MEDIA_PARAMS), None)
        has_projector = any(l.get("mediaType") == MEDIA_PROJECTOR for l in layers)

        if digest in entries:
            entries[digest]["tags"].append(name)
            continue
        params = _read_params(blobs, params_digest) if params_digest else {}
        entries[digest] = {
            "tags": [name],
            "tag": name,
            "digest": digest,
            "size": model_layer.get("size", 0),
            "num_ctx": params.get("num_ctx"),
            "temperature": params.get("temperature"),
            "top_k": params.get("top_k"),
            "top_p": params.get("top_p"),
            "stop": params.get("stop", []),
            "has_projector": has_projector,
            "blob": _blob_path(blobs, digest),
            "params_digest": params_digest,
        }

    out = list(entries.values())
    out.sort(key=lambda e: e["tag"])
    return out


def import_model(entry: dict, out_dir: str | Path, progress=None) -> dict:
    """把权重硬链接为 <out_dir>/<tag>.gguf。返回 {"ok","path","message"}"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{entry['tag']}.gguf"
    src: Path = entry["blob"]

    if not src.exists():
        return {"ok": False, "path": "", "message": f"blob 不存在：{src}"}

    if target.exists():
        try:
            if os.path.getsize(target) == entry["size"]:
                return {"ok": True, "path": str(target), "message": "已存在，跳过"}
            target.unlink()  # 大小不一致说明是坏副本，重建
        except OSError as e:
            return {"ok": False, "path": "", "message": f"无法覆盖旧文件：{e}"}

    try:
        os.link(str(src), str(target))
        return {"ok": True, "path": str(target), "message": "硬链接完成（零拷贝）"}
    except OSError:
        # 跨卷或文件系统不支持 → 回退复制
        try:
            total = entry["size"]
            done = 0
            with open(src, "rb") as fsrc, open(target, "wb") as fdst:
                while True:
                    chunk = fsrc.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    fdst.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
            return {"ok": True, "path": str(target), "message": "复制完成（跨卷回退）"}
        except Exception as e:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            return {"ok": False, "path": "", "message": f"导入失败：{e}"}


def human_size(n: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} GB"
