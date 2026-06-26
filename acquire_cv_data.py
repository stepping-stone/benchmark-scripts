import argparse
import json
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

from PIL import Image
import pypdfium2 as pdfium

HF_REPO = "d4rk3r/resumes-raw-pdf"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def hf_files(repo: str) -> list[str]:
    try:
        from huggingface_hub import list_repo_files

        return sorted(list_repo_files(repo, repo_type="dataset"))
    except Exception:
        url = f"https://huggingface.co/api/datasets/{repo}?blobs=false"
        req = urllib.request.Request(url, headers=auth_headers())
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return sorted(s["rfilename"] for s in data.get("siblings", []))


def auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def download_hf_file(repo: str, filename: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return

    quoted_repo = urllib.parse.quote(repo, safe="/")
    quoted_file = urllib.parse.quote(filename, safe="/")
    url = f"https://huggingface.co/datasets/{quoted_repo}/resolve/main/{quoted_file}?download=true"
    req = urllib.request.Request(url, headers=auth_headers())
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with urllib.request.urlopen(req, timeout=300) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(dst)


def resize_max_edge(img: Image.Image, max_edge: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_edge / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def write_image(img: Image.Image, out: Path, idx: int, max_edge: int) -> Path:
    path = out / f"cv_{idx:05d}.png"
    resize_max_edge(img, max_edge).save(path, optimize=True)
    return path


def local_inputs(src: Path) -> list[Path]:
    if not src.exists():
        return []
    allowed = IMAGE_EXTS | {".pdf"}
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in allowed)


def render_inputs(
    paths: Iterable[Path],
    out: Path,
    max_edge: int,
    dpi: int,
) -> int:
    manifest = out / "manifest.jsonl"
    written = 0
    with manifest.open("w", encoding="utf-8") as mf:
        for path in paths:
            if path.suffix.lower() == ".pdf":
                pdf = pdfium.PdfDocument(str(path))
                for page_no in range(len(pdf)):
                    img = pdf[page_no].render(scale=dpi / 72).to_pil()
                    out_path = write_image(img, out, written, max_edge)
                    mf.write(
                        json.dumps(
                            {
                                "image": str(out_path),
                                "source": str(path),
                                "page": page_no + 1,
                            }
                        )
                        + "\n"
                    )
                    written += 1
            else:
                with Image.open(path) as img:
                    out_path = write_image(img, out, written, max_edge)
                mf.write(
                    json.dumps({"image": str(out_path), "source": str(path), "page": 1})
                    + "\n"
                )
                written += 1
    return written


def download_resume_pdfs(repo: str, pdf_dir: Path, count: int) -> list[Path]:
    files = [f for f in hf_files(repo) if f.lower().endswith(".pdf")]
    if not files:
        raise SystemExit(f"no PDFs found in Hugging Face dataset {repo}")

    selected = files if count <= 0 else files[:count]
    local: list[Path] = []
    for i, filename in enumerate(selected, 1):
        dst = pdf_dir / filename
        print(f"[{i}/{len(selected)}] {filename}")
        download_hf_file(repo, filename, dst)
        local.append(dst)
    return local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", type=Path, help="optional local directory of real CV PDFs/images"
    )
    parser.add_argument(
        "--repo",
        default=HF_REPO,
        help="Hugging Face dataset repo containing resume PDFs",
    )
    parser.add_argument("--pdf-dir", type=Path, default=Path("cv_bench_pdfs"))
    parser.add_argument("--out", type=Path, default=Path("cv_bench_data"))
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="limit number of CVs to process (0 = all). Every page of each is rendered.",
    )
    parser.add_argument("--max-edge", type=int, default=1540)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.src:
        inputs = local_inputs(args.src)
        if not inputs:
            raise SystemExit(f"no PDFs/images found under {args.src}")
        if args.count > 0:
            inputs = inputs[: args.count]
    else:
        inputs = download_resume_pdfs(args.repo, args.pdf_dir, args.count)

    render_inputs(inputs, args.out, args.max_edge, args.dpi)


if __name__ == "__main__":
    main()
