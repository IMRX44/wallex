"""Render the ~5-minute demo video (1080p, 30 fps, H.264 + AAC).

    python -m video.render --data data                 # real run: data dir with pishro.db / reports
    python -m video.render --demo                      # demo data (run video.demo_run first)
    python -m video.render --demo --preview 20,60,150  # a few PNG frames only

Needs ffmpeg and Playwright/Chromium. Frames are rendered deterministically from
``player.html`` (``render(t)``) by several browser pages in parallel, encoded per
segment, concatenated, and muxed with the synthesized soundtrack (``audio.py``).
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import shutil
import subprocess
import time
from pathlib import Path

from .audio import make_track
from .collect import collect, safe

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"


def _chrome() -> str | None:
    c = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    return c[-1] if c else None


def build_page(data: dict, shot: Path | None) -> Path:
    out = BUILD / "page"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(HERE / "assets", out / "assets", dirs_exist_ok=True)
    if shot and shot.exists():
        shutil.copy2(shot, out / "shot.png")
        data["shot"] = "shot.png"
    html = (HERE / "player.html").read_text(encoding="utf-8")
    inject = "<script>window.DATA = " + json.dumps(safe(data), ensure_ascii=False) + ";</script>\n<script>"
    html = html.replace("<script>", inject, 1)
    (out / "index.html").write_text(html, encoding="utf-8")
    return out / "index.html"


async def _open(browser, page_path: Path):
    pg = await browser.new_page(viewport={"width": 1920, "height": 1080})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    await pg.goto(page_path.as_uri())
    await pg.wait_for_function("window.READY === true")
    await pg.evaluate("document.fonts.ready.then(() => true)")
    await pg.wait_for_timeout(400)
    if errors:
        raise RuntimeError("page errors: " + "; ".join(errors))
    return pg


async def _segment(browser, page_path: Path, frames: range, fps: int, out: Path, crf: int) -> None:
    pg = await _open(browser, page_path)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg",
                             "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                             "-r", str(fps), str(out)], stdin=subprocess.PIPE)
    for i in frames:
        await pg.evaluate("t => window.render(t)", i / fps)
        proc.stdin.write(await pg.screenshot(type="jpeg", quality=93))
    proc.stdin.close()
    proc.wait()
    await pg.close()


async def render(page_path: Path, out_mp4: Path, fps: int = 30, workers: int = 4, crf: int = 18,
                 preview: list[float] | None = None) -> None:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=_chrome(), args=["--disable-gpu", "--font-render-hinting=none"])
        pg = await _open(browser, page_path)
        duration = await pg.evaluate("window.DURATION")
        cues = await pg.evaluate("window.CUES")
        if preview:
            for t in preview:
                await pg.evaluate("t => window.render(t)", t)
                await pg.screenshot(path=str(BUILD / f"preview_{int(t):03d}.png"))
            await browser.close()
            print("previews written to", BUILD)
            return
        await pg.close()
        n = int(duration * fps)
        chunk = (n + workers - 1) // workers
        segs = [BUILD / f"seg_{k}.mp4" for k in range(workers)]
        t0 = time.time()
        await asyncio.gather(*[_segment(browser, page_path, range(k * chunk, min(n, (k + 1) * chunk)), fps, segs[k], crf)
                               for k in range(workers)])
        await browser.close()
    print(f"frames rendered in {time.time() - t0:.0f}s")
    lst = BUILD / "segments.txt"
    lst.write_text("".join(f"file '{s.name}'\n" for s in segs))
    video_only = BUILD / "video_only.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(video_only)], check=True, cwd=BUILD)
    wav = BUILD / "soundtrack.wav"
    make_track(duration, cues, str(wav))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_only), "-i", str(wav), "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out_mp4)], check=True)
    print("video:", out_mp4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="data dir of a real run (pishro.db, decisions.jsonl, reports/)")
    ap.add_argument("--demo", action="store_true", help="use video/build/demo_data (from video.demo_run)")
    ap.add_argument("--shot", default=None, help="dashboard screenshot to show (PNG)")
    ap.add_argument("--out", default=str(BUILD / "pishro_demo.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--preview", default="")
    a = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    data_dir = Path(a.data) if a.data else BUILD / "demo_data"
    data = collect(data_dir, demo=a.demo or not a.data)
    shot = Path(a.shot) if a.shot else next((p for p in (BUILD / "shots" / "mid_dashboard_full.png",
                                                         BUILD / "shots" / "dashboard_full.png") if p.exists()), None)
    page = build_page(data, shot)
    preview = [float(x) for x in a.preview.split(",") if x] or None
    asyncio.run(render(page, Path(a.out), a.fps, a.workers, preview=preview))


if __name__ == "__main__":
    main()
