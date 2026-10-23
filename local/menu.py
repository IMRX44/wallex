"""منوی اجرای محلی پیشرو — همه‌ی کارها روی سیستم خود شما (با اینترنت ایران).

اجرا: local/pishro.bat (ویندوز) یا bash local/pishro.sh (لینوکس/مک)
"""
from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
PY = sys.executable
PIP_MIRROR = "https://package-mirror.liara.ir/repository/pypi/simple"
NPM_MIRROR = "https://package-mirror.liara.ir/repository/npm/"

FIELDS = [
    ("WALLEX_API_KEY", "کلید API والکس (حساب چالش؛ فقط دسترسی معامله)", True),
    ("LLM_BASE_URL", "آدرس سرویس هوش مصنوعی لیارا (مثل https://ai.liara.ir/api/<id>/v1)", False),
    ("LLM_API_KEY", "کلید سرویس هوش مصنوعی لیارا", True),
    ("LLM_MODEL", "مدل هوش مصنوعی", False),
    ("CHALLENGE_END", "زمان پایان چالش (مثل 2026-10-26T23:50:00+03:30)", False),
    ("DASHBOARD_PASSWORD", "رمز دکمه‌های داشبورد", True),
    ("LIARA_API_TOKEN", "توکن API لیارا (برای دیپلوی)", True),
    ("LIARA_TEAM_ID", "شناسه‌ی تیم لیارا (اختیاری)", False),
    ("BALE_BOT_TOKEN", "توکن ربات بله برای اعلان (اختیاری)", True),
    ("BALE_CHAT_ID", "شناسه‌ی چت بله (اختیاری)", False),
]
DEFAULTS = {"LLM_MODEL": "anthropic/claude-opus-5.5", "PISHRO_MODE": "paper"}


def say(msg: str = "") -> None:
    print(msg, flush=True)


def read_env() -> dict[str, str]:
    out = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def write_env(vals: dict[str, str]) -> None:
    lines = ["# کلیدهای محلی پیشرو — این فایل را هرگز commit یا برای کسی ارسال نکنید"]
    lines += [f"{k}={v}" for k, v in vals.items() if v]
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(ENV, 0o600)
    except OSError:
        pass


def mask(v: str) -> str:
    return (v[:4] + "…" + v[-4:]) if len(v) > 10 else ("****" if v else "—")


def run(*args: str, env: dict | None = None, check: bool = False) -> int:
    say(f"\n$ {' '.join(args)}")
    e = {**os.environ, "PYTHONUTF8": "1", **(env or {})}
    return subprocess.run(list(args), cwd=ROOT, env=e, check=check).returncode


def pishro(*args: str, env: dict | None = None) -> int:
    return run(PY, "-m", "pishro", *args, env=env)


# ------------------------------------------------------------------ actions
def setup_keys() -> None:
    vals = {**DEFAULTS, **read_env()}
    say("\nبرای نگه داشتن مقدار فعلی فقط Enter بزنید.")
    for key, label, secret in FIELDS:
        cur = vals.get(key, "")
        shown = mask(cur) if secret else (cur or "—")
        prompt = f"{label}\n  [{key}] فعلی: {shown}\n  > "
        new = getpass.getpass(prompt) if secret else input(prompt)
        if new.strip():
            vals[key] = new.strip()
    write_env(vals)
    say(f"\nذخیره شد در {ENV} (فقط روی همین سیستم).")


def check() -> None:
    pishro("check")


def backtest() -> None:
    days = input("چند روز داده؟ [30]: ").strip() or "30"
    markets = input("حداکثر چند بازار؟ [40]: ").strip() or "40"
    if pishro("fetch", "--days", days, "--markets", markets) != 0:
        say("دانلود داده ناموفق بود؛ اتصال اینترنت به والکس را بررسی کنید.")
        return
    pishro("backtest")
    rep = ROOT / "reports" / "backtest.html"
    if not rep.exists():
        rep = ROOT / "data" / "reports" / "backtest.html"
    if rep.exists():
        webbrowser.open(rep.as_uri())
    say("مدل‌های ML در data/models ذخیره شدند و ربات از این به بعد از آن‌ها استفاده می‌کند.")


def serve(mode: str) -> None:
    env = {"PISHRO_MODE": mode}
    if mode == "live":
        say("\n⚠️  حالت پول واقعی: سفارش‌ها روی حساب والکس ثبت می‌شوند.")
        if input("برای تأیید دقیقاً بنویسید I_UNDERSTAND: ").strip() != "I_UNDERSTAND":
            say("لغو شد.")
            return
        env["LIVE_CONFIRM"] = "I_UNDERSTAND"
    port = os.environ.get("PORT", "8000")
    say(f"\nداشبورد: http://localhost:{port}   (برای توقف Ctrl+C)")
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:  # noqa: BLE001
        pass
    pishro("serve", "--port", port, env=env)


def _liara() -> str | None:
    exe = shutil.which("liara") or shutil.which("liara.cmd")
    if exe:
        return exe
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        say("Node.js نصب نیست. از https://nodejs.org نصب کنید و دوباره امتحان کنید.")
        return None
    say("در حال نصب Liara CLI (با میرور لیارا)…")
    if run(npm, "install", "-g", "@liara/cli", "--registry", NPM_MIRROR) != 0:
        run(npm, "install", "-g", "@liara/cli")
    return shutil.which("liara") or shutil.which("liara.cmd")


def deploy() -> None:
    vals = read_env()
    liara = _liara()
    if not liara:
        return
    app = input("نام اپ روی لیارا (یکتا، انگلیسی) [pishro]: ").strip() or "pishro"
    auth = []
    if vals.get("LIARA_API_TOKEN"):
        auth += ["--api-token", vals["LIARA_API_TOKEN"]]
    if vals.get("LIARA_TEAM_ID"):
        auth += ["--team-id", vals["LIARA_TEAM_ID"]]
    plan = input("پلن لیارا [g2-small]: ").strip() or "g2-small"
    run(liara, "create", "--app", app, "--platform", "docker", "--plan", plan, *auth)
    run(liara, "disk", "create", "--app", app, "--name", "data", "--size", "2", *auth)
    skip = {"LIARA_API_TOKEN", "LIARA_TEAM_ID"}
    envs = [f"{k}={v}" for k, v in vals.items() if v and k not in skip]
    mode = input("حالت اجرا روی لیارا: paper یا live؟ [paper]: ").strip() or "paper"
    envs.append(f"PISHRO_MODE={mode}")
    if mode == "live":
        if input("برای پول واقعی بنویسید I_UNDERSTAND: ").strip() != "I_UNDERSTAND":
            say("لغو شد.")
            return
        envs.append("LIVE_CONFIRM=I_UNDERSTAND")
    if envs:
        run(liara, "env", "set", "--app", app, *envs, "--force", *auth)
    models = ROOT / "data" / "models"
    if models.exists():                     # ship the models trained by the local backtest
        dst = ROOT / "models"
        for f in models.glob("*"):
            shutil.copy2(f, dst / f.name)
    reports = ROOT / "data" / "reports"
    if reports.exists():
        shutil.copytree(reports, ROOT / "reports", dirs_exist_ok=True)
    run(liara, "deploy", "--app", app, "--port", "8000", "--disks", "data:/app/data", "--detach", *auth)
    say(f"\nآدرس داشبورد: https://{app}.liara.run")


def export() -> None:
    pishro("export")
    say(f"تحویلی‌ها در {ROOT / 'deliverables'}")


def video() -> None:
    if not shutil.which("ffmpeg"):
        say("ffmpeg نصب نیست. ویندوز: winget install ffmpeg  |  مک: brew install ffmpeg  |  لینوکس: sudo apt install ffmpeg")
        return
    if run(PY, "-m", "pip", "install", "-q", "-r", "requirements-video.txt") != 0:
        run(PY, "-m", "pip", "install", "-q", "-i", PIP_MIRROR, "-r", "requirements-video.txt")
    run(PY, "-m", "playwright", "install", "chromium")
    src = input("داده از کجا؟ 1) همین سیستم (data/)  2) فایل‌های دانلودشده از لیارا [1]: ").strip() or "1"
    data_dir = "data" if src == "1" else input("مسیر پوشه‌ی داده (شامل pishro.db و decisions.jsonl): ").strip()
    url = input("آدرس داشبورد برای اسکرین‌شات (مثل https://pishro.liara.run یا Enter برای localhost:8000 یا - برای رد): ").strip()
    shot = ""
    if url != "-":
        url = url or "http://localhost:8000"
        shot = str(ROOT / "video" / "build" / "dashboard.png")
        code = ("from playwright.sync_api import sync_playwright\n"
                "with sync_playwright() as p:\n"
                " b=p.chromium.launch(); pg=b.new_page(viewport={'width':1600,'height':900},device_scale_factor=1.2,color_scheme='dark')\n"
                f" pg.goto('{url}', wait_until='networkidle'); pg.wait_for_timeout(3000)\n"
                f" pg.screenshot(path=r'{shot}', full_page=True); b.close()\n")
        (ROOT / "video" / "build").mkdir(parents=True, exist_ok=True)
        if run(PY, "-c", code) != 0:
            say("اسکرین‌شات گرفته نشد؛ ویدئو بدون صحنه‌ی داشبورد ساخته می‌شود.")
            shot = ""
    out = str(ROOT / "pishro_final.mp4")
    args = [PY, "-m", "video.render", "--data", data_dir, "--out", out]
    if shot:
        args += ["--shot", shot]
    if run(*args) == 0:
        say(f"\nویدئو آماده است: {out}")


def tests() -> None:
    run(PY, "-m", "pytest", "-q")


MENU = [
    ("ثبت/ویرایش کلیدها (.env)", setup_keys),
    ("بررسی اتصال (والکس، بایننس، نوبیتکس، حساب، هوش مصنوعی)", check),
    ("دانلود داده‌ی واقعی + بک‌تست + آموزش مدل ML", backtest),
    ("تست سایه (قیمت زنده، پول مجازی) + داشبورد", lambda: serve("paper")),
    ("معامله‌ی واقعی + داشبورد", lambda: serve("live")),
    ("دیپلوی روی لیارا", deploy),
    ("خروجی تحویلی‌ها (لاگ تصمیم، معاملات، منحنی سرمایه)", export),
    ("ساخت ویدئوی نهایی با داده‌ی واقعی", video),
    ("اجرای تست‌های خودکار", tests),
]


def main() -> None:
    os.chdir(ROOT)
    while True:
        say("\n" + "=" * 60)
        say("  پیشرو — منوی اجرای محلی")
        say("=" * 60)
        env = read_env()
        say(f"  کلید والکس: {'✓' if env.get('WALLEX_API_KEY') else '✗'}   هوش مصنوعی: "
            f"{'✓' if env.get('LLM_API_KEY') else '✗'}   لیارا: {'✓' if env.get('LIARA_API_TOKEN') else '✗'}")
        for i, (label, _) in enumerate(MENU, 1):
            say(f"  {i}) {label}")
        say("  0) خروج")
        choice = input("\nانتخاب: ").strip()
        if choice in ("0", "q", ""):
            return
        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            try:
                MENU[int(choice) - 1][1]()
            except KeyboardInterrupt:
                say("\nمتوقف شد.")
            time.sleep(0.3)


if __name__ == "__main__":
    main()
