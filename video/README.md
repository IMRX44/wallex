# ساخت ویدئوی دمو

ویدئوی ۵ دقیقه‌ای (1080p، ۳۰ فریم، با موسیقی و افکت صوتی ساخته‌شده در کد) **مستقیماً از داده‌ی خود ربات** ساخته می‌شود؛
هیچ عددی دستی نوشته نمی‌شود.

```bash
pip install playwright==1.55.0          # + ffmpeg و Chromium
# بعد از اجرای واقعی چالش (پوشه‌ی data شامل pishro.db، decisions.jsonl و reports/):
python -m video.render --data data --shot path/to/dashboard.png --out pishro_final.mp4
# نسخه‌ی نمایشی بدون داده‌ی واقعی (روی داده‌ی مصنوعی، با برچسب «نسخه‌ی نمایشی»):
python -m video.demo_run && python -m video.render --demo
```

- `player.html`: تایم‌لاین ۱۳ صحنه (مقدمه، چالش، ایده با نمودار متحرک، معماری، مدل ML، ایجنت Claude، ریسک، بک‌تست، مونت‌کارلو، اجرای ۳ روزه، داشبورد، اجرای مجدد، پایان)
- `collect.py`: جمع‌آوری اعداد از دیتابیس، لاگ تصمیم و گزارش بک‌تست
- `audio.py`: موسیقی و افکت‌ها (سینتی‌سایز در کد، بدون مشکل حق نشر)
- `render.py`: رندر فریم‌به‌فریم با Chromium و ffmpeg
- فونت Vazirmatn تحت مجوز SIL OFL (`assets/OFL.txt`)
