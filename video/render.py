"""Motion-graphics promo video built from the 4-panel gym photo collage.

The photos are only cropped, scaled and moved (Ken Burns / slides) — never
retouched or regenerated, so facial features stay exactly as in the source.

Usage: python3 render.py <collage.png> <out.mp4>
"""
import math
import os
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
BPM = 120
BEAT = 60 / BPM
DUR = 21.0
N = int(DUR * FPS)

YELLOW = (255, 214, 10)
RED = (230, 57, 70)
WHITE = (255, 255, 255)
BLACK = (8, 8, 8)

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_C = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
if not os.path.exists(FONT_C):
    FONT_C = FONT_B

# (box in source collage, focus point for Ken Burns as fraction of panel)
PANELS = {
    "flex": ((0, 0, 833, 941), (0.52, 0.30)),
    "portrait": ((837, 0, 1671, 442), (0.42, 0.32)),
    "punch": ((837, 446, 1217, 941), (0.42, 0.40)),
    "back": ((1221, 446, 1671, 941), (0.50, 0.40)),
}

# start, end, panel, english word, arabic word
SCENES = [
    (2.5, 6.5, "flex", "STRENGTH", "القوة"),
    (6.5, 10.0, "portrait", "FOCUS", "التركيز"),
    (10.0, 13.5, "punch", "POWER", "الطاقة"),
    (13.5, 17.0, "back", "DISCIPLINE", "الانضباط"),
]
OUTRO = 17.0


# ---------------------------------------------------------------- helpers
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, a, b):
    return clamp((t - a) / (b - a))


def ease_out(x):
    return 1 - (1 - x) ** 3


def ease_in(x):
    return x ** 3


def ease_back(x, s=1.7):
    x -= 1
    return x * x * ((s + 1) * x + s) + 1


_fonts = {}


def font(size, path=FONT_B):
    key = (size, path)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    return _fonts[key]


def text_img(s, size, fill, path=FONT_B, stroke=0, stroke_fill=None, direction=None):
    f = font(size, path)
    kw = {"direction": direction} if direction else {}
    l, t, r, b = f.getbbox(s, stroke_width=stroke, **kw)
    im = Image.new("RGBA", (r - l + 4, b - t + 4), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((2 - l, 2 - t), s, font=f, fill=fill,
                            stroke_width=stroke, stroke_fill=stroke_fill, **kw)
    return im


def fade(im, a):
    if a >= 1:
        return im
    im = im.copy()
    im.putalpha(im.getchannel("A").point(lambda v: int(v * a)))
    return im


def paste(dst, im, cx, cy, a=1.0):
    if a <= 0:
        return
    im = fade(im, a)
    dst.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


# ---------------------------------------------------------------- assets
class Assets:
    def __init__(self, path):
        src = Image.open(path).convert("RGB")
        self.src = src
        self.panels = {k: src.crop(box) for k, (box, _) in PANELS.items()}
        self.bgs = {}
        for k, p in self.panels.items():
            s = max(W / p.width, H / p.height) * 1.15
            bg = p.resize((int(p.width * s), int(p.height * s)), Image.BILINEAR)
            bg = bg.filter(ImageFilter.GaussianBlur(28))
            bg = Image.blend(bg, Image.new("RGB", bg.size, BLACK), 0.62)
            self.bgs[k] = bg
        grad = np.linspace(0, 1, H)[:, None] ** 2.2
        a = (np.clip(grad * 1.3 - 0.15, 0, 1) * 235).astype(np.uint8)
        g = np.zeros((H, W, 4), np.uint8)
        g[..., 3] = np.repeat(a, W, 1)
        self.vignette = Image.fromarray(g, "RGBA")
        rng = np.random.default_rng(7)
        self.particles = rng.random((70, 4))


A = None


def init(path):
    global A
    A = Assets(path)


# ---------------------------------------------------------------- layers
def draw_bg(frame, key, t, t0):
    bg = A.bgs[key]
    drift = (t - t0) * 18
    x = int((bg.width - W) / 2 + math.sin(t * 0.5) * 30)
    y = int((bg.height - H) / 2 - drift)
    y = max(0, min(bg.height - H, y))
    frame.paste(bg.crop((x, y, x + W, y + H)), (0, 0))


def draw_stripes(frame, t, alpha=60):
    d = ImageDraw.Draw(frame, "RGBA")
    off = (t * 120) % 80
    for i in range(-4, 8):
        x = i * 80 + off
        d.polygon([(x, 0), (x + 36, 0), (x - 180, 216), (x - 216, 216)],
                  fill=YELLOW + (alpha,))
    for i in range(-4, 8):
        x = W - i * 80 - off
        d.polygon([(x, H), (x - 36, H), (x + 180, H - 216), (x + 216, H - 216)],
                  fill=YELLOW + (alpha,))


def draw_particles(frame, t):
    d = ImageDraw.Draw(frame, "RGBA")
    for px, py, sp, sz in A.particles:
        y = (py * H - t * (60 + sp * 160)) % H
        x = px * W + math.sin(t * 1.3 + py * 9) * 20
        r = 1.5 + sz * 3.5
        a = int(80 + 120 * sz)
        col = YELLOW if sp > 0.3 else WHITE
        d.ellipse((x - r, y - r, x + r, y + r), fill=col + (a,))


def draw_ghost_word(frame, word, t):
    im = text_img((word + "  ") * 3, 300, (255, 255, 255, 0), stroke=3,
                  stroke_fill=(255, 255, 255, 40))
    x = -((t * 140) % (im.width / 3))
    frame.alpha_composite(im, (int(x), 120))


def make_card(key, zoom, tw):
    """Crop around the panel's focus point and scale to width tw (pure resample)."""
    p = A.panels[key]
    fx, fy = PANELS[key][1]
    cw, ch = p.width / zoom, p.height / zoom
    x0 = clamp(fx * p.width - cw / 2, 0, p.width - cw)
    y0 = clamp(fy * p.height - ch / 2, 0, p.height - ch)
    th = int(tw * p.height / p.width)
    img = p.resize((tw, th), Image.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch))
    return img


def framed(img, border=8, shadow=26):
    w, h = img.size
    out = Image.new("RGBA", (w + border * 2 + shadow, h + border * 2 + shadow), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle((shadow, shadow, out.width - 1, out.height - 1), fill=YELLOW)
    d.rectangle((0, 0, w + border * 2 - 1, h + border * 2 - 1), fill=WHITE)
    out.paste(img, (border, border))
    # corner brackets
    L, c = 60, BLACK
    for (x, y, sx, sy) in [(border, border, 1, 1), (border + w - 1, border, -1, 1),
                           (border, border + h - 1, 1, -1), (border + w - 1, border + h - 1, -1, -1)]:
        d.line((x + 14 * sx, y + 14 * sy, x + (14 + L) * sx, y + 14 * sy), fill=YELLOW, width=6)
        d.line((x + 14 * sx, y + 14 * sy, x + 14 * sx, y + (14 + L) * sy), fill=YELLOW, width=6)
    return out


def draw_hud(frame, t, idx):
    d = ImageDraw.Draw(frame, "RGBA")
    a = clamp(prog(t, 2.3, 2.9))
    if a <= 0:
        return
    al = int(255 * a)
    d.rectangle((40, 76, 330, 166), fill=BLACK + (int(210 * a),))
    d.rectangle((60, 92, 74, 150), fill=YELLOW + (al,))
    paste(frame, text_img("YAZEED", 44, WHITE + (al,)), 92 + 95, 108)
    paste(frame, text_img("TRAINING DAY", 22, YELLOW + (al,)), 92 + 85, 140)
    if idx is not None:
        paste(frame, text_img(f"0{idx + 1}", 56, YELLOW + (al,)), W - 150, 120)
        paste(frame, text_img("/04", 28, WHITE + (al,)), W - 85, 132)
    # progress bar
    d.rectangle((60, H - 70, W - 60, H - 64), fill=(255, 255, 255, int(50 * a)))
    d.rectangle((60, H - 70, 60 + (W - 120) * t / DUR, H - 64), fill=YELLOW + (al,))
    # REC dot
    if int(t * 2) % 2 == 0:
        d.ellipse((W - 92, H - 118, W - 72, H - 98), fill=RED + (al,))
    paste(frame, text_img("REC", 22, WHITE + (al,)), W - 125, H - 108)


def draw_title(frame, word, ar, t, t0, cy):
    lt = t - t0
    # English word, letter by letter drop-in
    f = font(150 if len(word) <= 6 else 118)
    total = f.getlength(word) + 6 * (len(word) - 1)
    x = (W - total) / 2
    for i, ch in enumerate(word):
        p = prog(lt, 0.35 + i * 0.05, 0.75 + i * 0.05)
        if p > 0:
            im = text_img(ch, f.size, WHITE, stroke=0)
            dy = (1 - ease_back(p)) * -90
            cx = x + f.getlength(ch) / 2
            paste(frame, im, cx, cy + dy, a=min(1, p * 2))
        x += f.getlength(ch) + 6
    # yellow bar with Arabic word, wipes in right->left
    p = ease_out(prog(lt, 0.9, 1.4))
    if p > 0:
        ar_im = text_img(ar, 84, BLACK, direction="rtl")
        bw, bh = ar_im.width + 90, ar_im.height + 44
        bar = Image.new("RGBA", (bw, bh), YELLOW + (255,))
        bar.alpha_composite(ar_im, ((bw - ar_im.width) // 2, (bh - ar_im.height) // 2))
        vis = int(bw * p)
        bar = bar.crop((bw - vis, 0, bw, bh)) if vis > 0 else None
        if bar:
            frame.alpha_composite(bar, (int((W + bw) / 2 - vis), int(cy + 115)))
    # underline accent
    p2 = ease_out(prog(lt, 1.2, 1.7))
    if p2 > 0:
        d = ImageDraw.Draw(frame)
        d.rectangle((W / 2 - 160 * p2, cy + 115 + 170, W / 2 + 160 * p2, cy + 115 + 176), fill=RED)


def glitch(frame, amt):
    if amt <= 0:
        return frame
    arr = np.asarray(frame.convert("RGB")).copy()
    s = int(amt * 24)
    arr[..., 0] = np.roll(arr[..., 0], s, axis=1)
    arr[..., 2] = np.roll(arr[..., 2], -s, axis=1)
    rng = np.random.default_rng(int(amt * 1000))
    for _ in range(int(amt * 10)):
        y = rng.integers(0, H - 40)
        h = rng.integers(6, 40)
        arr[y:y + h] = np.roll(arr[y:y + h], rng.integers(-60, 60), axis=1)
    return Image.fromarray(arr).convert("RGBA")


def zoom_blur(frame, s, blur):
    if s != 1:
        w, h = int(W * s), int(H * s)
        big = frame.resize((w, h), Image.BILINEAR)
        frame = big.crop(((w - W) // 2, (h - H) // 2, (w - W) // 2 + W, (h - H) // 2 + H))
    if blur > 0.5:
        frame = frame.filter(ImageFilter.GaussianBlur(blur))
    return frame


# ---------------------------------------------------------------- scenes
def intro(t):
    frame = Image.new("RGBA", (W, H), BLACK + (255,))
    d = ImageDraw.Draw(frame)
    # checker tiles pop in (echo of the gym floor)
    S = 135
    for gy in range(H // S + 1):
        for gx in range(W // S + 1):
            if (gx + gy) % 2:
                continue
            delay = (gx + gy) * 0.035
            p = ease_out(prog(t, delay, delay + 0.35)) * (1 - ease_in(prog(t, 1.6, 2.4)))
            if p <= 0:
                continue
            r = S / 2 * p
            cx, cy = gx * S + S / 2, gy * S + S / 2
            d.rectangle((cx - r, cy - r, cx + r, cy + r), fill=(28, 26, 8))
    draw_particles(frame, t)
    p = ease_back(prog(t, 0.3, 0.9))
    if p > 0:
        im = text_img("YAZEED", 210, WHITE)
        im = im.resize((max(1, int(im.width * p)), max(1, int(im.height * p))), Image.BICUBIC)
        paste(frame, im, W / 2, 820)
    p = ease_out(prog(t, 0.7, 1.1))
    if p > 0:
        ar = text_img("يزيد", 120, YELLOW, direction="rtl")
        paste(frame, ar, W / 2, 1010 + (1 - p) * 60, a=p)
    p = ease_out(prog(t, 1.0, 1.4))
    if p > 0:
        d.rectangle((W / 2 - 260 * p, 1110, W / 2 + 260 * p, 1170), fill=YELLOW)
        if p > 0.6:
            paste(frame, text_img("BEAST  MODE", 40, BLACK), W / 2, 1140, a=(p - 0.6) / 0.4)
    g = 0
    for gt in (0.3, 0.9, 1.5):
        g = max(g, 1 - prog(t, gt, gt + 0.15))
    if t < 0.3:
        g = 0
    frame = glitch(frame, g * 0.8)
    return frame


def scene(t, idx):
    t0, t1, key, word, ar = SCENES[idx]
    lt = t - t0
    frame = Image.new("RGBA", (W, H), BLACK + (255,))
    draw_bg(frame, key, t, t0)
    draw_ghost_word(frame, word, t)
    draw_stripes(frame, t, 45)
    frame.alpha_composite(A.vignette)
    draw_particles(frame, t)

    p = A.panels[key]
    tw = 1000 if p.width > p.height else min(860, int(980 * p.width / p.height))
    zoom = 1.0 + 0.14 * (lt / (t1 - t0))
    card = framed(make_card(key, zoom, tw))
    ep = ease_back(prog(lt, 0.0, 0.55), 1.4)
    sc = 0.75 + 0.25 * ep
    # beat pulse
    beat_ph = (t % BEAT) / BEAT
    sc *= 1 + 0.012 * math.exp(-beat_ph * 8)
    card = card.resize((int(card.width * sc), int(card.height * sc)), Image.BICUBIC)
    rot = (1 - ep) * (-6 if idx % 2 else 6) + math.sin(lt * 1.2) * 0.6
    card = card.rotate(rot, Image.BICUBIC, expand=True)
    cy = 820 if p.width > p.height else 800
    cx = W / 2 + (1 - ease_out(prog(lt, 0, 0.5))) * (W if idx % 2 else -W) * 0.6
    sx = sy = 0
    if key == "punch":  # punch impact shake
        sh = math.exp(-max(0, lt - 1.0) * 6) if lt > 1.0 else 0
        sx = math.sin(lt * 90) * 22 * sh
        sy = math.cos(lt * 70) * 14 * sh
    paste(frame, card, cx + sx, cy + sy, a=clamp(ep * 1.5))

    title_y = 1380 if p.width > p.height else 1440
    if p.width > p.height:
        title_y = 1300
    draw_title(frame, word, ar, t, t0, title_y)
    draw_hud(frame, t, idx)

    # transitions: flash + zoom-in at start, whip zoom out at end
    s_in = 1 + 0.25 * (1 - ease_out(prog(lt, 0, 0.3)))
    b_in = 14 * (1 - prog(lt, 0, 0.25))
    s_out = 1 + 0.35 * ease_in(prog(t, t1 - 0.25, t1))
    b_out = 16 * ease_in(prog(t, t1 - 0.25, t1))
    frame = zoom_blur(frame, s_in * s_out, max(b_in, b_out))
    fl = 1 - prog(lt, 0, 0.18)
    if fl > 0:
        frame.alpha_composite(Image.new("RGBA", (W, H), WHITE + (int(200 * fl),)))
    frame = glitch(frame, 0.6 * (1 - prog(lt, 0.05, 0.25)) if lt > 0.05 else 0)
    return frame


def outro(t):
    lt = t - OUTRO
    frame = Image.new("RGBA", (W, H), BLACK + (255,))
    draw_bg(frame, "flex", t, OUTRO)
    draw_stripes(frame, t, 55)
    frame.alpha_composite(A.vignette)
    draw_particles(frame, t)
    # rebuild the full collage panel by panel at width 1000
    s = 1000 / A.src.width
    ox, oy = (W - 1000) / 2, 560
    dirs = {"flex": (-1, 0), "portrait": (1, 0), "punch": (0, 1), "back": (1, 1)}
    bg = Image.new("RGBA", (1000 + 16, int(A.src.height * s) + 16), WHITE + (255,))
    pa = ease_out(prog(lt, 0.0, 0.3))
    paste(frame, bg, W / 2, oy + bg.height / 2 - 8, a=pa)
    for i, (k, ((x0, y0, x1, y1), _)) in enumerate(PANELS.items()):
        p = ease_back(prog(lt, 0.15 + i * 0.12, 0.6 + i * 0.12), 1.2)
        if p <= 0:
            continue
        im = A.panels[k].resize((int((x1 - x0) * s), int((y1 - y0) * s)), Image.LANCZOS).convert("RGBA")
        dx, dy = dirs[k]
        px = ox + x0 * s + im.width / 2 + dx * (1 - p) * 700
        py = oy + y0 * s + im.height / 2 + dy * (1 - p) * 900
        paste(frame, im, px, py, a=clamp(p * 2))
    # tagline
    p = ease_back(prog(lt, 0.9, 1.4))
    if p > 0:
        im = text_img("NO LIMITS", 150, WHITE)
        im = im.resize((max(1, int(im.width * p)), max(1, int(im.height * p))), Image.BICUBIC)
        paste(frame, im, W / 2, 360)
    p = ease_out(prog(lt, 1.3, 1.8))
    if p > 0:
        ar = text_img("بلا حدود", 96, BLACK, direction="rtl")
        bw = ar.width + 100
        bar = Image.new("RGBA", (bw, ar.height + 40), YELLOW + (255,))
        bar.alpha_composite(ar, (50, 20))
        vis = max(1, int(bw * p))
        frame.alpha_composite(bar.crop((bw - vis, 0, bw, bar.height)), (int((W + bw) / 2 - vis), 1220))
    p = ease_out(prog(lt, 1.8, 2.3))
    if p > 0:
        paste(frame, text_img("YAZEED", 110, YELLOW), W / 2, 1480 + (1 - p) * 40, a=p)
        paste(frame, text_img("TRAIN  •  FIGHT  •  REPEAT", 36, WHITE), W / 2, 1580, a=p)
    draw_hud(frame, t, None)
    fl = 1 - prog(lt, 0, 0.18)
    if fl > 0:
        frame.alpha_composite(Image.new("RGBA", (W, H), WHITE + (int(200 * fl),)))
    # final fade to black
    fo = prog(t, DUR - 0.6, DUR)
    if fo > 0:
        frame.alpha_composite(Image.new("RGBA", (W, H), BLACK + (int(255 * fo),)))
    return frame


def render(i):
    t = i / FPS
    if t < SCENES[0][0]:
        f = intro(t)
    elif t >= OUTRO:
        f = outro(t)
    else:
        idx = next(k for k, s in enumerate(SCENES) if s[0] <= t < s[1])
        f = scene(t, idx)
    return f.convert("RGB").tobytes()


# ---------------------------------------------------------------- audio
def make_audio(path, sr=44100):
    n = int(DUR * sr)
    out = np.zeros(n)
    rng = np.random.default_rng(1)

    def add(sig, t):
        s = int(t * sr)
        e = min(n, s + len(sig))
        if s < n:
            out[s:e] += sig[: e - s]

    tt = np.arange(int(0.45 * sr)) / sr
    kick = np.sin(2 * np.pi * np.cumsum(45 + 110 * np.exp(-tt * 30)) / sr) * np.exp(-tt * 7)
    th = np.arange(int(0.06 * sr)) / sr
    hat = np.diff(rng.standard_normal(len(th) + 1)) * np.exp(-th * 70) * 0.18
    snare_t = np.arange(int(0.25 * sr)) / sr
    snare = (rng.standard_normal(len(snare_t)) * 0.35 + np.sin(2 * np.pi * 190 * snare_t) * 0.4) * np.exp(-snare_t * 18)
    bass_notes = [55, 55, 65.4, 49]
    t = 2.5
    bar = 0
    while t < DUR - 0.6:
        add(kick * 0.9, t)
        add(hat, t + BEAT / 2)
        if bar % 2 == 1:
            add(snare * 0.5, t)
        f = bass_notes[(bar // 4) % 4]
        bt = np.arange(int(BEAT * sr)) / sr
        add(np.tanh(2.5 * np.sin(2 * np.pi * f * bt)) * np.exp(-bt * 3) * 0.22, t)
        t += BEAT
        bar += 1
    # intro: riser
    rt = np.arange(int(2.5 * sr)) / sr
    riser = rng.standard_normal(len(rt)) * (rt / 2.5) ** 2 * 0.25
    riser += np.sin(2 * np.pi * np.cumsum(200 + 900 * (rt / 2.5) ** 2) / sr) * (rt / 2.5) ** 2 * 0.15
    add(riser, 0)
    # impacts + whooshes at cuts
    it = np.arange(int(1.2 * sr)) / sr
    boom = np.sin(2 * np.pi * np.cumsum(30 + 60 * np.exp(-it * 8)) / sr) * np.exp(-it * 3.5) * 0.9
    boom += rng.standard_normal(len(it)) * np.exp(-it * 25) * 0.4
    wt = np.arange(int(0.4 * sr)) / sr
    whoosh = rng.standard_normal(len(wt))
    whoosh = np.convolve(whoosh, np.ones(30) / 30, "same") * np.sin(np.pi * wt / 0.4) ** 2 * 1.2
    for c in [s[0] for s in SCENES] + [OUTRO]:
        add(whoosh, c - 0.38)
        add(boom, c)
    add(boom * 0.8, SCENES[2][0] + 1.0)  # punch hit
    fade_n = int(0.8 * sr)
    out[-fade_n:] *= np.linspace(1, 0, fade_n)
    out = np.tanh(out * 1.2)
    out /= np.abs(out).max() + 1e-9
    pcm = (out * 0.89 * 32767).astype(np.int16)
    st = np.stack([pcm, pcm], 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(st.tobytes())


def main():
    src, out = sys.argv[1], sys.argv[2]
    wav = os.path.splitext(out)[0] + "_audio.wav"
    make_audio(wav)
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-movflags", "+faststart", out,
    ], stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init, initargs=(src,)) as pool:
        for i, buf in enumerate(pool.imap(render, range(N), chunksize=4)):
            ff.stdin.write(buf)
            if i % 60 == 0:
                print(f"frame {i}/{N}", flush=True)
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    print("done:", out)


if __name__ == "__main__":
    main()
