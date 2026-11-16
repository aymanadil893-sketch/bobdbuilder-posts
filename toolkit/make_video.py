#!/usr/bin/env python3
"""BobDBuilder video maker.
Usage: python3 make_video.py script.json out.mp4 [--models DIR]

script.json:
{
  "handle": "@bobdbuildergaming",
  "voice": "am_michael", "speed": 1.15,
  "beats": [
    {"say": "spoken line", "caption_key": "word to colour",
     "pose": "stand|cheer|point|shrug|shock|think",
     "visual": {...}}
  ]
}
visual types:
  {"type":"title","text":"GTA 6","sub":"WHY THE HYPE?"}
  {"type":"counter","value":230000000,"label":"COPIES SOLD","suffix":""}
  {"type":"bars","items":[["GTA 6 TRAILER",475,"475M"],["MARVEL TRAILER",365,"365M"]],"title":"VIEWS IN 24H"}
  {"type":"icon","icon":"calendar|trophy|play|check|x|clock|controller|globe|fire|chart|lock|map|warning|money","label":"NOV 19","sub":"PS5 • XBOX"}
  {"type":"compare","left":"2013","right":"2026","label":"13 YEARS"}
  {"type":"list","title":"WHAT'S NEW","items":["VICE CITY","2 HEROES","LEONIDA"]}
  {"type":"stamp","text":"DELAYED","sub":"TWICE"}
  {"type":"bubble","text":"Are you buying it day one?"}
  {"type":"follow"}   (end card with animated tap on Follow)
Every beat is one voiceover phrase (aim for 2-4 s each).
"""
import json, math, os, subprocess, sys, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, SR = 1080, 1920, 30, 24000
NAVY = (14, 6, 30); DEEP = (30, 10, 54); PURPLE = (96, 40, 146); MAGENTA = (226, 54, 150)
PINK = (255, 72, 140); ORANGE = (255, 140, 66); PEACH = (255, 200, 125)
TEAL = (46, 214, 214); WHITE = (255, 248, 252); GREY = (170, 150, 190); YELLOW = (255, 214, 64)

def find_font(names):
    dirs = ["/usr/share/fonts/opentype/inter/", "/usr/share/fonts/truetype/google-fonts/",
            "/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/"]
    for n in names:
        for d in dirs:
            p = os.path.join(d, n)
            if os.path.exists(p): return p
    out = subprocess.run(["fc-match", "-f", "%{file}", "sans:bold"], capture_output=True, text=True).stdout
    return out or None
F_BLACK = find_font(["InterDisplay-Black.otf", "Inter-Black.otf", "Poppins-Bold.ttf", "DejaVuSans-Bold.ttf"])
F_BOLD = find_font(["InterDisplay-Bold.otf", "Inter-Bold.otf", "Poppins-Bold.ttf", "DejaVuSans-Bold.ttf"])
F_CAP = find_font(["Poppins-Bold.ttf", "InterDisplay-ExtraBold.otf", "DejaVuSans-Bold.ttf"])
_fc = {}
def font(path, size):
    k = (path, size)
    if k not in _fc: _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def lerp(a, b, t): return a + (b - a) * t
def ease(t): t = clamp(t); return 1 - (1 - t) ** 3
def back(t):
    t = clamp(t); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
def mix(a, b, t): return tuple(int(lerp(x, y, t)) for x, y in zip(a, b))

# ---------------- audio ----------------
def synth_voice(beats, voice, speed, models):
    from kokoro_onnx import Kokoro
    k = Kokoro(os.path.join(models, "kokoro.onnx"), os.path.join(models, "voices-v1.0.bin"))
    out, t = [], 0.0
    for i, b in enumerate(beats):
        if i: out.append(np.zeros(int(SR * 0.14), np.float32)); t += 0.14
        a, _ = k.create(b["say"], voice=voice, speed=speed, lang="en-us")
        a = a.astype(np.float32)
        idx = np.where(np.abs(a) > 0.01)[0]
        if len(idx): a = a[max(0, idx[0] - 600): idx[-1] + 1800]
        b["t0"], b["t1"] = t, t + len(a) / SR
        out.append(a); t += len(a) / SR
    v = np.concatenate(out); v = v / (np.abs(v).max() + 1e-6) * 0.85
    return v, t

def music_bed(total, cut_times):
    n = int(SR * total); t = np.arange(n) / SR
    bpm = 100; beat = 60 / bpm
    # soft synth pad: minor-ish chord progression
    chords = [[220.0, 261.6, 329.6], [196.0, 246.9, 293.7], [174.6, 220.0, 261.6], [196.0, 246.9, 311.1]]
    pad = np.zeros(n, np.float32)
    bar = beat * 4
    for ci in range(int(total / bar) + 1):
        s = int(ci * bar * SR); e = min(n, int((ci + 1) * bar * SR))
        if s >= n: break
        tt = t[s:e]
        env = np.minimum(1, (tt - tt[0]) / 0.4) * np.minimum(1, (tt[-1] - tt + 1e-3) / 0.4)
        for f in chords[ci % 4]:
            pad[s:e] += (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * f * 2 * tt)) * env
    pad *= 0.018
    # kick + hat
    drums = np.zeros(n, np.float32)
    kl = int(0.18 * SR); kt = np.arange(kl) / SR
    kick = np.sin(2 * np.pi * (55 + 90 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 14)
    hl = int(0.05 * SR); hat = np.random.randn(hl) * np.exp(-np.arange(hl) / SR * 80)
    i = 0
    while i * beat < total:
        s = int(i * beat * SR)
        e = min(n, s + kl); drums[s:e] += kick[:e - s] * 0.22
        hs = int((i + 0.5) * beat * SR); he = min(n, hs + hl)
        if hs < n: drums[hs:he] += hat[:he - hs] * 0.035
        i += 1
    # whoosh on each cut
    fx = np.zeros(n, np.float32)
    wl = int(0.35 * SR); wt = np.arange(wl) / SR
    nz = np.random.randn(wl)
    whoosh = np.convolve(nz, np.ones(30) / 30, "same") * np.sin(np.pi * wt / wt[-1]) ** 2 * 0.18
    pl = int(0.06 * SR); pt = np.arange(pl) / SR
    pop = np.sin(2 * np.pi * 900 * pt) * np.exp(-pt * 70) * 0.12
    for c in cut_times:
        s = int(max(0, c - 0.18) * SR); e = min(n, s + wl); fx[s:e] += whoosh[:e - s]
        s2 = int((c + 0.12) * SR); e2 = min(n, s2 + pl)
        if s2 < n: fx[s2:e2] += pop[:e2 - s2]
    return pad + drums + fx

# ---------------- drawing helpers ----------------
def text_img(txt, f, fill, stroke=0, stroke_fill=(0, 0, 0)):
    bb = f.getbbox(txt, stroke_width=stroke)
    im = Image.new("RGBA", (bb[2] - bb[0] + 40, bb[3] - bb[1] + 40), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((20 - bb[0], 20 - bb[1]), txt, font=f, fill=fill,
                            stroke_width=stroke, stroke_fill=stroke_fill)
    return im

def grad_text(txt, f, c1, c2):
    m = text_img(txt, f, (255, 255, 255, 255))
    g = Image.new("RGBA", m.size)
    a = np.linspace(0, 1, m.size[0])[None, :, None]
    arr = (np.array(c1)[None, None, :] * (1 - a) + np.array(c2)[None, None, :] * a)
    arr = np.repeat(arr, m.size[1], 0).astype(np.uint8)
    g = Image.fromarray(np.dstack([arr, np.array(m.split()[3])]), "RGBA")
    return g

def fit(txt, path, size, maxw):
    while size > 30 and font(path, size).getlength(txt) > maxw: size -= 6
    return font(path, size)

def glow(im, radius=18, strength=1.0):
    g = im.filter(ImageFilter.GaussianBlur(radius))
    if strength != 1.0:
        a = g.split()[3].point(lambda v: int(min(255, v * strength))); g.putalpha(a)
    out = Image.new("RGBA", im.size); out.alpha_composite(g); out.alpha_composite(im)
    return out

def place(frame, im, cx, cy, alpha=1.0, scale=1.0, rot=0):
    if alpha <= 0.01 or scale <= 0.01: return
    if rot: im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    if abs(scale - 1) > 0.003:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    if alpha < 0.999:
        im = im.copy(); im.putalpha(im.split()[3].point(lambda v: int(v * alpha)))
    frame.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

def layer(): return Image.new("RGBA", (W, H), (0, 0, 0, 0))

# ---------------- icons (simple original line art) ----------------
def icon(name, size=300, col=WHITE, acc=PINK):
    s = size; im = Image.new("RGBA", (s + 40, s + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    o = 20; lw = max(6, s // 28); c = col + (255,); a = acc + (255,)
    if name == "calendar":
        d.rounded_rectangle([o, o + s * .12, o + s, o + s], 28, outline=c, width=lw)
        d.rectangle([o, o + s * .12, o + s, o + s * .32], fill=a)
        for x in (.28, .72): d.line([o + s * x, o, o + s * x, o + s * .2], fill=c, width=lw)
    elif name == "trophy":
        d.arc([o + s * .2, o, o + s * .8, o + s * .65], 0, 180, fill=c, width=lw)
        d.line([o + s * .2, o + s * .02, o + s * .8, o + s * .02], fill=c, width=lw)
        d.arc([o, o + s * .05, o + s * .3, o + s * .35], 90, 270, fill=a, width=lw)
        d.arc([o + s * .7, o + s * .05, o + s, o + s * .35], 270, 90, fill=a, width=lw)
        d.line([o + s * .5, o + s * .65, o + s * .5, o + s * .85], fill=c, width=lw)
        d.rounded_rectangle([o + s * .28, o + s * .85, o + s * .72, o + s], 10, outline=c, width=lw)
    elif name == "play":
        d.ellipse([o, o, o + s, o + s], outline=c, width=lw)
        d.polygon([(o + s * .4, o + s * .28), (o + s * .4, o + s * .72), (o + s * .74, o + s * .5)], fill=a)
    elif name == "check":
        d.ellipse([o, o, o + s, o + s], outline=c, width=lw)
        d.line([(o + s * .27, o + s * .52), (o + s * .44, o + s * .68), (o + s * .75, o + s * .34)], fill=(46, 214, 140, 255), width=lw * 2, joint="curve")
    elif name == "x":
        d.ellipse([o, o, o + s, o + s], outline=c, width=lw)
        for p in [((.3, .3), (.7, .7)), ((.7, .3), (.3, .7))]:
            d.line([(o + s * p[0][0], o + s * p[0][1]), (o + s * p[1][0], o + s * p[1][1])], fill=a, width=lw * 2)
    elif name == "clock":
        d.ellipse([o, o, o + s, o + s], outline=c, width=lw)
        d.line([o + s / 2, o + s / 2, o + s / 2, o + s * .2], fill=c, width=lw)
        d.line([o + s / 2, o + s / 2, o + s * .72, o + s * .6], fill=a, width=lw)
    elif name == "controller":
        d.rounded_rectangle([o, o + s * .25, o + s, o + s * .78], int(s * .25), outline=c, width=lw)
        d.line([o + s * .2, o + s * .51, o + s * .38, o + s * .51], fill=c, width=lw)
        d.line([o + s * .29, o + s * .42, o + s * .29, o + s * .6], fill=c, width=lw)
        for (x, y, k) in [(.7, .44, a), (.8, .56, (46, 214, 214, 255))]:
            d.ellipse([o + s * x - s * .05, o + s * y - s * .05, o + s * x + s * .05, o + s * y + s * .05], fill=k)
    elif name == "globe":
        d.ellipse([o, o, o + s, o + s], outline=c, width=lw)
        d.ellipse([o + s * .28, o, o + s * .72, o + s], outline=c, width=lw // 1)
        d.line([o, o + s / 2, o + s, o + s / 2], fill=a, width=lw)
    elif name == "fire":
        d.polygon([(o + s * .5, o), (o + s * .85, o + s * .55), (o + s * .72, o + s), (o + s * .28, o + s), (o + s * .15, o + s * .55)], outline=c, fill=None, width=lw)
        d.polygon([(o + s * .5, o + s * .42), (o + s * .66, o + s * .72), (o + s * .58, o + s * .92), (o + s * .42, o + s * .92), (o + s * .34, o + s * .72)], fill=a)
    elif name == "chart":
        d.line([o, o, o, o + s, o + s, o + s], fill=c, width=lw)
        d.line([(o + s * .1, o + s * .85), (o + s * .4, o + s * .6), (o + s * .6, o + s * .7), (o + s * .95, o + s * .15)], fill=a, width=lw, joint="curve")
    elif name == "lock":
        d.arc([o + s * .25, o, o + s * .75, o + s * .55], 180, 360, fill=c, width=lw)
        d.line([o + s * .25, o + s * .28, o + s * .25, o + s * .45], fill=c, width=lw)
        d.line([o + s * .75, o + s * .28, o + s * .75, o + s * .45], fill=c, width=lw)
        d.rounded_rectangle([o + s * .12, o + s * .45, o + s * .88, o + s], 24, outline=c, width=lw)
        d.ellipse([o + s * .44, o + s * .64, o + s * .56, o + s * .76], fill=a)
    elif name == "map":
        pts = [(o, o + s * .1), (o + s * .33, o), (o + s * .66, o + s * .1), (o + s, o), (o + s, o + s * .9), (o + s * .66, o + s), (o + s * .33, o + s * .9), (o, o + s)]
        d.polygon(pts, outline=c, width=lw)
        d.line([o + s * .33, o, o + s * .33, o + s * .9], fill=c, width=lw // 2)
        d.line([o + s * .66, o + s * .1, o + s * .66, o + s], fill=c, width=lw // 2)
        d.ellipse([o + s * .45, o + s * .35, o + s * .6, o + s * .5], fill=a)
    elif name == "warning":
        d.polygon([(o + s * .5, o), (o + s, o + s), (o, o + s)], outline=c, width=lw)
        d.line([o + s * .5, o + s * .35, o + s * .5, o + s * .7], fill=a, width=lw * 2)
        d.ellipse([o + s * .46, o + s * .8, o + s * .54, o + s * .88], fill=a)
    elif name == "money":
        d.rounded_rectangle([o, o + s * .2, o + s, o + s * .8], 24, outline=c, width=lw)
        f = font(F_BLACK, int(s * .45)); d.text((o + s / 2, o + s / 2), "$", font=f, fill=a, anchor="mm")
    else:
        d.ellipse([o, o, o + s, o + s], outline=c, width=lw)
    return glow(im, 14, 0.9)

# ---------------- character (original: glowing figure with hard hat) ----------------
def character(pose, t, size=1.0):
    im = Image.new("RGBA", (420, 520), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = WHITE + (255,); lw = 12; cx = 210
    bob = math.sin(t * 3.2) * 4
    hy = 120 + bob
    d.ellipse([cx - 52, hy - 52, cx + 52, hy + 52], outline=c, width=lw)
    # hard hat
    d.chord([cx - 62, hy - 78, cx + 62, hy + 30], 180, 360, fill=YELLOW + (255,))
    d.rounded_rectangle([cx - 78, hy - 30, cx + 78, hy - 14], 8, fill=YELLOW + (255,))
    d.rectangle([cx - 8, hy - 80, cx + 8, hy - 30], fill=(230, 180, 30, 255))
    # eyes (blink)
    blink = (t % 3.3) < 0.12
    for ex in (-18, 18):
        if blink: d.line([cx + ex - 8, hy + 8, cx + ex + 8, hy + 8], fill=c, width=6)
        else: d.ellipse([cx + ex - 7, hy + 1, cx + ex + 7, hy + 15], fill=c)
    if pose == "shock":
        d.ellipse([cx - 10, hy + 24, cx + 10, hy + 42], outline=c, width=5)
    elif pose in ("cheer", "point", "stand"):
        d.arc([cx - 20, hy + 14, cx + 20, hy + 38], 20, 160, fill=c, width=5)
    else:
        d.line([cx - 14, hy + 32, cx + 14, hy + 30], fill=c, width=5)
    neck = hy + 52; hip = neck + 170
    d.line([cx, neck, cx, hip], fill=c, width=lw)
    sh = neck + 30
    wav = math.sin(t * 9) * 12
    arms = {
        "stand": [(-70, 110), (70, 110)],
        "cheer": [(-90, -110 + wav), (90, -110 - wav)],
        "point": [(-60, 110), (150, -40)],
        "shrug": [(-110, -20), (110, -20)],
        "shock": [(-100, -90), (100, -90)],
        "think": [(-60, 110), (40, -50)],
    }.get(pose, [(-70, 110), (70, 110)])
    for ax, ay in arms:
        mx, my = cx + ax * 0.55, sh + ay * 0.45 + 18
        d.line([(cx, sh), (mx, my), (cx + ax, sh + ay)], fill=c, width=lw, joint="curve")
    for lx in (-55, 55):
        d.line([(cx, hip), (cx + lx * 0.6, hip + 80), (cx + lx, hip + 165)], fill=c, width=lw, joint="curve")
    im = glow(im, 16, 1.2)
    if size != 1.0: im = im.resize((int(im.width * size), int(im.height * size)), Image.BILINEAR)
    return im

# ---------------- background ----------------
def make_bg():
    y = np.linspace(0, 1, H)[:, None]
    top = np.array(NAVY); mid = np.array(DEEP); bot = np.array((70, 22, 90))
    col = np.where(y < 0.6, top + (mid - top) * (y / 0.6), mid + (bot - mid) * ((y - 0.6) / 0.4))
    arr = np.repeat(col[:, None, :], W, 1).reshape(H, W, 3)
    bg = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(bg)
    for gy in range(40, H, 54):
        for gx in range(30, W, 54):
            d.point((gx, gy), fill=(120, 80, 160, 255))
    # horizon glow
    g = layer(); ImageDraw.Draw(g).ellipse([W / 2 - 700, H - 420, W / 2 + 700, H + 300], fill=PINK + (90,))
    bg.alpha_composite(g.filter(ImageFilter.GaussianBlur(140)))
    # floor dashed line
    for x in range(0, W, 36): d.line([x, 1610, x + 18, 1610], fill=(200, 150, 220, 160), width=3)
    return bg

def corner_icon():
    im = Image.new("RGBA", (150, 110), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for yy in range(10, 80):
        half = math.sqrt(max(0, 45 ** 2 - (yy - 55) ** 2))
        if yy > 55: break
        if yy > 32 and (yy % 9) < 3: continue
        d.line([75 - half, yy, 75 + half, yy], fill=mix(PEACH, PINK, (yy - 10) / 45) + (255,))
    d.line([20, 58, 130, 58], fill=PINK + (255,), width=3)
    return glow(im, 8)

# ---------------- visual renderers ----------------
CENTER_Y = 640

def draw_visual(fr, v, lt, dur, enter, key_col):
    """lt = local time in beat, enter = 0..1 entrance progress."""
    typ = v.get("type", "title")
    pop = back(lt / 0.45); fade = ease(lt / 0.25)
    if typ == "title":
        f = fit(v["text"], F_BLACK, 240, 960)
        place(fr, grad_text(v["text"], f, PEACH, PINK), W / 2, CENTER_Y - 60, fade, lerp(0.6, 1, pop))
        if v.get("sub"):
            a = ease((lt - dur * 0.35) / 0.3)
            f2 = fit(v["sub"], F_BLACK, 110, 960)
            place(fr, text_img(v["sub"], f2, WHITE), W / 2, CENTER_Y + 140, a, lerp(0.7, 1, back((lt - dur * 0.35) / 0.4)))
    elif typ == "counter":
        p = ease((lt - 0.2) / max(0.6, dur * 0.7))
        val = v["value"] * p
        txt = f"{int(val):,}{v.get('suffix', '')}"
        f = fit(f"{int(v['value']):,}{v.get('suffix', '')}", F_BLACK, 170, 980)
        place(fr, grad_text(txt, f, PEACH, PINK), W / 2, CENTER_Y - 40, fade, lerp(0.7, 1, pop))
        if p > 0.98:
            place(fr, glow(text_img(txt, f, PINK + (90,)), 30), W / 2, CENTER_Y - 40, 0.6 * (1 - ease((lt - dur * 0.75) / 0.4)))
        a = ease((lt - 0.4) / 0.3)
        place(fr, text_img(v.get("label", ""), fit(v.get("label", ""), F_BOLD, 68, 960), key_col), W / 2, CENTER_Y + 110, a)
    elif typ == "bars":
        items = v["items"]; mx = max(i[1] for i in items)
        if v.get("title"):
            place(fr, text_img(v["title"], fit(v["title"], F_BLACK, 90, 960), WHITE), W / 2, CENTER_Y - 280, fade)
        L = layer(); d = ImageDraw.Draw(L); x0 = 120; bw = 840
        cols = [PINK, (255, 255, 255), TEAL, ORANGE]
        for k, (lab, val, disp) in enumerate(items):
            g = ease((lt - 0.3 - k * 0.35) / 1.0)
            y = CENTER_Y - 120 + k * 200
            d.rounded_rectangle([x0, y, x0 + max(40, bw * val / mx * g), y + 84], 42,
                                fill=cols[k % 4] + ((255 if k == 0 else 140),))
            place(L, text_img(lab, font(F_BOLD, 44), (255, 255, 255, 220)), x0 + font(F_BOLD, 44).getlength(lab) / 2 + 10, y - 40, ease((lt - 0.2) / 0.3))
            if g > 0.1:
                place(L, text_img(disp, font(F_BLACK, 56), NAVY if k else WHITE), x0 + max(40, bw * val / mx * g) - 90, y + 42, g)
        fr.alpha_composite(L)
    elif typ == "icon":
        wob = math.sin(lt * 4) * 3
        place(fr, icon(v.get("icon", "fire"), 300), W / 2, CENTER_Y - 120, fade, lerp(0.3, 1, pop), wob)
        if v.get("label"):
            f = fit(v["label"], F_BLACK, 150, 960)
            place(fr, grad_text(v["label"], f, PEACH, PINK), W / 2, CENTER_Y + 140, ease((lt - 0.35) / 0.3), lerp(0.7, 1, back((lt - 0.35) / 0.4)))
        if v.get("sub"):
            place(fr, text_img(v["sub"], fit(v["sub"], F_BOLD, 62, 960), TEAL), W / 2, CENTER_Y + 270, ease((lt - dur * 0.45) / 0.3))
    elif typ == "compare":
        f = fit(max(v["left"], v["right"], key=len), F_BLACK, 150, 380)
        place(fr, text_img(v["left"], f, (255, 255, 255, 150)), W / 2 - 290, CENTER_Y - 60, fade, lerp(0.6, 1, pop))
        place(fr, text_img("→", f, PEACH), W / 2, CENTER_Y - 60, ease((lt - 0.4) / 0.3))
        place(fr, grad_text(v["right"], f, PEACH, PINK), W / 2 + 290, CENTER_Y - 60, ease((lt - 0.7) / 0.3), lerp(0.6, 1, back((lt - 0.7) / 0.4)))
        if v.get("label"):
            place(fr, text_img(v["label"], fit(v["label"], F_BLACK, 90, 960), WHITE), W / 2, CENTER_Y + 140, ease((lt - dur * 0.45) / 0.3))
    elif typ == "list":
        if v.get("title"):
            place(fr, text_img(v["title"], fit(v["title"], F_BLACK, 84, 960), TEAL), W / 2, CENTER_Y - 300, fade)
        n = len(v["items"]); step = max(0.35, (dur - 0.6) / max(1, n))
        for k, it in enumerate(v["items"]):
            a = ease((lt - 0.3 - k * step) / 0.25)
            card = Image.new("RGBA", (900, 150), (0, 0, 0, 0)); cd = ImageDraw.Draw(card)
            cd.rounded_rectangle([0, 0, 899, 149], 36, fill=(255, 255, 255, 18), outline=PINK + (220,), width=4)
            cd.ellipse([40, 45, 100, 105], fill=PINK + (255,))
            cd.text((70, 75), str(k + 1), font=font(F_BLACK, 40), fill=WHITE, anchor="mm")
            cd.text((140, 75), it, font=fit(it, F_BLACK, 66, 720), fill=WHITE, anchor="lm")
            place(fr, card, W / 2 + (1 - a) * 300, CENTER_Y - 150 + k * 185, a)
    elif typ == "stamp":
        f = fit(v["text"], F_BLACK, 170, 820)
        s = lerp(2.2, 1, ease(lt / 0.22))
        st = Image.new("RGBA", (1000, 300), (0, 0, 0, 0)); sd = ImageDraw.Draw(st)
        sd.rounded_rectangle([20, 20, 980, 280], 30, outline=PINK + (255,), width=14)
        sd.text((500, 150), v["text"], font=f, fill=PINK, anchor="mm")
        place(fr, st, W / 2, CENTER_Y - 60, ease(lt / 0.15), s * 0.95, -6)
        if v.get("sub"):
            a = ease((lt - dur * 0.4) / 0.25)
            place(fr, grad_text(v["sub"], fit(v["sub"], F_BLACK, 150, 960), PEACH, ORANGE), W / 2, CENTER_Y + 190, a, lerp(1.6, 1, ease((lt - dur * 0.4) / 0.25)))
    elif typ == "bubble":
        f = font(F_BOLD, 64)
        lines = wrap(v["text"], f, 760)
        bh = 70 * len(lines) + 90
        b = Image.new("RGBA", (900, bh + 60), (0, 0, 0, 0)); bd = ImageDraw.Draw(b)
        bd.rounded_rectangle([10, 10, 890, bh], 50, fill=WHITE + (255,))
        bd.polygon([(600, bh - 4), (690, bh + 55), (700, bh - 4)], fill=WHITE + (255,))
        for i, ln in enumerate(lines):
            bd.text((450, 55 + 70 * i + 30), ln, font=f, fill=NAVY, anchor="mm")
        place(fr, b, W / 2, CENTER_Y + 60, fade, lerp(0.4, 1, pop))
    elif typ == "follow":
        btn = Image.new("RGBA", (620, 170), (0, 0, 0, 0)); bd = ImageDraw.Draw(btn)
        tapped = lt > dur * 0.5
        bd.rounded_rectangle([0, 0, 619, 169], 85, fill=(PINK if not tapped else (70, 60, 90)) + (255,))
        bd.text((310, 85), "Following ✓" if tapped else "Follow", font=font(F_BLACK, 72), fill=WHITE, anchor="mm")
        press = 1 - 0.08 * math.exp(-((lt - dur * 0.5) / 0.08) ** 2)
        place(fr, btn, W / 2, CENTER_Y + 40, fade, lerp(0.5, 1, pop) * press)
        place(fr, text_img("for daily GTA 6 news", font(F_BOLD, 56), PEACH), W / 2, CENTER_Y - 120, fade)
        # tapping hand (simple pointer)
        hx = lerp(W / 2 + 320, W / 2 + 120, ease((lt - dur * 0.2) / (dur * 0.28)))
        hy = lerp(CENTER_Y + 330, CENTER_Y + 90, ease((lt - dur * 0.2) / (dur * 0.28)))
        hand = Image.new("RGBA", (120, 200), (0, 0, 0, 0)); hd = ImageDraw.Draw(hand)
        hd.rounded_rectangle([40, 0, 80, 120], 20, fill=WHITE + (255,))
        hd.rounded_rectangle([10, 80, 110, 200], 40, fill=WHITE + (255,))
        place(fr, glow(hand, 10), hx, hy + 60, ease((lt - dur * 0.15) / 0.2))
        if tapped:
            r = (lt - dur * 0.5) * 900
            ring = layer(); ImageDraw.Draw(ring).ellipse([W / 2 + 120 - r, CENTER_Y + 90 - r, W / 2 + 120 + r, CENTER_Y + 90 + r], outline=PINK + (int(255 * clamp(1 - r / 300)),), width=8)
            fr.alpha_composite(ring)

def wrap(txt, f, maxw):
    words, lines, cur = txt.split(), [], ""
    for w_ in words:
        tst = (cur + " " + w_).strip()
        if f.getlength(tst) <= maxw: cur = tst
        else: lines.append(cur); cur = w_
    lines.append(cur); return lines

# ---------------- captions ----------------
def caption_chunks(b):
    words = b["say"].split()
    chunks, cur = [], []
    for w_ in words:
        cur.append(w_)
        if len(cur) >= 3 or w_.endswith((",", ".", "?", "!")):
            chunks.append(cur); cur = []
    if cur: chunks.append(cur)
    total_chars = sum(len(" ".join(c)) for c in chunks)
    t = b["t0"]; dur = b["t1"] - b["t0"]; out = []
    for c in chunks:
        d = dur * len(" ".join(c)) / max(1, total_chars)
        out.append((t, t + d, c)); t += d
    return out

def draw_caption(fr, chunk, t, key):
    t0, t1, words = chunk
    f = font(F_CAP, 76)
    keyset = set(k.lower().strip(".,?!") for k in (key or "").split())
    widths = [f.getlength(w_ + " ") for w_ in words]
    total = sum(widths); x = W / 2 - total / 2
    s = back((t - t0) / 0.18)
    L = layer(); d = ImageDraw.Draw(L)
    for w_, wd in zip(words, widths):
        col = YELLOW if w_.lower().strip(".,?!'\"") in keyset else WHITE
        d.text((x, 1170), w_, font=f, fill=col, stroke_width=7, stroke_fill=(10, 4, 20), anchor="lm")
        x += wd
    if s < 0.999:
        L = L.resize((W, H), Image.BILINEAR)
    place(fr, L, W / 2, H / 2, 1, lerp(0.85, 1, s)) if s < 0.999 else fr.alpha_composite(L)

# ---------------- main ----------------
def main():
    script = json.load(open(sys.argv[1])); out = sys.argv[2]
    models = sys.argv[sys.argv.index("--models") + 1] if "--models" in sys.argv else "."
    beats = script["beats"]
    voice, total = synth_voice(beats, script.get("voice", "am_michael"), script.get("speed", 1.15), models)
    total += 0.6
    cuts = [b["t0"] for b in beats]
    audio = music_bed(total, cuts)
    va = np.zeros_like(audio); va[:len(voice)] = voice[:len(audio)]
    mixa = va + audio; mixa = mixa / (np.abs(mixa).max() + 1e-6) * 0.92
    import soundfile as sf
    wav = out + ".wav"; sf.write(wav, mixa, SR)

    bg = make_bg(); corner = corner_icon()
    caps = [c for b in beats for c in caption_chunks(b)]
    handle = text_img(script.get("handle", "@bobdbuildergaming"), font(F_BOLD, 42), (255, 255, 255, 170))
    NF = int(total * FPS)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                             "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.2f}", "-movflags", "+faststart", out],
                            stdin=subprocess.PIPE)
    keycols = [PEACH, TEAL, ORANGE, PINK]
    for i in range(NF):
        t = i / FPS
        bi = max(0, max([k for k, b in enumerate(beats) if b["t0"] <= t + 1e-6] or [0]))
        b = beats[bi]
        end = beats[bi + 1]["t0"] if bi + 1 < len(beats) else total
        lt = t - b["t0"]; dur = end - b["t0"]
        # camera: slow push + slide-in from right, slide-out to left
        zoom = 1 + 0.05 * (lt / max(dur, 0.1))
        fr = bg.copy()
        content = layer()
        draw_visual(content, b.get("visual", {"type": "title", "text": b["say"][:20]}), lt, dur, ease(lt / 0.3), keycols[bi % 4])
        slide_in = (1 - ease(lt / 0.28)) * 260
        slide_out = ease((lt - (dur - 0.18)) / 0.18) * -260 if bi + 1 < len(beats) else 0
        if abs(zoom - 1) > 0.002:
            cw, ch = int(W * zoom), int(H * zoom)
            content = content.resize((cw, ch), Image.BILINEAR).crop(((cw - W) // 2, (ch - H) // 2 - 40, (cw - W) // 2 + W, (ch - H) // 2 - 40 + H))
        fr.alpha_composite(content, (int(slide_in + slide_out), 0))
        # flash on cut
        if lt < 0.08 and bi > 0:
            fl = Image.new("RGBA", (W, H), PINK + (int(70 * (1 - lt / 0.08)),)); fr.alpha_composite(fl)
        # character
        pose = b.get("pose", "stand")
        jump = -40 * math.sin(math.pi * clamp(lt / 0.3)) if lt < 0.3 else 0
        place(fr, character(pose, t, 0.62), W / 2 - 250, 1450 + jump)
        # caption
        for c in caps:
            if c[0] <= t < c[1] + 0.05:
                draw_caption(fr, c, t, b.get("caption_key", "")); break
        place(fr, corner, W - 120, 110)
        d = ImageDraw.Draw(fr); d.rectangle([0, 0, W * t / total, 8], fill=PINK + (255,))
        place(fr, handle, W / 2, H - 110)
        proc.stdin.write(fr.convert("RGB").tobytes())
    proc.stdin.close(); proc.wait()
    os.remove(wav)
    print("wrote", out, round(total, 1), "s")

if __name__ == "__main__":
    main()
