# Build a before/after boundary-sharpening montage for the slides.
# Columns: input | GT | PPA baseline | + boundary sharpening
# Rows: test images where the sharpening most reduces boundary-band MAE.
import os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT, "data", "images")
GT_DIR = os.path.join(ROOT, "data", "ground_truth_mask")
BASE_DIR = os.path.join(ROOT, "results", "baseline")
SHARP_DIR = os.path.join(ROOT, "results", "A7")
FONT = os.path.join(ROOT, "..", "TemplateNKU-master", "fonts", "LXGWWenKai-Regular.ttf")
OUT = os.path.join(ROOT, "..", "TemplateNKU-master", "image", "boundary_sharpen_grid.png")

def load_gray(path, size=None):
    im = Image.open(path).convert("L")
    if size is not None:
        im = im.resize(size, Image.BILINEAR)
    return im

def band_mask(gt_im):
    # morphological edge then dilate to a ~5px band
    dil = gt_im.filter(ImageFilter.MaxFilter(3))
    ero = gt_im.filter(ImageFilter.MinFilter(3))
    edge = np.asarray(dil, np.int16) - np.asarray(ero, np.int16)
    edge_im = Image.fromarray((edge > 10).astype(np.uint8) * 255)
    band = np.asarray(edge_im.filter(ImageFilter.MaxFilter(11))) > 0
    return band

def bmae(pred, gt, band):
    p = np.asarray(pred, np.float32) / 255.0
    g = (np.asarray(gt, np.float32) / 255.0 >= 0.5).astype(np.float32)
    if band.sum() == 0:
        return 0.0
    return float(np.abs(p - g)[band].mean())

def iou(pred, gt):
    p = np.asarray(pred) >= 128
    g = np.asarray(gt) >= 128
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else 0.0

names = sorted(f for f in os.listdir(BASE_DIR) if f.endswith(".png"))
rows = []
for n in names:
    stem = os.path.splitext(n)[0]
    gp = os.path.join(GT_DIR, stem + ".png")
    if not os.path.exists(gp):
        continue
    gt = load_gray(gp)
    W, H = gt.size
    base = load_gray(os.path.join(BASE_DIR, n), (W, H))
    sharp = load_gray(os.path.join(SHARP_DIR, n), (W, H))
    band = band_mask(gt)
    fg = (np.asarray(gt) >= 128).mean()
    if fg < 0.05 or fg > 0.55:   # skip extreme tiny/huge masks
        continue
    # both versions must already detect the object well -> isolate EDGE quality,
    # not detection success/failure (otherwise we'd cherry-pick baseline failures)
    if iou(base, gt) < 0.78 or iou(sharp, gt) < 0.78:
        continue
    band_frac = band.mean()      # prefer thin / complex contours
    imp = bmae(base, gt, band) - bmae(sharp, gt, band)
    rows.append((imp, stem, W, H, band_frac))

rows.sort(reverse=True)
pick = [r[1] for r in rows[:4]]
print("selected:", pick, "top improvements:", [round(r[0], 4) for r in rows[:4]])

# ---- compose montage ----
CW, CH = 260, 200          # cell content area
PAD = 8
HEAD = 46
cols = ["输入图", "GT 标注", "PPA 基线", "+ 边界锐化（本方案）"]
ncol, nrow = 4, len(pick)
GW = ncol * CW + (ncol + 1) * PAD
GH = HEAD + nrow * CH + (nrow + 1) * PAD
canvas = Image.new("RGB", (GW, GH), "white")
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype(FONT, 26)

def paste_fit(im, cx, cy):
    im = im.convert("RGB")
    w, h = im.size
    s = min(CW / w, CH / h)
    im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    ox = cx + (CW - im.size[0]) // 2
    oy = cy + (CH - im.size[1]) // 2
    canvas.paste(im, (ox, oy))

for j, title in enumerate(cols):
    x = PAD + j * (CW + PAD)
    tb = draw.textbbox((0, 0), title, font=font)
    tw = tb[2] - tb[0]
    color = (45, 125, 77) if j == 3 else (60, 60, 60)
    draw.text((x + (CW - tw) // 2, 8), title, fill=color, font=font)

def gt_bbox(gt, margin=0.18):
    a = np.asarray(gt) >= 128
    ys, xs = np.where(a)
    W, H = gt.size
    if len(xs) == 0:
        return (0, 0, W, H)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx = int((x1 - x0) * margin) + 4
    my = int((y1 - y0) * margin) + 4
    return (max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my))

for i, stem in enumerate(pick):
    y = HEAD + PAD + i * (CH + PAD)
    gt = load_gray(os.path.join(GT_DIR, stem + ".png"))
    W, H = gt.size
    inp = Image.open(os.path.join(IMG_DIR, stem + ".jpg")).convert("RGB")
    base = load_gray(os.path.join(BASE_DIR, stem + ".png"), (W, H))
    sharp = load_gray(os.path.join(SHARP_DIR, stem + ".png"), (W, H))
    box = gt_bbox(gt)          # zoom every cell to the object so contours are legible
    for j, im in enumerate([inp, gt, base, sharp]):
        x = PAD + j * (CW + PAD)
        paste_fit(im.crop(box), x, y)

canvas.save(OUT)
print("saved:", OUT, canvas.size)
