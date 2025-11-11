"""Create a presentation PNG with the membership plot and the if->then thresholds.

Reads `membership_from_summary.png` and writes `membership_slide.png`.
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

IN = Path('membership_from_summary.png')
OUT = Path('membership_slide.png')

if not IN.exists():
    raise SystemExit(f'{IN} not found. Run plot_membership_from_stats.py first.')

img = Image.open(IN).convert('RGBA')
W, H = img.size

# Slide size: extend width for text
pad = 40
text_area_w = 520
out_w = W + text_area_w + pad*3
out_h = max(H, 700)
out = Image.new('RGBA', (out_w, out_h), (255,255,255,255))
out.paste(img, (pad, (out_h - H)//2))

draw = ImageDraw.Draw(out)

# Load a truetype font if available, else default
try:
    font_title = ImageFont.truetype('arial.ttf', 28)
    font_body = ImageFont.truetype('arial.ttf', 18)
except Exception:
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

# Text block on right
tx_x = W + pad*2
tx_y = pad
draw.text((tx_x, tx_y), 'Membership thresholds (probability 0..1)', fill=(0,0,0), font=font_title)
tx_y += 44

lines = [
    'If p ≤ 0.40 -> NOT DIABETIC (low risk)',
    'If 0.40 < p ≤ 0.60 -> NOT LIKELY (low risk)',
    'If 0.60 < p ≤ 0.70 -> LIKELY (moderate risk)',
    'If 0.70 < p ≤ 0.85 -> DIABETIC (high risk)',
    'If p > 0.85 -> HIGHLY DIABETIC (very high risk)',
    '',
    'Caution: thresholds are for this model and must be',
    'calibrated and clinically validated before use.'
]

for line in lines:
    draw.text((tx_x, tx_y), line, fill=(0,0,0), font=font_body)
    tx_y += 26

out.save(OUT)
print('Saved slide to', OUT)
