from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from rembg import remove
from io import BytesIO
from PIL import Image, ImageEnhance, UnidentifiedImageError
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    remove_bg: bool = Form(False),
    enhance: bool = Form(False),
    order: str = Form("enhance_first"),  # enhance_first | remove_first
):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
    except UnidentifiedImageError:
        return JSONResponse(content={"error": "Invalid or corrupted image file"}, status_code=400)

    # Normalize formats (no effect on final look, only prevents crashes)
    if img.mode not in ["RGBA", "RGB"]:
        img = img.convert("RGBA") if "A" in img.getbands() else img.convert("RGB")
    if img.mode == "RGB":
        img = img.convert("RGBA")

    def do_enhance(input_image):
        alpha = input_image.split()[3]
        rgb = input_image.convert("RGB")
        rgb_np = np.array(rgb)

        # Highlight protection
        hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v_mask = v > 220
        v[v_mask] = (v[v_mask] * 0.985).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        rgb_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        rgb = Image.fromarray(rgb_np)
        rgb = ImageEnhance.Brightness(rgb).enhance(1.02)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.05)
        rgb = ImageEnhance.Sharpness(rgb).enhance(1.05)

        rgb.putalpha(alpha)
        return rgb

    def do_bg_removal(input_image):
        try:
            removed = remove(input_image)
        except Exception:
            # rembg throws recursion errors on some 30MB PNGs — prevent failure
            removed = remove(input_image.convert("RGB"))

        removed_np = np.array(removed)
        alpha = removed_np[:, :, 3]

        mask = (alpha > 5)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        alpha = np.where(mask, alpha, 0).astype(np.uint8)

        removed_np[:, :, 3] = alpha
        return Image.fromarray(removed_np)

    # Decision logic (unchanged results)
    if enhance and remove_bg:
        if order == "enhance_first":
            img = do_enhance(img)
            img = do_bg_removal(img)
        else:
            img = do_bg_removal(img)
            img = do_enhance(img)
    elif enhance:
        img = do_enhance(img)
    elif remove_bg:
        img = do_bg_removal(img)

    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return JSONResponse(content={"enhanced_image": img_str})
