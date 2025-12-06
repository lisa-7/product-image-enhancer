from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from rembg import remove
from io import BytesIO
from PIL import Image, ImageEnhance
import base64

app = FastAPI()

# Allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_white_product(image_np):
    """Check if the product is mostly white (>60% bright pixels)."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray > 200)
    total_pixels = gray.size
    return (bright_pixels / total_pixels) > 0.60


@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):

    contents = await file.read()
    input_image = Image.open(BytesIO(contents)).convert("RGBA")

    # Step 1 — Remove background
    removed_bg = remove(input_image)
    removed_np = np.array(removed_bg)

    # Step 2 — Refine edges (smooth alpha channel)
    alpha = removed_np[:, :, 3]
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)  # smoother edge
    removed_np[:, :, 3] = alpha
    removed_bg = Image.fromarray(removed_np)

    # Step 3 — Enhancement applied only on the object
    product = removed_bg.copy()
    alpha = product.split()[3]
    rgb_product = product.convert("RGB")

    # mild denoise (retain texture)
    rgb_np = np.array(rgb_product)
    rgb_np = cv2.fastNlMeansDenoisingColored(rgb_np, None, 3, 3, 7, 21)
    rgb_product = Image.fromarray(rgb_np)

    # brightness / contrast
    rgb_product = ImageEnhance.Brightness(rgb_product).enhance(1.05)
    rgb_product = ImageEnhance.Contrast(rgb_product).enhance(1.10)

    # 🔥 natural sharpness
    rgb_product = ImageEnhance.Sharpness(rgb_product).enhance(1.12)

    # merge back alpha to preserve transparency
    enhanced = rgb_product.convert("RGBA")
    enhanced.putalpha(alpha)

    # Step 4 — Return Base64 PNG (transparent)
    buffer = BytesIO()
    enhanced.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return JSONResponse(content={"enhanced_image": img_str})
