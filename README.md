# Product Image Enhancer

A FastAPI backend and simple HTML frontend tool to enhance product images by removing backgrounds, refining edges, and adjusting brightness, contrast, and sharpness.

## Features

- Remove image background using `rembg`
- Refine object edges
- Brightness, contrast, and sharpness enhancement
- Transparent PNG output
- Simple HTML frontend for preview

## Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- rembg
- Pillow
- OpenCV (`opencv-python-headless`)
- NumPy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/product-image-enhancer.git
cd product-image-enhancer
