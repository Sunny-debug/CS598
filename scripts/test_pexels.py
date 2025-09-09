import os
from pathlib import Path
import requests
from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.getenv("PEXELS_API_KEY")
if not API_KEY:
    raise ValueError("PEXELS_API_KEY not set. Please add it to your .env file.")

headers = {"Authorization": API_KEY}

def download_first_image(query="nature", out_file="pexels_test.jpg"):
    url = "https://api.pexels.com/v1/search"
    params = {"query": query, "per_page": 1}
    res = requests.get(url, headers=headers, params=params, timeout=30)
    res.raise_for_status()
    data = res.json()
    if not data.get("photos"):
        print("No photos found.")
        return
    img_url = data["photos"][0]["src"]["original"]
    img = requests.get(img_url, timeout=60)
    img.raise_for_status()
    Path(out_file).write_bytes(img.content)
    print(f"Downloaded {query} image to {out_file}")

if __name__ == "__main__":
    download_first_image()