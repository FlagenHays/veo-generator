import time
import sys
import json
import requests
import re
from google import genai
from google.genai import types
from google.genai.errors import ClientError

def generate_video_with_refs():
    if len(sys.argv) < 4:
        sys.exit(1)

    api_key = sys.argv[1]
    full_raw_prompt = sys.argv[2]
    image_urls = json.loads(sys.argv[3])
    output_filename = "final_video.mp4"
    client = genai.Client(api_key=api_key)

    # --- 1. EXTRACTION ET NETTOYAGE ---
    try:
        parts = full_raw_prompt.split('|')
        scenario = parts[0].replace('SCENARIO:', '').strip()
        voix_off_raw = parts[1].replace('VOICE-OVER (FRENCH):', '').strip()
        voix_off_clean = re.sub(r'\(.*?\)', '', voix_off_raw).strip()
        musique = parts[2].replace('MUSIC:', '').strip() if len(parts) > 2 else ""
    except:
        scenario = full_raw_prompt
        voix_off_clean = full_raw_prompt
        musique = ""

    v_words = voix_off_clean.split()
    vn = len(v_words)
    v_part1 = " ".join(v_words[:vn//3])
    v_part2 = " ".join(v_words[vn//3:2*vn//3])
    v_part3 = " ".join(v_words[2*vn//3:])

    # --- 2. CHARGEMENT DES IMAGES ---
    reference_images = []
    if isinstance(image_urls, list):
        for url in image_urls[:3]:
            try:
                img_response = requests.get(url, timeout=15)
                if img_response.status_code == 200:
                    ref = types.VideoGenerationReferenceImage(
                        image=types.Image(bytes=img_response.content, mime_type="image/jpeg"),
                        reference_type="ASSET"
                    )
                    reference_images.append(ref)
            except Exception: pass

    final_refs = None
    if len(reference_images) >= 2:
        final_refs = reference_images
    elif len(reference_images) == 1:
        final_refs = [reference_images[0], reference_images[0]]

    def wait_for_op(op):
        while not op.done:
            time.sleep(20)
            op = client.operations.get(op)
        return op.result.generated_videos[0].video if (op.result and op.result.generated_videos) else None

    def call_with_retry(func, *args, **kwargs):
        """Tente l'appel API et attend 90s si le quota est épuisé"""
        for i in range(3): # 3 tentatives
            try:
                return func(*args, **kwargs)
            except ClientError as e:
                if "429" in str(e):
                    print(f"Quota atteint (429). Pause de 90 secondes... (Tentative {i+1}/3)")
                    time.sleep(90)
                else:
                    raise e
        return None

    # --- 3. GÉNÉRATION DES 3 ÉTAPES ---

    # ÉTAPE 1
    print(f"Étape 1: Narration: {v_part1}")
    prompt_1 = f"VISUAL: {scenario}. STYLE: cinematic. START VIDEO. VOICE-OVER: {v_part1}"
    op1 = call_with_retry(client.models.generate_videos, 
        model="veo-3.1-generate-preview",
        prompt=prompt_1,
        config=types.GenerateVideosConfig(reference_images=final_refs, duration_seconds=8, aspect_ratio="16:9")
    )
    current_video = wait_for_op(op1)
    if not current_video: sys.exit(1)

    print("Attente quota (70s)...")
    time.sleep(70) 

    # ÉTAPE 2
    print(f"Étape 2: Narration: {v_part2}")
    prompt_2 = f"CONTINUE VISUALS. STRICT CONSISTENCY. VOICE-OVER ONLY: {v_part2}"
    op2 = call_with_retry(client.models.generate_videos,
        model="veo-3.1-generate-preview",
        video=current_video,
        prompt=prompt_2,
        config=types.GenerateVideosConfig(resolution="720p")
    )
    current_video = wait_for_op(op2)
    if not current_video: sys.exit(1)

    print("Attente quota (70s)...")
    time.sleep(70)

    # ÉTAPE 3
    print(f"Étape 3: Narration: {v_part3}")
    prompt_3 = f"FINAL PART. END VIDEO. LOGO ANIMATION. VOICE-OVER ONLY: {v_part3}"
    op3 = call_with_retry(client.models.generate_videos,
        model="veo-3.1-generate-preview",
        video=current_video,
        prompt=prompt_3,
        config=types.GenerateVideosConfig(resolution="720p")
    )
    current_video = wait_for_op(op3)

    # --- 4. TÉLÉCHARGEMENT ---
    if current_video:
        file_content = client.files.download(file=current_video.uri)
        with open(output_filename, "wb") as f:
            f.write(file_content)
        print("Succès !")
    else:
        sys.exit(1)

if __name__ == "__main__":
    generate_video_with_refs()
