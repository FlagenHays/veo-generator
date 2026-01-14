import time
import sys
import json
import requests
import re
from google import genai
from google.genai import types

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
        # Séparation par le caractère | utilisé dans Laravel
        parts = full_raw_prompt.split('|')
        scenario = parts[0].replace('SCENARIO:', '').strip()
        # On nettoie la voix off des parenthèses comme "(Voix chaleureuse...)"
        voix_off_raw = parts[1].replace('VOICE-OVER (FRENCH):', '').strip()
        voix_off_clean = re.sub(r'\(.*?\)', '', voix_off_raw).strip()
        musique = parts[2].replace('MUSIC:', '').strip() if len(parts) > 2 else ""
    except:
        scenario = full_raw_prompt
        voix_off_clean = full_raw_prompt
        musique = ""

    # Découpage de la VOIX OFF en 3 segments
    v_words = voix_off_clean.split()
    vn = len(v_words)
    v_part1 = " ".join(v_words[:vn//3])
    v_part2 = " ".join(v_words[vn//3:2*vn//3])
    v_part3 = " ".join(v_words[2*vn//3:])

    # --- 2. CHARGEMENT DES IMAGES (Sécurité 400) ---
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

    # Sécurité vitale pour Veo 3.1 : Il faut 0 ou >= 2 images
    final_refs = None
    if len(reference_images) >= 2:
        final_refs = reference_images
    elif len(reference_images) == 1:
        # On duplique la seule image pour arriver à 2 et éviter l'erreur 400
        final_refs = [reference_images[0], reference_images[0]]

    def wait_for_op(op):
        while not op.done:
            time.sleep(20)
            op = client.operations.get(op)
        return op.result.generated_videos[0].video if (op.result and op.result.generated_videos) else None

    # --- 3. GÉNÉRATION DES 3 ÉTAPES ---

    # ÉTAPE 1 (0-8s)
    print(f"Étape 1: Narration: {v_part1}")
    prompt_1 = f"VISUAL: {scenario}. STYLE: cinematic. START VIDEO. VOICE-OVER: {v_part1}"
    op1 = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt_1,
        config=types.GenerateVideosConfig(reference_images=final_refs, duration_seconds=8, aspect_ratio="16:9"),
    )
    current_video = wait_for_op(op1)
    if not current_video: sys.exit(1)

    time.sleep(40) # Pause Quota

    # ÉTAPE 2 (8-15s)
    print(f"Étape 2: Narration: {v_part2}")
    prompt_2 = f"CONTINUE VISUALS. STRICT CONSISTENCY. VOICE-OVER ONLY: {v_part2}"
    op2 = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        video=current_video,
        prompt=prompt_2,
        config=types.GenerateVideosConfig(resolution="720p")
    )
    current_video = wait_for_op(op2)
    if not current_video: sys.exit(1)

    time.sleep(40) # Pause Quota

    # ÉTAPE 3 (15-22s)
    print(f"Étape 3: Narration: {v_part3}")
    prompt_3 = f"FINAL PART. END VIDEO. LOGO ANIMATION. VOICE-OVER ONLY: {v_part3}"
    op3 = client.models.generate_videos(
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
