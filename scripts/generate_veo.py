import time
import sys
import json
import requests
import re
from google import genai
from google.genai import types

def extract_parts(full_prompt):
    """Découpe le prompt en utilisant le délimiteur ###"""
    parts = {"scenario": "", "voice_over": "", "music": ""}
    segments = full_prompt.split("###")
    for segment in segments:
        segment = segment.strip()
        if segment.upper().startswith("SCENARIO:"):
            parts["scenario"] = segment.replace("SCENARIO:", "").strip()
        elif segment.upper().startswith("VOICE-OVER:"):
            parts["voice_over"] = segment.replace("VOICE-OVER:", "").strip()
        elif segment.upper().startswith("MUSIC:"):
            parts["music"] = segment.replace("MUSIC:", "").strip()
    if not parts["voice_over"]:
        parts["voice_over"] = full_prompt
    return parts

def split_text_into_three(text):
    """Divise le texte en 3 parties égales basées sur le nombre de mots"""
    words = text.split()
    n = len(words)
    if n == 0:
        return "", "", ""
    
    # Calcul des points de coupure
    part1 = " ".join(words[0 : n//3])
    part2 = " ".join(words[n//3 : 2*n//3])
    part3 = " ".join(words[2*n//3 : ])
    
    return part1, part2, part3

def generate_video_with_refs():
    if len(sys.argv) < 2:
        print("Clé API manquante")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    try:
        with open("payload.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Erreur lecture payload.json:", e)
        sys.exit(1)
    
    full_prompt = data.get("prompt", "")
    image_urls = data.get("images", [])
    output_filename = "final_video.mp4"
    client = genai.Client(api_key=api_key)

    # 1. Extraction et Segmentation du texte
    extracted = extract_parts(full_prompt)
    visual_scenario = extracted["scenario"]
    
    # Division de la voix off en 3 parties pour éviter les répétitions
    v1, v2, v3 = split_text_into_three(extracted["voice_over"])

    # 2. Chargement des images de référence
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

    def wait_for_op(op):
        while not op.done:
            time.sleep(20)
            op = client.operations.get(op)
        if op.result and hasattr(op.result, 'generated_videos') and op.result.generated_videos:
            return op.result.generated_videos[0].video
        return None

    # --- ÉTAPE 1 : 0-8s (Partie 1 du texte) ---
    print(f"Étape 1/3 - Narration: {v1}")
    prompt_1 = (
        f"STRICT INSTRUCTION: Commercial Part 1. VISUAL: {visual_scenario}. "
        f"AUDIO: The narrator speaks ONLY this: '{v1}'"
    )
    
    op1 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt_1,
        config=types.GenerateVideosConfig(
            reference_images=reference_images if reference_images else None,
            duration_seconds=8,
            aspect_ratio="16:9"
        ),
    )
    current_video = wait_for_op(op1)
    if not current_video: sys.exit(1)

    time.sleep(40)

    # --- ÉTAPE 2 : 8-15s (Partie 2 du texte) ---
    print(f"Étape 2/3 - Narration: {v2}")
    prompt_2 = (
        f"CONTINUE SCENE. VISUAL: {visual_scenario}. "
        f"AUDIO: Continue narration with ONLY this: '{v2}'"
    )
    
    op2 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        video=current_video,
        prompt=prompt_2,
        config=types.GenerateVideosConfig(resolution="720p")
    )
    current_video = wait_for_op(op2)
    if not current_video: sys.exit(1)

    time.sleep(40)

    # --- ÉTAPE 3 : 15-22s (Partie 3 du texte) ---
    print(f"Étape 3/3 - Narration: {v3}")
    prompt_3 = (
        f"FINAL PART. VISUAL: {visual_scenario}. "
        f"AUDIO: Finish narration with ONLY this: '{v3}'"
    )
    
    op3 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        video=current_video,
        prompt=prompt_3,
        config=types.GenerateVideosConfig(resolution="720p")
    )
    current_video = wait_for_op(op3)
    if not current_video: sys.exit(1)

    # --- SAUVEGARDE FINALE ---
    try:
        file_content = client.files.download(file=current_video.uri)
        with open(output_filename, "wb") as f:
            f.write(file_content)
        print("Succès !")
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    generate_video_with_refs()
