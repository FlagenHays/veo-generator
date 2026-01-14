import time
import sys
import json
import requests
from google import genai
from google.genai import types

def generate_video_with_refs():
    if len(sys.argv) < 4:
        sys.exit(1)

    api_key = sys.argv[1]
    full_prompt = sys.argv[2]
    image_urls = json.loads(sys.argv[3])
    output_filename = "final_video.mp4"
    client = genai.Client(api_key=api_key)

    # --- LOGIQUE DE DÉCOUPAGE DU TEXTE ---
    # On sépare le texte en 3 parties pour éviter que l'IA ne lise 3 fois la même chose
    words = full_prompt.split()
    n = len(words)
    part1 = " ".join(words[:n//3])
    part2 = " ".join(words[n//3:2*n//3])
    part3 = " ".join(words[2*n//3:])

    # 1. Chargement des images de référence
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

    # --- ÉTAPE 1 : 0-8s ---
    print(f"Étape 1/3: Narration: '{part1[:50]}...'")
    prompt_1 = f"STRICT VISUAL FIDELITY. Use references. START OF VIDEO. French Voice-over: {part1}"
    
    op1 = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt_1,
        config=types.GenerateVideosConfig(
            reference_images=reference_images if reference_images else None,
            duration_seconds=8,
            aspect_ratio="16:9"
        ),
    )
    current_video = wait_for_op(op1)
    if not current_video: sys.exit(1)

    time.sleep(40) # Pause Quota

    # --- ÉTAPE 2 : 8-15s ---
    print(f"Étape 2/3: Narration: '{part2[:50]}...'")
    prompt_2 = f"CONTINUE VIDEO. KEEP SAME DESIGN. Smoothly continue French narration with ONLY these words: {part2}"
    
    op2 = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        video=current_video,
        prompt=prompt_2,
        config=types.GenerateVideosConfig(resolution="720p")
    )
    current_video = wait_for_op(op2)
    if not current_video: sys.exit(1)

    time.sleep(40) # Pause Quota

    # --- ÉTAPE 3 : 15-22s ---
    print(f"Étape 3/3: Narration: '{part3[:50]}...'")
    prompt_3 = f"FINAL PART. Finish video. End French narration with ONLY these words: {part3}. Add logo animation."
    
    op3 = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        video=current_video,
        prompt=prompt_3,
        config=types.GenerateVideosConfig(resolution="720p")
    )
    current_video = wait_for_op(op3)
    if not current_video: sys_exit(1)

    # --- SAUVEGARDE ---
    file_content = client.files.download(file=current_video.uri)
    with open(output_filename, "wb") as f:
        f.write(file_content)
    print("Succès ! Voix off fluide sans répétition.")

if __name__ == "__main__":
    generate_video_with_refs()
