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
    full_raw_prompt = sys.argv[2] # Le prompt mixé venant de Laravel
    image_urls = json.loads(sys.argv[3])
    output_filename = "final_video.mp4"
    client = genai.Client(api_key=api_key)

    # --- EXTRACTION INTELLIGENTE ---
    # On sépare pour isoler uniquement ce que l'IA doit DIRE
    try:
        parts = full_raw_prompt.split('|')
        scenario = parts[0].replace('SCENARIO:', '').strip()
        voix_off_complete = parts[1].replace('VOICE-OVER (FRENCH):', '').strip()
        musique = parts[2].replace('MUSIC:', '').strip() if len(parts) > 2 else ""
    except:
        # Fallback si le format est bizarre
        scenario = full_raw_prompt
        voix_off_complete = full_raw_prompt
        musique = ""

    # Découpage de la VOIX OFF uniquement
    v_words = voix_off_complete.split()
    vn = len(v_words)
    v_part1 = " ".join(v_words[:vn//3])
    v_part2 = " ".join(v_words[vn//3:2*vn//3])
    v_part3 = " ".join(v_words[2*vn//3:])

    # 1. Chargement des images
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
        return op.result.generated_videos[0].video if op.result else None

    # --- ÉTAPE 1 : 0-8s ---
    print(f"Étape 1: Narration: {v_part1}")
    # On donne le scénario en "Instruction visuelle" et la v_part1 en "Voice-over"
    prompt_1 = f"VISUAL INSTRUCTION: {scenario}. BACKGROUND MUSIC: {musique}. START VIDEO. FRENCH VOICE-OVER TO SPEAK: {v_part1}"
    
    op1 = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt_1,
        config=types.GenerateVideosConfig(reference_images=reference_images, duration_seconds=8, aspect_ratio="16:9"),
    )
    current_video = wait_for_op(op1)
    
    time.sleep(40) # Quota

    # --- ÉTAPE 2 : 8-15s ---
    print(f"Étape 2: Narration: {v_part2}")
    prompt_2 = f"CONTINUE VISUALS: {scenario}. DO NOT REPEAT PREVIOUS WORDS. FRENCH VOICE-OVER TO SPEAK ONLY: {v_part2}"
    
    op2 = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        video=current_video,
        prompt=prompt_2,
        config=types.GenerateVideosConfig(resolution="720p")
    )
    current_video = wait_for_op(op2)

    time.sleep(40) # Quota

    # --- ÉTAPE 3 : 15-22s ---
    print(f"Étape 3: Narration: {v_part3}")
    prompt_3 = f"FINAL VISUALS: {scenario}. END VIDEO. FRENCH VOICE-OVER TO SPEAK ONLY: {v_part3}"
    
    op3 = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        video=current_video,
        prompt=prompt_3,
        config=types.GenerateVideosConfig(resolution="720p")
    )
    current_video = wait_for_op(op3)

    # --- SAUVEGARDE ---
    file_content = client.files.download(file=current_video.uri)
    with open(output_filename, "wb") as f:
        f.write(file_content)
    print("Succès ! La voix off est maintenant isolée du scénario.")

if __name__ == "__main__":
    generate_video_with_refs()
