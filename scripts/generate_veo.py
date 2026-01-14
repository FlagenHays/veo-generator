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

    # --- ÉTAPE 1 : 0-8s (Focus sur l'Asset) ---
    print("Étape 1: Initialisation avec images de référence...")
    # On insiste lourdement sur la fidélité visuelle aux images fournies
    prompt_1 = f"STRICT VISUAL FIDELITY: Use the EXACT object from reference images. START OF VIDEO (0-8s). Voice-over starts. Full script for context: {full_prompt}"
    
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt_1,
        config=types.GenerateVideosConfig(
            reference_images=reference_images if reference_images else None,
            duration_seconds=8,
            aspect_ratio="16:9"
        ),
    )

    while not operation.done:
        time.sleep(15)
        operation = client.operations.get(operation)
    
    current_video = operation.result.generated_videos[0].video

    # --- ÉTAPE 2 : 8-15s (Continuation) ---
    print("Étape 2: Extension médiane...")
    prompt_2 = f"CONTINUE the scene. KEEP the exact same bag design. DO NOT REPEAT the beginning of the voice-over. Move to the middle part of this script: {full_prompt}"
    
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        video=current_video,
        prompt=prompt_2,
        config=types.GenerateVideosConfig(resolution="720p")
    )

    while not operation.done:
        time.sleep(15)
        operation = client.operations.get(operation)
    
    current_video = operation.result.generated_videos[0].video

    # --- ÉTAPE 3 : 15-22s (Finalisation - On réduit le prompt ici pour éviter la répétition) ---
    print("Étape 3: Conclusion finale...")
    # Ici on ne donne QUE la fin du script pour éviter que l'IA ne boucle
    prompt_3 = f"FINAL 7 SECONDS. End the narration. End the music. Smooth fade out. Use ONLY the end part of the script: {full_prompt[-150:]}"
    
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        video=current_video,
        prompt=prompt_3,
        config=types.GenerateVideosConfig(resolution="720p")
    )

    while not operation.done:
        time.sleep(15)
        operation = client.operations.get(operation)
    
    current_video = operation.result.generated_videos[0].video

    # --- SAUVEGARDE ---
    try:
        file_content = client.files.download(file=current_video.uri)
        with open(output_filename, "wb") as f:
            f.write(file_content)
        print("Succès !")
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    generate_video_with_refs()
