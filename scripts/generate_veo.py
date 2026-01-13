import time
import sys
import json
import requests
from google import genai
from google.genai import types

def generate_video_with_refs():
    if len(sys.argv) < 4:
        print("Erreur: Arguments manquants")
        sys.exit(1)

    api_key = sys.argv[1]
    prompt = sys.argv[2] # Votre prompt combiné (Scénario + Voix off)
    image_urls = json.loads(sys.argv[3])
    output_filename = "final_video.mp4"
    client = genai.Client(api_key=api_key)

    # Préparation des images
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

    # --- ÉTAPE 1 : GÉNÉRATION INITIALALE (0-8s) ---
    print("Génération du segment initial...")
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=f"FRANÇAIS / FRENCH VOICE-OVER: {prompt}",
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

    # --- ÉTAPE 2 : EXTENSIONS (Pour atteindre 22s) ---
    # On étend 2 fois (7s chaque fois)
    for i in range(2):
        print(f"Extension de la vidéo (Etape {i+1}/2)...")
        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            video=current_video, # Utilise la vidéo précédente
            prompt=f"Continue the cinematic commercial in French. {prompt}",
            config=types.GenerateVideosConfig(resolution="720p")
        )

        while not operation.done:
            time.sleep(15)
            operation = client.operations.get(operation)
        
        current_video = operation.result.generated_videos[0].video

    # --- ÉTAPE 3 : SAUVEGARDE FINALE ---
    try:
        file_content = client.files.download(file=current_video.uri)
        with open(output_filename, "wb") as f:
            f.write(file_content)
        print(f"Succès ! Vidéo de 22s sauvegardée.")
    except Exception as e:
        print(f"Erreur téléchargement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_video_with_refs()
