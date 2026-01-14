import time
import sys
import json
import requests
from google import genai
from google.genai import types

def generate_video_with_refs():
    if len(sys.argv) < 4:
        print("Erreur: Arguments manquants (API_KEY, PROMPT, IMAGES_JSON)")
        sys.exit(1)

    api_key = sys.argv[1]
    full_prompt = sys.argv[2] # Le script complet de 22s
    image_urls = json.loads(sys.argv[3])
    output_filename = "final_video.mp4"
    client = genai.Client(api_key=api_key)

    # 1. Préparation des images de référence
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
                    print(f"Image chargée: {url}")
            except Exception as e:
                print(f"Erreur image {url}: {e}")

    # --- ÉTAPE 1 : GÉNÉRATION INITIALE (0 à 8 secondes) ---
    print("Étape 1/3: Génération des 8 premières secondes...")
    prompt_1 = f"START OF COMMERCIAL (0-8s). Voice-over starts now. Full Script: {full_prompt}"
    
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

    # --- ÉTAPE 2 : PREMIÈRE EXTENSION (8 à 15 secondes) ---
    print("Étape 2/3: Extension vers 15 secondes...")
    # On dit à l'IA de CONTINUER la voix off sans la reprendre au début
    prompt_2 = f"CONTINUATION (8-15s). Smoothly continue the French voice-over from where it left off. DO NOT REPEAT the beginning. Script: {full_prompt}"
    
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

    # --- ÉTAPE 3 : DEUXIÈME EXTENSION (15 à 22 secondes) ---
    print("Étape 3/3: Extension finale vers 22 secondes...")
    # On demande la fin du script et l'animation du logo
    prompt_3 = f"FINAL PART (15-22s). Finish the French narration and the music. End with a premium logo animation. Script: {full_prompt}"
    
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

    # --- TÉLÉCHARGEMENT ---
    try:
        file_content = client.files.download(file=current_video.uri)
        with open(output_filename, "wb") as f:
            f.write(file_content)
        print(f"Succès: Vidéo de 22s enregistrée sous {output_filename}")
    except Exception as e:
        print(f"Erreur finale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_video_with_refs()
