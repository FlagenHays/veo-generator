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
    prompt = sys.argv[2]
    
    try:
        image_urls = json.loads(sys.argv[3])
    except Exception as e:
        print(f"Erreur JSON: {e}")
        image_urls = []

    output_filename = "final_video.mp4"
    client = genai.Client(api_key=api_key)
    
    # Préparation images de référence
    reference_images = []
    if isinstance(image_urls, list):
        for url in image_urls[:3]:
            try:
                img_res = requests.get(url, timeout=15)
                if img_res.status_code == 200:
                    reference_images.append(types.VideoGenerationReferenceImage(
                        image=types.Image(bytes=img_res.content, mime_type="image/jpeg"),
                        reference_type="ASSET"
                    ))
            except Exception: continue

    # ÉTAPE 1: Génération initiale (8s)
    print("Lancement Phase 1 (8s)...")
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
        config=types.GenerateVideosConfig(
            reference_images=reference_images if reference_images else None,
            duration_seconds=8,
            aspect_ratio="9:16"
        ),
    )

    while not operation.done:
        time.sleep(15)
        operation = client.operations.get(operation)

    video_data = operation.result.generated_videos[0].video

    # ÉTAPE 2: Extensions pour atteindre ~22s
    for i in range(2):
        print(f"Lancement Extension {i+1}...")
        op_ext = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=prompt,
            config=types.GenerateVideosConfig(
                video=video_data, # On passe la vidéo précédente
                prompt=prompt + " Continue narrative.",
                duration_seconds=8
            ),
        )
        while not op_ext.done:
            time.sleep(15)
            op_ext = client.operations.get(op_ext)
        video_data = op_ext.result.generated_videos[0].video

    # Sauvegarde finale
    try:
        file_content = client.files.download(file=video_data.uri)
        with open(output_filename, "wb") as f:
            f.write(file_content)
        print(f"Succès: {output_filename}")
    except Exception as e:
        print(f"Erreur téléchargement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_video_with_refs()
