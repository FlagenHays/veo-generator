import time
import sys
import json
import requests
from google import genai
from google.genai import types

def generate_video_with_refs():
    # 1. Récupération des arguments
    if len(sys.argv) < 4:
        print("Erreur: Arguments manquants (API_KEY, PROMPT, IMAGES_JSON)")
        sys.exit(1)

    api_key = sys.argv[1]
    prompt = sys.argv[2]
    
    try:
        # Nettoyage et chargement du JSON des images
        images_raw = sys.argv[3]
        image_urls = json.loads(images_raw)
    except Exception as e:
        print(f"Erreur lors du parsing JSON des images: {e}")
        image_urls = []

    output_filename = "final_video.mp4"
    client = genai.Client(api_key=api_key)
    
    # 2. Préparation des images de référence (Max 3)
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
                    print(f"Image ajoutée: {url}")
            except Exception as e:
                print(f"Impossible de charger l'image {url}: {e}")

    # 3. ÉTAPE 1: Génération initiale (8s)
    print(f"Lancement Phase 1 (8s) avec prompt: {prompt[:50]}...")
    try:
        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt=prompt,
            config=types.GenerateVideosConfig(
                reference_images=reference_images if reference_images else None,
                duration_seconds=8,
                aspect_ratio="16:9"
            ),
        )

        while not operation.done:
            print("Génération en cours (Phase 1)...")
            time.sleep(15)
            operation = client.operations.get(operation)

        video_data = operation.result.generated_videos[0].video

        # 4. ÉTAPE 2: Extensions pour atteindre 22s (2 x 7s supplémentaires)
        for i in range(2):
            print(f"Lancement Extension {i+1} (Ajout de durée)...")
            op_ext = client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    video=video_data, # Référence à la vidéo précédente
                    prompt=prompt + " Continue the action naturally.",
                    duration_seconds=8
                ),
            )
            while not op_ext.done:
                print(f"Extension {i+1} en cours...")
                time.sleep(15)
                op_ext = client.operations.get(op_ext)
            
            # Mise à jour de la référence vidéo pour l'extension suivante
            video_data = op_ext.result.generated_videos[0].video

        # 5. Sauvegarde finale
        print("Téléchargement de la vidéo finale de 22s...")
        file_content = client.files.download(file=video_data.uri)
        with open(output_filename, "wb") as f:
            f.write(file_content)
            
        print(f"Succès ! Vidéo 22s sauvegardée sous {output_filename}")

    except Exception as e:
        print(f"Erreur durant la génération Veo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_video_with_refs()
