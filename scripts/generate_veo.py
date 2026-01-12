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
        image_urls = json.loads(sys.argv[3])
    except Exception as e:
        print(f"Erreur lors du parsing JSON des images: {e}")
        image_urls = []

    output_filename = "final_video.mp4"
    client = genai.Client(api_key=api_key)
    
    # 2. Préparation des images de référence (Max 3 pour Veo)
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

    # 3. Lancement de la génération (Veo 3.1 Preview)
    print(f"Lancement de la génération avec le prompt: {prompt[:50]}...")
    
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
        config=types.GenerateVideosConfig(
            reference_images=reference_images if reference_images else None,
            duration_seconds=8,
            aspect_ratio="9:16"
        ),
    )

    # 4. Attente de la fin du traitement
    while not operation.done:
        print("Génération en cours (Veo travaille)...")
        time.sleep(15)
        operation = client.operations.get(operation)

    # 5. Sauvegarde de la vidéo
    try:
        # Récupération du résultat
        video_data = operation.result.generated_videos[0]
        
        # Le SDK permet souvent d'accéder aux bytes via l'objet vidéo
        # Sinon, on télécharge via le nom du fichier généré
        print(f"Téléchargement de la vidéo finale...")
        
        # Méthode la plus stable pour le SDK actuel :
        with open(output_filename, "wb") as f:
            # On récupère les bytes du fichier généré sur Google Cloud
            file_content = client.files.download(file=video_data.video.name)
            f.write(file_content)
            
        print(f"Succès ! Vidéo sauvegardée sous {output_filename}")
        
    except Exception as e:
        print(f"Erreur lors de la récupération de la vidéo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_video_with_refs()
