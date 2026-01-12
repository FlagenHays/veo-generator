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
        # On décode le JSON envoyé par GitHub Actions
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
                    print(f"Image ajoutée avec succès: {url}")
            except Exception as e:
                print(f"Impossible de charger l'image {url}: {e}")

    # 3. Lancement de la génération (Veo 3.1 Preview)
    print(f"Lancement de la génération Veo avec le prompt: {prompt[:100]}...")
    
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
        config=types.GenerateVideosConfig(
            reference_images=reference_images if reference_images else None,
            duration_seconds=8,
            aspect_ratio="16:9"
        ),
    )

    # 4. Attente de la fin du traitement (Boucle de vérification)
    while not operation.done:
        print("Génération en cours (Veo travaille toujours)...")
        time.sleep(15)
        operation = client.operations.get(operation)

    # 5. Sauvegarde de la vidéo (Correction URI)
    try:
        # Récupération de l'objet vidéo généré
        generated_video = operation.result.generated_videos[0]
        
        print(f"Téléchargement de la vidéo finale depuis l'URI...")
        
        # Utilisation de l'URI pour le téléchargement (Correction du bug 'name')
        file_uri = generated_video.video.uri
        
        # Téléchargement du contenu binaire
        file_content = client.files.download(file=file_uri)
        
        # Écriture du fichier sur le disque du runner GitHub
        with open(output_filename, "wb") as f:
            f.write(file_content)
            
        print(f"Succès ! Vidéo sauvegardée localement : {output_filename}")
        
    except Exception as e:
        print(f"Erreur lors de la récupération ou sauvegarde de la vidéo: {e}")
        # Optionnel: Affiche les attributs disponibles en cas de nouvel échec
        if 'generated_video' in locals():
             print(f"Attributs disponibles dans video: {dir(generated_video.video)}")
        sys.exit(1)

if __name__ == "__main__":
    generate_video_with_refs()
