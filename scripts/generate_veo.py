import time
import sys
import json
import requests
from google import genai
from google.genai import types

def generate_video_with_refs():
    # Arguments passés par GitHub Action
    api_key = sys.argv[1]
    prompt = sys.argv[2]
    image_urls = json.loads(sys.argv[3])
    output_filename = "final_video.mp4"

    client = genai.Client(api_key=api_key)
    
    # Préparation des images de référence (Max 3)
    reference_images = []
    for url in image_urls[:3]:
        try:
            img_response = requests.get(url, timeout=15)
            if img_response.status_code == 200:
                ref = types.VideoGenerationReferenceImage(
                    image=types.Image(bytes=img_response.content, mime_type="image/jpeg"),
                    reference_type="ASSET"
                )
                reference_images.append(ref)
        except Exception as e:
            print(f"Erreur image: {e}")

    # Lancement Veo 3.1
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
        config=types.GenerateVideosConfig(
            reference_images=reference_images,
            duration_seconds=8,
            aspect_ratio="9:16"
        ),
    )

    while not operation.done:
        print("Génération en cours...")
        time.sleep(10)
        operation = client.operations.get(operation)

    # Téléchargement
    video_data = operation.result.generated_videos[0]
    video_file = client.files.get(name=video_data.video.name)
    content = client.files.download(file=video_file)
    
    with open(output_filename, "wb") as f:
        f.write(content)

if __name__ == "__main__":
    generate_video_with_refs()