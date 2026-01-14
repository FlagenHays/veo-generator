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
        """Attend la fin de l'opération et vérifie s'il y a un résultat"""
        while not op.done:
            time.sleep(20)
            op = client.operations.get(op)
        
        if op.result and hasattr(op.result, 'generated_videos') and op.result.generated_videos:
            return op.result.generated_videos[0].video
        else:
            print(f"Erreur: L'opération s'est terminée sans vidéo. Détails: {op.error if hasattr(op, 'error') else 'Inconnu'}")
            return None

    # --- ÉTAPE 1 : 0-8s ---
    print("Étape 1/3: Initialisation (0-8s)...")
    prompt_1 = f"STRICT VISUAL FIDELITY. Use the exact object from references. START COMMERCIAL. Voice-over starts. Script: {full_prompt}"
    
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

    # PAUSE OBLIGATOIRE (Pour éviter le 429 RESOURCE_EXHAUSTED)
    print("Pause de sécurité (40s)...")
    time.sleep(40)

    # --- ÉTAPE 2 : 8-15s ---
    print("Étape 2/3: Extension (8-15s)...")
    prompt_2 = f"CONTINUE scene. KEEP bag design. DO NOT REPEAT start of voice-over. Part 2 of script: {full_prompt}"
    
    op2 = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        video=current_video,
        prompt=prompt_2,
        config=types.GenerateVideosConfig(resolution="720p")
    )
    current_video = wait_for_op(op2)
    if not current_video: sys.exit(1)

    print("Pause de sécurité (40s)...")
    time.sleep(40)

    # --- ÉTAPE 3 : 15-22s ---
    print("Étape 3/3: Finalisation (15-22s)...")
    # On donne une fin très claire pour éviter que l'IA ne boucle
    prompt_3 = f"FINAL PART. Finish French narration and music. End with logo. Final script part: {full_prompt[-150:]}"
    
    op3 = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        video=current_video,
        prompt=prompt_3,
        config=types.GenerateVideosConfig(resolution="720p")
    )
    current_video = wait_for_op(op3)
    if not current_video: sys.exit(1)

    # --- SAUVEGARDE FINALE ---
    try:
        print("Téléchargement de la vidéo finale...")
        file_content = client.files.download(file=current_video.uri)
        with open(output_filename, "wb") as f:
            f.write(file_content)
        print("Succès total !")
    except Exception as e:
        print(f"Erreur téléchargement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_video_with_refs()
