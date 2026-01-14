import time
import sys
import json
import requests
import re
from google import genai
from google.genai import types

def extract_parts(full_prompt):
    """Découpe le prompt en utilisant le délimiteur ###"""
    parts = {
        "scenario": "",
        "voice_over": "",
        "music": ""
    }
    
    # Découpage par le délimiteur
    segments = full_prompt.split("###")
    
    for segment in segments:
        segment = segment.strip()
        if segment.upper().startswith("SCENARIO:"):
            parts["scenario"] = segment.replace("SCENARIO:", "").strip()
        elif segment.upper().startswith("VOICE-OVER:"):
            parts["voice_over"] = segment.replace("VOICE-OVER:", "").strip()
        elif segment.upper().startswith("MUSIC:"):
            parts["music"] = segment.replace("MUSIC:", "").strip()
            
    # Si le découpage échoue, on prend tout par défaut
    if not parts["voice_over"]:
        parts["voice_over"] = full_prompt
        
    return parts

def generate_video_with_refs():
    if len(sys.argv) < 4:
        sys.exit(1)

    api_key = sys.argv[1]
    full_prompt = sys.argv[2]
    image_urls = json.loads(sys.argv[3])
    output_filename = "final_video.mp4"
    client = genai.Client(api_key=api_key)

    # Extraction propre des composants
    extracted = extract_parts(full_prompt)
    voice_script = extracted["voice_over"]
    visual_scenario = extracted["scenario"]

    # 1. Chargement des images de référence (Personnages/Produits)
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
        
        if op.result and hasattr(op.result, 'generated_videos') and op.result.generated_videos:
            return op.result.generated_videos[0].video
        else:
            return None

    # --- ÉTAPE 1 : 0-8s ---
    print(f"Étape 1/3: Narration détectée : {voice_script[:60]}...")
    
    # On isole strictement la voix pour l'IA
    prompt_1 = (
        f"STRICT INSTRUCTION: This is a professional commercial. "
        f"VISUAL SCENE: {visual_scenario}. "
        f"AUDIO VOICE-OVER: The narrator must ONLY speak the following text in French: '{voice_script}'"
    )
    
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

    # Pause pour éviter 429
    time.sleep(40)

    # --- ÉTAPE 2 : 8-15s ---
    print("Étape 2/3: Extension de la vidéo...")
    prompt_2 = f"CONTINUE THE SCENE. Visuals: {visual_scenario}. Voice-over: Continue speaking the script: '{voice_script}'"
    
    op2 = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        video=current_video,
        prompt=prompt_2,
        config=types.GenerateVideosConfig(resolution="720p")
    )
    current_video = wait_for_op(op2)
    if not current_video: sys.exit(1)

    time.sleep(40)

    # --- ÉTAPE 3 : 15-22s ---
    print("Étape 3/3: Finalisation et fermeture...")
    prompt_3 = f"FINAL PART. Visuals: {visual_scenario}. Complete the narration: '{voice_script}'"
    
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
        file_content = client.files.download(file=current_video.uri)
        with open(output_filename, "wb") as f:
            f.write(file_content)
        print("Succès !")
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    generate_video_with_refs()
