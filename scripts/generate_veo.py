import time
import sys
import json
import requests
from google import genai
from google.genai import types


def extract_parts(full_prompt):
    """Découpe le prompt en utilisant le délimiteur ###"""
    parts = {"scenario": "", "voice_over": "", "music": ""}
    segments = full_prompt.split("###")
    for segment in segments:
        segment = segment.strip()
        if segment.upper().startswith("SCENARIO:"):
            parts["scenario"] = segment.replace("SCENARIO:", "").strip()
        elif segment.upper().startswith("VOICE-OVER:"):
            parts["voice_over"] = segment.replace("VOICE-OVER:", "").strip()
        elif segment.upper().startswith("MUSIC:"):
            parts["music"] = segment.replace("MUSIC:", "").strip()
    if not parts["voice_over"]:
        parts["voice_over"] = full_prompt
    return parts


def split_text_into_two(text):
    """Divise en 2 parties (première ~57%)"""
    words = text.split()
    n = len(words)
    if n == 0:
        return "", ""
    split_point = int(n * 0.57)
    part1 = " ".join(words[:split_point])
    part2 = " ".join(words[split_point:])
    return part1, part2


def wait_for_op(client, operation_name: str, timeout_sec=1200):
    """Polling propre pour les ops longues (string name)"""
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout_sec:
            raise TimeoutError(f"Timeout après {timeout_sec}s pour {operation_name}")
        op = client.operations.get(operation_name)
        if op.done:
            break
        print(f"En attente... ({int(time.time() - start_time)}s écoulés)")
        time.sleep(15)
    if op.error:
        raise RuntimeError(f"Erreur dans l'opération : {op.error}")
    # Priorité à .response (plus courant dans SDK récent)
    if hasattr(op, 'response') and op.response and op.response.generated_videos:
        return op.response.generated_videos[0].video
    if op.result and hasattr(op.result, 'generated_videos') and op.result.generated_videos:
        return op.result.generated_videos[0].video
    raise ValueError("Aucune vidéo générée")


def generate_video_with_refs():
    if len(sys.argv) < 4:
        print("Usage: python script.py <api_key> \"prompt\" '[\"url1\",\"url2\"]'")
        sys.exit(1)

    api_key = sys.argv[1]
    full_prompt = sys.argv[2]
    try:
        image_urls = json.loads(sys.argv[3])
    except:
        image_urls = []

    output_filename = "final_video.mp4"  # Cohérent avec ton webhook curl
    client = genai.Client(api_key=api_key)

    extracted = extract_parts(full_prompt)
    visual_scenario = extracted["scenario"]

    v1, v2 = split_text_into_two(extracted["voice_over"])
    print(f"Partie 1 (≈8s) : {v1}")
    print(f"Partie 2 (≈7s) : {v2}")

    # Chargement images → exactement comme ton ancien code qui marchait
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
                    print(f"Image de référence chargée : {url}")
                else:
                    print(f"Échec chargement {url} (status {img_response.status_code})")
            except Exception as e:
                print(f"Erreur chargement image {url}: {e}")

    # ───────────────────────────────────────────────
    # ÉTAPE 1 : Génération initiale (comme avant : pas de resolution)
    # ───────────────────────────────────────────────
    print("\nGénération partie 1 (0–8s)...")
    prompt_1 = (
        f"STRICT INSTRUCTION: Commercial beginning. VISUAL: {visual_scenario}. "
        f"AUDIO: The narrator speaks ONLY this text: '{v1}'"
    )
    op_name_1 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt_1,
        config=types.GenerateVideosConfig(
            reference_images=reference_images if reference_images else None,
            duration_seconds=8,
            aspect_ratio="16:9"  # Obligatoire pour extensions
        ),
    )
    current_video = wait_for_op(client, op_name_1)
    if not current_video:
        print("Échec partie 1")
        sys.exit(1)

    time.sleep(20)

    # ───────────────────────────────────────────────
    # ÉTAPE 2 : Extension (resolution="720p" comme avant)
    # ───────────────────────────────────────────────
    print("\nExtension partie 2 (8–15s)...")
    prompt_2 = (
        f"CONTINUE THE EXACT SAME SCENE SMOOTHLY. VISUAL: {visual_scenario}. "
        f"AUDIO: Continue narration seamlessly with ONLY this: '{v2}'"
    )
    op_name_2 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        video=current_video,
        prompt=prompt_2,
        config=types.GenerateVideosConfig(
            resolution="720p"
        ),
    )
    current_video = wait_for_op(client, op_name_2)
    if not current_video:
        print("Échec extension")
        sys.exit(1)

    # Sauvegarde
    try:
        print("\nTéléchargement...")
        file_content = client.files.download(file=current_video.uri)
        with open(output_filename, "wb") as f:
            f.write(file_content)
        print(f"Succès ! Vidéo : {output_filename}")
    except Exception as e:
        print(f"Erreur téléchargement : {e}")
        sys.exit(1)


if __name__ == "__main__":
    generate_video_with_refs()
