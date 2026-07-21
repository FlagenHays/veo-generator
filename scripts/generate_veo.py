import time
import sys
import json
import requests
import math
from google import genai
from google.genai import types

def extract_parts(full_prompt):
    """Découpe le prompt en utilisant le délimiteur ###"""
    parts = {"scenario": "", "voice_over": "", "music": "", "aspect_ratio": "9:16"}
    segments = full_prompt.split("###")
    for segment in segments:
        segment = segment.strip()
        if segment.upper().startswith("SCENARIO:"):
            parts["scenario"] = segment[len("SCENARIO:"):].strip()
        elif segment.upper().startswith("VOICE-OVER:"):
            parts["voice_over"] = segment[len("VOICE-OVER:"):].strip()
        elif segment.upper().startswith("MUSIC:"):
            parts["music"] = segment[len("MUSIC:"):].strip()
    
    if not parts["voice_over"]:
        parts["voice_over"] = full_prompt
    return parts

def split_text_into_two(text):
    """
    Divise le texte en 2 parties.
    Première partie légèrement plus longue pour l'impact initial.
    """
    words = text.split()
    n = len(words)
    if n == 0:
        return "", ""
    mid = math.ceil(n / 2)
    part1 = " ".join(words[:mid])
    part2 = " ".join(words[mid:])
    return part1, part2

def load_reference_images(image_urls):
    """Charge les images de référence produit."""
    reference_images = []
    if not isinstance(image_urls, list):
        return reference_images
    
    for url in image_urls[:3]:
        try:
            img_response = requests.get(url, timeout=15)
            if img_response.status_code == 200:
                ref = types.VideoGenerationReferenceImage(
                    image=types.Image(bytes=img_response.content, mime_type="image/jpeg"),
                    reference_type="ASSET"
                )
                reference_images.append(ref)
                print(f"Image de référence chargée: {url[:60]}...")
        except Exception as e:
            print(f"Erreur chargement image {url[:60]}: {e}")
    
    return reference_images

def wait_for_operation(client, op, timeout_seconds=600):
    """Attend la fin d'une opération avec timeout."""
    elapsed = 0
    while not op.done:
        time.sleep(20)
        elapsed += 20
        if elapsed > timeout_seconds:
            print(f"Timeout après {timeout_seconds}s")
            return None
        op = client.operations.get(op)
        print(f"En attente... {elapsed}s écoulées")
    
    if op.result and hasattr(op.result, 'generated_videos') and op.result.generated_videos:
        return op.result.generated_videos[0].video
    
    if hasattr(op, 'error') and op.error:
        print(f"Erreur opération: {op.error}")
    return None

def generate_video_with_refs():
    if len(sys.argv) < 4:
        print("Usage: python generate_veo.py <api_key> <prompt> <image_urls_json> [aspect_ratio]")
        sys.exit(1)

    api_key       = sys.argv[1]
    full_prompt   = sys.argv[2]
    image_urls    = json.loads(sys.argv[3])
    aspect_ratio  = sys.argv[4] if len(sys.argv) > 4 else "9:16"
    output_filename = "final_video.mp4"
    
    client = genai.Client(api_key=api_key)

    # ── 1. Extraction et segmentation ────────────────────────────────────────
    extracted      = extract_parts(full_prompt)
    visual_scenario = extracted["scenario"]
    
    # Division optimisée de la voix-off
    v1, v2 = split_text_into_two(extracted["voice_over"])
    
    print(f"Format: {aspect_ratio}")
    print(f"Scénario: {visual_scenario[:100]}...")
    print(f"Voix-off partie 1 ({len(v1.split())} mots): {v1[:80]}...")
    print(f"Voix-off partie 2 ({len(v2.split())} mots): {v2[:80]}...")

    # ── 2. Chargement des images de référence ────────────────────────────────
    reference_images = load_reference_images(image_urls)
    print(f"{len(reference_images)} image(s) de référence chargée(s)")

    # ── 3. Génération partie 1 (0-8s) ────────────────────────────────────────
    # Le prompt de la partie 1 contient l'accroche visuelle la plus puissante
    # Les 2-3 premières secondes doivent STOPPER le scroll immédiatement
    print(f"\nÉtape 1/2 — 8 premières secondes | Mots voix-off: {len(v1.split())}")
    
    prompt_1 = (
        f"VERTICAL {aspect_ratio} FORMAT — Mobile Stories/Reels. "
        f"HOOK IN FIRST 2 SECONDS: Create an immediate visual impact that stops scrolling. "
        f"VISUAL SCENARIO: {visual_scenario}. "
        f"AUDIO: The narrator speaks ONLY this French text: '{v1}'. "
        f"Premium cinematic quality. No floating text. Brand name fixed bottom center."
    )
    
    op1 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt_1,
        config=types.GenerateVideosConfig(
            reference_images=reference_images if reference_images else None,
            duration_seconds=8,
            aspect_ratio=aspect_ratio,
        ),
    )
    
    current_video = wait_for_operation(client, op1)
    if not current_video:
        print("Échec étape 1")
        sys.exit(1)
    
    print("Étape 1 réussie. Pause avant étape 2...")
    time.sleep(30)

    # ── 4. Génération partie 2 (8-15s) — Extension et conclusion ─────────────
    print(f"\nÉtape 2/2 — Extension 7 secondes | Mots voix-off: {len(v2.split())}")
    
    # La partie 2 conclut avec le CTA et l'animation du nom de marque
    prompt_2 = (
        f"CONTINUE SEAMLESSLY from previous scene. {aspect_ratio} vertical format. "
        f"FINAL PART: Build to emotional climax and clear call-to-action. "
        f"End with brand name animation (elegant reveal). "
        f"VISUAL: Continue — {visual_scenario}. "
        f"AUDIO: Narrator concludes with ONLY: '{v2}'. "
        f"Smooth invisible transition. No floating text."
    )
    
    op2 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        video=current_video,
        prompt=prompt_2,
        config=types.GenerateVideosConfig(resolution="720p"),
    )
    
    current_video = wait_for_operation(client, op2)
    if not current_video:
        print("Échec étape 2")
        sys.exit(1)

    # ── 5. Sauvegarde finale ──────────────────────────────────────────────────
    try:
        print("\nTéléchargement de la vidéo finale...")
        file_content = client.files.download(file=current_video.uri)
        with open(output_filename, "wb") as f:
            f.write(file_content)
        print(f"Succès ! Vidéo {aspect_ratio} de 15s générée → {output_filename}")
    except Exception as e:
        print(f"Erreur sauvegarde: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_video_with_refs()
