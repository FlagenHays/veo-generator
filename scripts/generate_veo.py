import time
import sys
import json
import requests
import math
import tempfile
import os
from google import genai
from google.genai import types


def extract_parts(full_prompt):
    parts = {"scenario": "", "voice_over": "", "music": ""}
    segments = full_prompt.split("###")
    for segment in segments:
        segment = segment.strip()
        upper = segment.upper()
        if upper.startswith("SCENARIO:"):
            parts["scenario"] = segment[len("SCENARIO:"):].strip()
        elif upper.startswith("VOICE-OVER:"):
            parts["voice_over"] = segment[len("VOICE-OVER:"):].strip()
        elif upper.startswith("MUSIC:"):
            parts["music"] = segment[len("MUSIC:"):].strip()
    if not parts["voice_over"]:
        parts["voice_over"] = full_prompt
    return parts


def split_text_into_two(text):
    words = text.split()
    n = len(words)
    if n == 0:
        return "", ""
    mid = math.ceil(n / 2)
    return " ".join(words[:mid]), " ".join(words[mid:])


def load_reference_images(client, image_urls):
    """
    Charge les images et retourne une liste de VideoGenerationReferenceImage.
    google-genai 2.13.0 : on passe les bytes bruts directement sans types.Image wrapper.
    """
    reference_images = []
    if not isinstance(image_urls, list):
        return reference_images

    for url in image_urls[:3]:
        try:
            img_response = requests.get(url, timeout=15)
            if img_response.status_code != 200:
                print(f"HTTP {img_response.status_code} pour {url[:60]}")
                continue

            content_type = img_response.headers.get('Content-Type', 'image/jpeg')
            mime_type = content_type.split(';')[0].strip()
            if mime_type not in ('image/jpeg', 'image/png', 'image/webp'):
                mime_type = 'image/jpeg'

            # google-genai 2.13.0 : VideoGenerationReferenceImage accepte
            # image_bytes et mime_type directement (pas de types.Image wrapper)
            ref = types.VideoGenerationReferenceImage(
                image_bytes=img_response.content,
                mime_type=mime_type,
                reference_type="ASSET",
            )
            reference_images.append(ref)
            print(f"Image de référence chargée: {url[:60]}...")

        except Exception as e:
            print(f"Erreur image {url[:60]}: {e}")

    return reference_images


def wait_for_operation(client, op, timeout_seconds=700):
    elapsed = 0
    while not op.done:
        time.sleep(20)
        elapsed += 20
        if elapsed > timeout_seconds:
            print(f"Timeout après {timeout_seconds}s")
            return None
        op = client.operations.get(op)
        print(f"En attente... {elapsed}s écoulées")

    if op.response and hasattr(op.response, 'generated_videos') and op.response.generated_videos:
        return op.response.generated_videos[0]

    if hasattr(op, 'error') and op.error:
        print(f"Erreur opération: {op.error}")
    return None


def generate_video_with_refs():
    if len(sys.argv) < 4:
        print("Usage: python generate_veo.py <api_key> <prompt> <image_urls_json> [aspect_ratio]")
        sys.exit(1)

    api_key         = sys.argv[1]
    full_prompt     = sys.argv[2]
    image_urls      = json.loads(sys.argv[3])
    aspect_ratio    = sys.argv[4] if len(sys.argv) > 4 else "9:16"
    output_filename = "final_video.mp4"

    client = genai.Client(api_key=api_key)

    # ── 1. Extraction ─────────────────────────────────────────────────────────
    extracted       = extract_parts(full_prompt)
    visual_scenario = extracted["scenario"] or full_prompt
    v1, v2          = split_text_into_two(extracted["voice_over"])

    print(f"Format: {aspect_ratio}")
    print(f"Scénario: {visual_scenario[:120]}...")
    print(f"Voix-off partie 1 ({len(v1.split())} mots): {v1[:80]}...")
    print(f"Voix-off partie 2 ({len(v2.split())} mots): {v2[:80]}...")

    # ── 2. Chargement images de référence ─────────────────────────────────────
    reference_images = load_reference_images(client, image_urls)
    print(f"{len(reference_images)} image(s) de référence chargée(s)")

    # ── 3. Génération partie 1 (8s) ───────────────────────────────────────────
    print(f"\nÉtape 1/2 — 8 secondes | Mots voix-off: {len(v1.split())}")

    prompt_1 = (
        f"VERTICAL {aspect_ratio} FORMAT — Mobile Stories/Reels. "
        f"HOOK IN FIRST 2 SECONDS: immediate visual impact that stops scrolling. "
        f"VISUAL SCENARIO: {visual_scenario}. "
        f"AUDIO: narrator speaks ONLY this French text: '{v1}'. "
        f"Premium cinematic. Photorealistic. No text overlay. No watermark."
    )

    op1 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt_1,
        config=types.GenerateVideosConfig(
            reference_images=reference_images if reference_images else None,
            duration_seconds=8,
            aspect_ratio=aspect_ratio,
            resolution="720p",
            # person_generation retiré — cause erreur selon la région
        ),
    )

    result1 = wait_for_operation(client, op1)
    if not result1:
        print("Échec étape 1")
        sys.exit(1)

    print("Étape 1 réussie. Pause 30s avant étape 2...")
    time.sleep(30)

    # ── 4. Extension partie 2 (8s) ────────────────────────────────────────────
    # Sans aspect_ratio (hérité), sans person_generation, résolution 720p forcée
    print(f"\nÉtape 2/2 — Extension 8s | Mots voix-off: {len(v2.split())}")

    prompt_2 = (
        f"CONTINUE SEAMLESSLY. {aspect_ratio} vertical. "
        f"Build to climax and CTA. End with brand name elegant reveal bottom center. "
        f"VISUAL: continue — {visual_scenario}. "
        f"AUDIO: narrator concludes ONLY: '{v2}'. "
        f"No text overlay. Smooth invisible transition."
    )

    op2 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt_2,
        video=result1.video,
        config=types.GenerateVideosConfig(
            duration_seconds=8,
            resolution="720p",
            # aspect_ratio intentionnellement absent — hérité de la vidéo source
        ),
    )

    result2 = wait_for_operation(client, op2)
    if not result2:
        print("Étape 2 échouée — sauvegarde partie 1 uniquement")
        result2 = result1

    # ── 5. Sauvegarde ─────────────────────────────────────────────────────────
    try:
        print("\nTéléchargement de la vidéo finale...")
        client.files.download(file=result2.video)
        result2.video.save(output_filename)
        print(f"Succès ! Vidéo {aspect_ratio} générée → {output_filename}")
    except Exception as e:
        print(f"Erreur sauvegarde finale: {e}")
        try:
            client.files.download(file=result1.video)
            result1.video.save(output_filename)
            print(f"Fallback partie 1 sauvegardée → {output_filename}")
        except Exception as e2:
            print(f"Erreur fallback: {e2}")
            sys.exit(1)


if __name__ == "__main__":
    generate_video_with_refs()
