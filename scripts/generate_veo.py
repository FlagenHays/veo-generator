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


def load_reference_images(client, image_urls):
    """
    Upload les images via client.files.upload et retourne des Part avec file_uri.
    Compatible avec google-genai sans dépendre de types.Image ou types.VideoGenerationReferenceImage.
    """
    parts = []
    if not isinstance(image_urls, list):
        return parts

    for url in image_urls[:3]:
        tmp_path = None
        try:
            img_response = requests.get(url, timeout=15)
            if img_response.status_code != 200:
                print(f"HTTP {img_response.status_code} pour {url[:60]}")
                continue

            content_type = img_response.headers.get('Content-Type', 'image/jpeg')
            mime_type = content_type.split(';')[0].strip()
            if mime_type not in ('image/jpeg', 'image/png', 'image/webp'):
                mime_type = 'image/jpeg'

            ext = '.png' if 'png' in mime_type else '.jpg'

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(img_response.content)
                tmp_path = tmp.name

            uploaded = client.files.upload(
                file=tmp_path,
                config={"mime_type": mime_type}
            )

            # On retourne l'URI — utilisé dans le prompt texte directement
            parts.append(uploaded.uri)
            print(f"Image uploadée: {uploaded.uri} ({url[:50]}...)")

        except Exception as e:
            print(f"Erreur image {url[:60]}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return parts


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

    if op.result and hasattr(op.result, 'generated_videos') and op.result.generated_videos:
        return op.result.generated_videos[0].video

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
    voice_over      = extracted["voice_over"]

    print(f"Format: {aspect_ratio}")
    print(f"Scénario: {visual_scenario[:120]}...")
    print(f"Voix-off complète ({len(voice_over.split())} mots): {voice_over[:100]}...")

    # ── 2. Upload images de référence ─────────────────────────────────────────
    uploaded_uris = load_reference_images(client, image_urls)
    print(f"{len(uploaded_uris)} image(s) de référence uploadée(s)")

    # ── 3. Construction du prompt final ───────────────────────────────────────
    # On intègre les URIs des images directement dans le prompt texte
    # car VideoGenerationReferenceImage est instable selon la version de la lib
    image_context = ""
    if uploaded_uris:
        image_context = (
            f"\nREFERENCE IMAGES (use these as visual reference for the product/subject): "
            + ", ".join(uploaded_uris)
            + "\n"
        )

    prompt_final = (
        f"VERTICAL {aspect_ratio} FORMAT — Mobile Stories/Reels. "
        f"DURATION: 15 seconds total.\n"
        f"{image_context}"
        f"HOOK IN FIRST 2 SECONDS: Create an immediate visual impact that stops scrolling.\n"
        f"VISUAL SCENARIO (follow precisely): {visual_scenario}\n"
        f"AUDIO: The narrator speaks ONLY this French text throughout the video: '{voice_over}'.\n"
        f"QUALITY: Premium cinematic. Photorealistic. Smooth camera movements.\n"
        f"FORBIDDEN: floating text overlay, watermark, subtitles.\n"
        f"Brand name fixed bottom center only at the very end (last 2 seconds)."
    )

    print(f"\nGénération vidéo unique 15s en {aspect_ratio}...")

    # ── 4. Génération unique 15s ───────────────────────────────────────────────
    # On ne fait PAS d'extension vidéo (étape 2) car veo-3.1-fast ne supporte
    # pas l'extension en 9:16 — erreur API confirmée deux fois.
    op = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt_final,
        config=types.GenerateVideosConfig(
            duration_seconds=8,   # max supporté par veo-3.1-fast en une passe
            aspect_ratio=aspect_ratio,
        ),
    )

    final_video = wait_for_operation(client, op)
    if not final_video:
        print("Échec génération vidéo")
        sys.exit(1)

    # ── 5. Sauvegarde ─────────────────────────────────────────────────────────
    try:
        print("\nTéléchargement de la vidéo finale...")
        file_content = client.files.download(file=final_video.uri)
        with open(output_filename, "wb") as f:
            f.write(file_content)
        print(f"Succès ! Vidéo {aspect_ratio} générée → {output_filename}")
    except Exception as e:
        print(f"Erreur sauvegarde: {e}")
        sys.exit(1)


if __name__ == "__main__":
    generate_video_with_refs()
