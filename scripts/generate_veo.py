import time
import sys
import json
import requests
import math
import subprocess
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


def load_reference_images(image_urls):
    reference_images = []
    if not isinstance(image_urls, list):
        return reference_images

    for url in image_urls[:3]:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code} pour {url[:60]}")
                continue

            content_type = resp.headers.get('Content-Type', 'image/jpeg')
            mime_type = content_type.split(';')[0].strip()
            if mime_type not in ('image/jpeg', 'image/png', 'image/webp'):
                mime_type = 'image/jpeg'

            ref = types.VideoGenerationReferenceImage(
                image=types.Image(bytes=resp.content, mime_type=mime_type),
                reference_type="ASSET"
            )
            reference_images.append(ref)
            print(f"Image de référence chargée: {url[:60]}...")

        except Exception as e:
            print(f"Erreur image {url[:60]}: {e}")

    return reference_images


def wait_for_op(client, op, timeout_seconds=700):
    elapsed = 0
    while not op.done:
        time.sleep(20)
        elapsed += 20
        if elapsed > timeout_seconds:
            print(f"Timeout après {timeout_seconds}s")
            return None
        op = client.operations.get(op)
        print(f"En attente... {elapsed}s")

    if op.result and hasattr(op.result, 'generated_videos') and op.result.generated_videos:
        return op.result.generated_videos[0].video
    if hasattr(op, 'error') and op.error:
        print(f"Erreur: {op.error}")
    return None


def download_video(client, video_obj, output_path):
    try:
        file_content = client.files.download(file=video_obj.uri)
        with open(output_path, "wb") as f:
            f.write(file_content)
        import os
        size = os.path.getsize(output_path)
        print(f"Téléchargé: {output_path} ({size} bytes)")
        return size > 10000
    except Exception as e:
        print(f"Erreur téléchargement: {e}")
        return False


def crop_to_portrait(input_file, output_file):
    """Crop 16:9 → 9:16 en gardant le centre. Qualité inchangée."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
        "-c:a", "copy",
        output_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Erreur ffmpeg crop: {result.stderr[-300:]}")
        return False
    import os
    size = os.path.getsize(output_file)
    print(f"Crop 9:16 réussi → {output_file} ({size} bytes)")
    return True


def generate_video_with_refs():
    if len(sys.argv) < 4:
        print("Usage: python generate_veo.py <api_key> <prompt> <image_urls_json>")
        sys.exit(1)

    api_key     = sys.argv[1]
    full_prompt = sys.argv[2]
    image_urls  = json.loads(sys.argv[3])
    output_file = "final_video.mp4"

    client = genai.Client(api_key=api_key)

    # ── 1. Extraction ─────────────────────────────────────────────────────────
    extracted       = extract_parts(full_prompt)
    visual_scenario = extracted["scenario"] or full_prompt
    v1, v2          = split_text_into_two(extracted["voice_over"])

    print(f"Stratégie: 8s + extension 7s en 16:9 → crop ffmpeg → 9:16 (15s total)")
    print(f"Scénario: {visual_scenario[:120]}...")
    print(f"V1 ({len(v1.split())} mots): {v1[:80]}...")
    print(f"V2 ({len(v2.split())} mots): {v2[:80]}...")

    # ── 2. Images de référence ────────────────────────────────────────────────
    reference_images = load_reference_images(image_urls)
    print(f"{len(reference_images)} image(s) de référence chargée(s)")

    # ── 3. Génération partie 1 — 8s en 16:9 ──────────────────────────────────
    print(f"\nÉtape 1/2 — 8s en 16:9")

    prompt_1 = (
        f"LANDSCAPE 16:9 FORMAT. "
        f"CRITICAL COMPOSITION: Keep ALL subjects and products STRICTLY CENTERED horizontally. "
        f"Left and right 25% of frame must stay background only "
        f"(video will be cropped to 9:16 portrait, only center kept). "
        f"HOOK IN FIRST 2 SECONDS: immediate visual impact that stops scrolling. "
        f"VISUAL SCENARIO: {visual_scenario}. "
        f"AUDIO: narrator speaks ONLY this French text: '{v1}'. "
        f"Premium cinematic. Photorealistic. No floating text overlay. No watermark. "
        f"Slow sensual camera movements. All action stays center frame."
    )

    op1 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt_1,
        config=types.GenerateVideosConfig(
            reference_images=reference_images if reference_images else None,
            duration_seconds=8,
            aspect_ratio="16:9",
            resolution="720p",
        ),
    )

    video1 = wait_for_op(client, op1)
    if not video1:
        print("Échec étape 1")
        sys.exit(1)

    if not download_video(client, video1, "part1_16x9.mp4"):
        print("Échec téléchargement partie 1")
        sys.exit(1)

    print("Étape 1 réussie. Pause 30s avant extension...")
    time.sleep(30)

    # ── 4. Extension unique — 7s supplémentaires en 16:9 ─────────────────────
    # Une seule extension (pas deux) — exactement comme l'ancien code
    # aspect_ratio absent → hérité 16:9 de la vidéo source → pas d'erreur 9:16
    print(f"\nÉtape 2/2 — Extension 7s en 16:9")

    prompt_2 = (
        f"CONTINUATION OF THE PREVIOUS CLIP. LANDSCAPE 16:9. "
        f"ALL subjects STRICTLY CENTERED horizontally. Left/right 25% background only. "
        f"AUDIO: The narrator speaks ONLY these new words (continuation): '{v2}'. "
        f"Build to emotional climax and CTA. Brand name elegant reveal bottom center at the very end. "
        f"VISUAL: continue seamlessly — {visual_scenario}. "
        f"No floating text overlay. Smooth invisible transition from previous clip."
    )

    op2 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt_2,
        video=video1,
        config=types.GenerateVideosConfig(
            duration_seconds=7,
            resolution="720p",
            # aspect_ratio intentionnellement absent — hérité 16:9 de video1
        ),
    )

    video2 = wait_for_op(client, op2)

    if not video2 or not download_video(client, video2, "raw_16x9.mp4"):
        print("Extension échouée — crop partie 1 uniquement (8s)")
        crop_to_portrait("part1_16x9.mp4", output_file)
        sys.exit(0)

    # ── 5. Crop final 16:9 → 9:16 ────────────────────────────────────────────
    # La vidéo "raw_16x9.mp4" contient la vidéo étendue (15s en 16:9)
    # Le crop garde uniquement le centre → parfait pour Stories/Reels
    print("\nCrop 16:9 → 9:16 (centre)...")
    if not crop_to_portrait("raw_16x9.mp4", output_file):
        print("Erreur crop vidéo étendue — fallback crop partie 1")
        crop_to_portrait("part1_16x9.mp4", output_file)

    print(f"\nSuccès ! Vidéo 9:16 ~15s → {output_file}")


if __name__ == "__main__":
    generate_video_with_refs()
