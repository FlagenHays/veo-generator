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
    """
    Crop centre 9:16 depuis une source 16:9.
    On extrait la bande centrale : largeur = hauteur * 9/16, centrée horizontalement.
    Filtre scale2ref évité — crop simple et précis.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
        "-c:v", "libx264",
        "-preset", "slow",        # meilleure qualité d'encodage
        "-crf", "16",             # quasi-lossless (0=parfait, 18=excellent, 23=défaut)
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

    print(f"Stratégie: 16:9 SDK → 1 extension 16:9 → crop ffmpeg CRF16 → 9:16")
    print(f"Scénario: {visual_scenario[:120]}...")
    print(f"V1 ({len(v1.split())} mots): {v1[:80]}...")
    print(f"V2 ({len(v2.split())} mots): {v2[:80]}...")

    # ── 2. Images de référence ────────────────────────────────────────────────
    reference_images = load_reference_images(image_urls)
    print(f"{len(reference_images)} image(s) de référence chargée(s)")

    # ── 3. Génération partie 1 — 8s en 16:9 ──────────────────────────────────
    # RÈGLE DE COMPOSITION :
    # La vidéo finale sera rognée à la bande centrale (56% de la largeur 16:9).
    # Tout sujet, produit ou personnage DOIT rester dans cette bande centrale.
    # Les 22% gauche et droite seront coupés — ils ne doivent contenir que
    # du décor, fond ou éléments secondaires.
    # Le prompt doit décrire la scène COMME SI elle était filmée en 9:16 :
    # cadrages serrés, portrait vertical, plan américain ou rapproché.
    print(f"\nÉtape 1/2 — 8s en 16:9 (composition 9:16 native)")

    prompt_1 = (
        # ── Contrainte de format et de crop ──────────────────────────────────
        "TECHNICAL REQUIREMENT — READ BEFORE GENERATING: "
        "This 16:9 video will be cropped to 9:16 by cutting the left 22% and right 22% of the frame. "
        "Only the CENTER 56% of the horizontal width will survive. "
        "THEREFORE: compose this scene AS IF you are shooting in 9:16 portrait format. "
        "Every subject, person, and product MUST be framed entirely within the center 56% of the frame at ALL times. "
        "Use portrait-style framing: tight vertical shots, close-ups, medium shots (waist-up), "
        "vertical movement (top-to-bottom), never horizontal panning. "
        "Camera movements allowed: slow zoom in/out, tilt up/down, gentle vertical tracking. "
        "FORBIDDEN: horizontal pan, subjects moving left/right out of center, wide landscape shots. "
        # ── Qualité ───────────────────────────────────────────────────────────
        "QUALITY: 4K cinematic, ultra-sharp focus on subjects, no motion blur, "
        "no grain, no artifacts, clean crisp image. "
        "Professional studio or location lighting. Photorealistic. "
        "FORBIDDEN: floating text, watermark, subtitles, logo overlays, cartoon, CGI. "
        # ── Scénario ─────────────────────────────────────────────────────────
        f"VISUAL SCENARIO (first half): {visual_scenario}. "
        "HOOK: immediate visual impact in first 2 seconds — draw viewer's eye to subject. "
        # ── Audio ─────────────────────────────────────────────────────────────
        f"AUDIO: narrator speaks ONLY this French text: '{v1}'."
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

    print("Étape 1 réussie. Pause 30s...")
    time.sleep(30)

    # ── 4. UNE SEULE extension — 8s en 16:9 ──────────────────────────────────
    print(f"\nÉtape 2/2 — Extension 8s en 16:9 (même contrainte de composition)")

    prompt_2 = (
        # ── Contrainte de crop — répétée pour l'extension ────────────────────
        "CONTINUE SEAMLESSLY FROM PREVIOUS CLIP. "
        "SAME TECHNICAL REQUIREMENT: video will be cropped to center 56% horizontally (9:16 final). "
        "ALL subjects MUST remain in the center 56% of frame. "
        "Portrait-style framing: tight vertical shots, no horizontal pan. "
        "Camera: slow zoom or gentle vertical movement only. "
        # ── Qualité ───────────────────────────────────────────────────────────
        "Ultra-sharp, clean image, no motion blur, photorealistic, cinematic lighting. "
        "FORBIDDEN: floating text, watermark, subtitles, horizontal pan. "
        # ── Scénario ─────────────────────────────────────────────────────────
        f"VISUAL: build to climax — {visual_scenario}. "
        "End with strong CTA moment: subject looks directly at camera, confident, engaging. "
        "Brand/product clearly visible at center bottom of frame (within safe zone). "
        # ── Audio ─────────────────────────────────────────────────────────────
        f"AUDIO: narrator concludes ONLY: '{v2}'."
    )

    op2 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt_2,
        video=video1,
        config=types.GenerateVideosConfig(
            duration_seconds=8,
            resolution="720p",
            # aspect_ratio absent — hérité 16:9 de la vidéo source
        ),
    )

    video2 = wait_for_op(client, op2)

    if not video2 or not download_video(client, video2, "part2_16x9.mp4"):
        print("Extension échouée — crop partie 1 uniquement (8s)")
        crop_to_portrait("part1_16x9.mp4", output_file)
        sys.exit(0)

    # ── 5. Concaténation part1 + part2 ───────────────────────────────────────
    print("\nConcaténation des deux parties...")
    with open("filelist.txt", "w") as f:
        f.write("file 'part1_16x9.mp4'\n")
        f.write("file 'part2_16x9.mp4'\n")

    concat = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", "filelist.txt", "-c", "copy", "raw_16x9.mp4"],
        capture_output=True, text=True
    )

    if concat.returncode != 0:
        print(f"Erreur concat: {concat.stderr[-300:]}")
        crop_to_portrait("part2_16x9.mp4", output_file)
        sys.exit(0)

    print("Concaténation réussie — ~16s en 16:9")

    # ── 6. Crop final → 9:16 avec encodage haute qualité ─────────────────────
    print("\nCrop 16:9 → 9:16 (CRF 16, preset slow)...")
    if not crop_to_portrait("raw_16x9.mp4", output_file):
        crop_to_portrait("part1_16x9.mp4", output_file)

    print(f"\nSuccès ! Vidéo 9:16 ~16s → {output_file}")


if __name__ == "__main__":
    generate_video_with_refs()
