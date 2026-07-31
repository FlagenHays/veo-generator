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
    """
    Charge les images de référence depuis les URLs et retourne :
    - reference_images : liste de VideoGenerationReferenceImage pour Veo
    - raw_images       : liste de (bytes, mime_type) pour l'analyse Gemini Vision
    """
    reference_images = []
    raw_images = []

    if not isinstance(image_urls, list):
        return reference_images, raw_images

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

            img_bytes = resp.content

            # Pour Veo reference_images
            ref = types.VideoGenerationReferenceImage(
                image=types.Image(bytes=img_bytes, mime_type=mime_type),
                reference_type="ASSET"
            )
            reference_images.append(ref)

            # Pour Gemini Vision (analyse du sujet)
            raw_images.append((img_bytes, mime_type))

            print(f"Image de référence chargée: {url[:60]}...")

        except Exception as e:
            print(f"Erreur image {url[:60]}: {e}")

    return reference_images, raw_images


def analyze_subject_with_gemini(client, raw_images: list) -> str:
    """
    Utilise Gemini Vision pour analyser précisément le sujet des images de référence.
    Retourne une description dense en anglais à injecter dans le prompt Veo.

    C'est le même principe que ProcessIaCampaignComplexe pour les affiches :
    Gemini analyse d'abord le sujet, puis cette description ancre la génération.
    """
    if not raw_images:
        return ""

    print(f"Analyse Gemini Vision du sujet ({len(raw_images)} image(s))...")

    # Construction du payload multimodal
    parts = []
    for img_bytes, mime_type in raw_images:
        import base64
        parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(img_bytes).decode("utf-8")
            }
        })

    parts.append({"text":
        "You are an expert in visual description for AI video generation (Veo). "
        "Analyze the subject in these reference images with ABSOLUTE PRECISION. "
        "Your description will be injected directly into a Veo prompt to reproduce "
        "this exact subject in a video — it must be detailed enough that Veo can "
        "match the reference without ambiguity.\n\n"

        "IF PERSON(S):\n"
        "Describe: exact face shape (oval/round/square/triangular), jawline, cheekbones, forehead. "
        "Eyes: shape, exact color, size, inter-ocular distance, eyebrows (thickness, arch). "
        "Nose: shape, width. Lips: fullness, shape. "
        "Exact skin tone (use precise terms: 'deep ebony', 'warm caramel', 'golden brown', 'dark chocolate', etc). "
        "Hair: exact color, length, texture (straight/wavy/curly/coily), cut, style. "
        "Apparent age, gender. "
        "Outfit: colors, style, key elements. "
        "Expression and general silhouette. "
        "Do NOT change or approximate — describe what you literally see.\n\n"

        "IF PRODUCT(S):\n"
        "Describe: exact shape, relative dimensions. "
        "Precise colors (use HEX if possible, otherwise very specific: 'deep cobalt blue', 'matte black'). "
        "Material, texture, finish (glossy/matte/satin). "
        "Visible logo, brand name, text on packaging. "
        "Distinctive features, packaging type (bottle/box/bag/etc).\n\n"

        "RESPONSE FORMAT:\n"
        "Respond in ENGLISH only. Dense, precise, continuous text (no markdown, no headers). "
        "Start with the subject type: 'A [person/product]: ...' "
        "Maximum 250 words. Every detail matters — be exhaustive."
    })

    try:
        import urllib.request
        import os

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            print("GEMINI_API_KEY non disponible pour analyse Vision — ancrage textuel ignoré")
            return ""

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.5-pro:generateContent?key=" + api_key
        )

        payload = json.dumps({
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024}
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            description = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
            )

        if description:
            print(f"Description sujet extraite ({len(description)} chars): {description[:120]}...")
            return description
        else:
            print("Gemini Vision n'a retourné aucune description")
            return ""

    except Exception as e:
        print(f"Erreur analyse Gemini Vision: {e}")
        return ""


def build_subject_anchor(description: str) -> str:
    """
    Construit le bloc d'ancrage sujet à insérer EN PREMIER dans le prompt Veo.
    L'ancrage combine la description textuelle précise + rappel des images de référence.
    """
    if not description or not description.strip():
        # Fallback minimaliste si pas de description
        return (
            "SUBJECT FIDELITY: Use the provided reference images to reproduce the exact subject. "
            "Do NOT invent or substitute the subject. Match reference images precisely. "
        )

    return (
        "SUBJECT IDENTITY — ABSOLUTE RULE: "
        "Reproduce the subject from the reference images with MAXIMUM fidelity. "
        "The reference images show the EXACT subject that must appear in every frame. "
        "Detailed subject description: " + description.strip() + " "
        "FORBIDDEN: changing skin tone, hair, face features, product color/shape/branding. "
        "The subject must be IDENTICAL to the reference images throughout the video. "
    )


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
    Ré-encodage haute qualité CRF 16 pour éviter toute dégradation visible.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "16",
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

    # ── 1. Chargement des images de référence ─────────────────────────────────
    # reference_images → passées à Veo directement (ancrage visuel)
    # raw_images       → passées à Gemini Vision pour extraire la description textuelle
    reference_images, raw_images = load_reference_images(image_urls)
    print(f"{len(reference_images)} image(s) de référence chargée(s)")

    # ── 2. Analyse Gemini Vision du sujet ────────────────────────────────────
    # Gemini Vision décrit précisément le sujet en anglais.
    # Cette description est injectée EN PREMIER dans le prompt Veo
    # pour ancrer le sujet avec précision — en plus des images de référence.
    # Double ancrage : images + description textuelle précise = fidélité maximale.
    subject_description = analyze_subject_with_gemini(client, raw_images)
    subject_anchor      = build_subject_anchor(subject_description)

    # ── 3. Extraction du scénario ─────────────────────────────────────────────
    extracted       = extract_parts(full_prompt)
    visual_scenario = extracted["scenario"] or full_prompt
    v1, v2          = split_text_into_two(extracted["voice_over"])

    print(f"\nStratégie: 16:9 SDK → 1 extension 16:9 → crop ffmpeg CRF16 → 9:16")
    print(f"Ancrage sujet: {'OUI (' + str(len(subject_description)) + ' chars)' if subject_description else 'fallback (images only)'}")
    print(f"Scénario: {visual_scenario[:120]}...")
    print(f"V1 ({len(v1.split())} mots): {v1[:80]}...")
    print(f"V2 ({len(v2.split())} mots): {v2[:80]}...")

    # ── 4. Génération partie 1 — 8s en 16:9 ──────────────────────────────────
    # Structure du prompt : [ANCRAGE SUJET] → [RÈGLE CROP 9:16] → [SCÉNARIO] → [QUALITÉ] → [AUDIO]
    # L'ancrage sujet est EN PREMIER — c'est la contrainte prioritaire pour Veo.
    print(f"\nÉtape 1/2 — 8s en 16:9")

    prompt_1 = (
        subject_anchor

        + "TECHNICAL REQUIREMENT: "
        + "This 16:9 video will be cropped to 9:16 by cutting the left 22% and right 22% of the frame. "
        + "Only the CENTER 56% of the horizontal width will survive. "
        + "Compose this scene AS IF shooting in 9:16 portrait format. "
        + "Every subject, person, and product MUST be framed within the center 56% at ALL times. "
        + "Use portrait-style framing: tight vertical shots, close-ups, medium shots (waist-up). "
        + "Camera: slow zoom in/out, tilt up/down, gentle vertical tracking only. "
        + "FORBIDDEN: horizontal pan, subjects moving to frame edges, wide landscape shots. "

        + "QUALITY: 4K cinematic, ultra-sharp focus, no motion blur, no grain, no artifacts. "
        + "Professional cinematic lighting. Photorealistic. "
        + "FORBIDDEN: floating text, watermark, subtitles, cartoon, CGI. "

        + f"VISUAL SCENARIO (first half): {visual_scenario}. "
        + "HOOK: immediate visual impact in first 2 seconds. "

        + f"AUDIO: narrator speaks ONLY this French text: '{v1}'."
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

    # ── 5. Extension — 8s en 16:9 ────────────────────────────────────────────
    # Même ancrage sujet répété pour cohérence visuelle entre partie 1 et 2.
    print(f"\nÉtape 2/2 — Extension 8s en 16:9")

    prompt_2 = (
        subject_anchor

        + "CONTINUE SEAMLESSLY FROM PREVIOUS CLIP. "
        + "SAME REQUIREMENT: crop to center 56% horizontally (9:16 final). "
        + "ALL subjects MUST remain in center 56% of frame. "
        + "Portrait-style framing, no horizontal pan. "
        + "Camera: slow zoom or gentle vertical movement only. "

        + "Ultra-sharp, clean image, no motion blur, photorealistic, cinematic lighting. "
        + "FORBIDDEN: floating text, watermark, subtitles, horizontal pan. "

        + f"VISUAL: build to climax — {visual_scenario}. "
        + "End with strong CTA moment: subject looks directly at camera, confident. "
        + "Brand/product clearly visible at center bottom (within safe zone). "

        + f"AUDIO: narrator concludes ONLY: '{v2}'."
    )

    op2 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt_2,
        video=video1,
        config=types.GenerateVideosConfig(
            duration_seconds=8,
            resolution="720p",
        ),
    )

    video2 = wait_for_op(client, op2)

    if not video2 or not download_video(client, video2, "raw_16x9.mp4"):
        print("Extension échouée — crop partie 1 uniquement (8s)")
        crop_to_portrait("part1_16x9.mp4", output_file)
        sys.exit(0)

    # video2 contient DÉJÀ la vidéo complète (part1 + extension = ~15s)
    # Veo retourne la vidéo combinée — pas besoin de concaténer avec ffmpeg
    print("Extension reçue — vidéo complète ~15s en 16:9")

    # ── 6. Crop final → 9:16 haute qualité ───────────────────────────────────
    print("\nCrop 16:9 → 9:16 (CRF 16, preset slow)...")
    if not crop_to_portrait("raw_16x9.mp4", output_file):
        crop_to_portrait("part1_16x9.mp4", output_file)

    print(f"\nSuccès ! Vidéo 9:16 ~15s → {output_file}")


if __name__ == "__main__":
    generate_video_with_refs()
