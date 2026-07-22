import time
import sys
import json
import requests
import math
import io
import subprocess
from PIL import Image
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
    Charge les images via PIL — syntaxe exacte du notebook officiel Google.
    types.VideoGenerationReferenceImage(image=pil_image, reference_type="asset")
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

            pil_image = Image.open(io.BytesIO(img_response.content))

            ref = types.VideoGenerationReferenceImage(
                image=pil_image,
                reference_type="asset",  # minuscules — syntaxe notebook officiel
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

    if op.result and hasattr(op.result, 'generated_videos') and op.result.generated_videos:
        return op.result.generated_videos[0]

    if hasattr(op, 'error') and op.error:
        print(f"Erreur opération: {op.error}")
    return None


def crop_to_portrait(input_file, output_file):
    """
    Rogne une vidéo 16:9 en 9:16 en gardant le centre.
    Formule : largeur_finale = hauteur * 9/16, centré horizontalement.
    La qualité n'est pas dégradée — crop pur, pas de resize.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0",
        "-c:a", "copy",  # audio non touché
        output_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Erreur ffmpeg: {result.stderr}")
        return False
    print(f"Crop 9:16 réussi → {output_file}")
    return True


def generate_video_with_refs():
    if len(sys.argv) < 4:
        print("Usage: python generate_veo.py <api_key> <prompt> <image_urls_json> [aspect_ratio]")
        sys.exit(1)

    api_key         = sys.argv[1]
    full_prompt     = sys.argv[2]
    image_urls      = json.loads(sys.argv[3])
    # aspect_ratio ignoré — on génère toujours en 16:9 puis on crop en 9:16
    output_filename = "final_video.mp4"
    raw_filename    = "raw_16x9.mp4"

    client = genai.Client(api_key=api_key)

    # ── 1. Extraction ─────────────────────────────────────────────────────────
    extracted       = extract_parts(full_prompt)
    visual_scenario = extracted["scenario"] or full_prompt
    v1, v2          = split_text_into_two(extracted["voice_over"])

    print(f"Stratégie: génération 16:9 + crop ffmpeg → 9:16")
    print(f"Scénario: {visual_scenario[:120]}...")
    print(f"Voix-off partie 1 ({len(v1.split())} mots): {v1[:80]}...")
    print(f"Voix-off partie 2 ({len(v2.split())} mots): {v2[:80]}...")

    # ── 2. Chargement images de référence ─────────────────────────────────────
    reference_images = load_reference_images(image_urls)
    print(f"{len(reference_images)} image(s) de référence chargée(s)")

    # ── 3. Génération partie 1 — 8s en 16:9 ──────────────────────────────────
    # CRITIQUE : on demande explicitement à Veo de centrer le sujet
    # pour que le crop 9:16 soit parfait sans rien couper d'important
    print(f"\nÉtape 1/2 — 8s en 16:9 (sera cropé en 9:16 après)")

    prompt_1 = (
        f"LANDSCAPE 16:9 FORMAT. "
        f"CRITICAL COMPOSITION RULE: Keep ALL subjects, products, and key visual elements "
        f"STRICTLY CENTERED horizontally at all times. "
        f"The left and right 25% of the frame must remain empty or with background only — "
        f"because this video will be cropped to 9:16 portrait and only the CENTER will be kept. "
        f"HOOK IN FIRST 2 SECONDS: immediate visual impact that stops scrolling. "
        f"VISUAL SCENARIO: {visual_scenario}. "
        f"AUDIO: narrator speaks ONLY this French text: '{v1}'. "
        f"Premium cinematic. Photorealistic. No floating text overlay. No watermark. "
        f"Slow sensual camera movements. All movement stays center frame."
    )

    # Modèle avec images de référence → veo-3.1-generate-preview (notebook officiel)
    # Sans images → veo-3.1-fast-generate-preview
    model_p1 = "veo-3.1-fast-generate-preview" if not reference_images else "veo-3.1-fast-generate-preview"

    op1 = client.models.generate_videos(
        model=model_p1,
        prompt=prompt_1,
        config=types.GenerateVideosConfig(
            reference_images=reference_images if reference_images else None,
            duration_seconds=8,
            aspect_ratio="16:9",
            resolution="720p",
        ),
    )

    result1 = wait_for_operation(client, op1)
    if not result1:
        print("Échec étape 1")
        sys.exit(1)

    client.files.download(file=result1.video)
    result1.video.save("part1_16x9.mp4")
    print("Étape 1 réussie. Pause 30s avant étape 2...")
    time.sleep(30)

    # ── 4. Extension partie 2 — 8s en 16:9 ───────────────────────────────────
    # Extension en 16:9 → fonctionne parfaitement (confirmé)
    print(f"\nÉtape 2/2 — Extension 8s en 16:9")

    prompt_2 = (
        f"CONTINUE SEAMLESSLY from previous scene. LANDSCAPE 16:9 FORMAT. "
        f"CRITICAL COMPOSITION RULE: Keep ALL subjects STRICTLY CENTERED horizontally. "
        f"Left and right 25% of frame = background only. "
        f"Build to emotional climax and clear call-to-action. "
        f"End with brand name elegant reveal at bottom center. "
        f"VISUAL: continue — {visual_scenario}. "
        f"AUDIO: narrator concludes ONLY: '{v2}'. "
        f"No floating text overlay. Smooth invisible transition from previous clip."
    )

    op2 = client.models.generate_videos(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt_2,
        video=result1.video,
        config=types.GenerateVideosConfig(
            duration_seconds=8,
            resolution="720p",
            # aspect_ratio absent — hérité de la vidéo source 16:9
        ),
    )

    result2 = wait_for_operation(client, op2)
    if not result2:
        print("Étape 2 échouée — utilisation partie 1 uniquement")
        # Crop direct de la partie 1
        if crop_to_portrait("part1_16x9.mp4", output_filename):
            print(f"Vidéo 9:16 (8s) sauvegardée → {output_filename}")
            sys.exit(0)
        sys.exit(1)

    # ── 5. Sauvegarde partie 2 ────────────────────────────────────────────────
    try:
        client.files.download(file=result2.video)
        result2.video.save("part2_16x9.mp4")
        print("Partie 2 téléchargée")
    except Exception as e:
        print(f"Erreur téléchargement partie 2: {e} — utilisation partie 1")
        crop_to_portrait("part1_16x9.mp4", output_filename)
        sys.exit(0)

    # ── 6. Concaténation des deux parties ─────────────────────────────────────
    print("\nConcaténation des deux parties 16:9...")
    with open("filelist.txt", "w") as f:
        f.write("file 'part1_16x9.mp4'\n")
        f.write("file 'part2_16x9.mp4'\n")

    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", "filelist.txt",
        "-c", "copy",
        raw_filename
    ]
    concat_result = subprocess.run(concat_cmd, capture_output=True, text=True)

    if concat_result.returncode != 0:
        print(f"Erreur concaténation: {concat_result.stderr}")
        print("Fallback : crop partie 2 uniquement")
        crop_to_portrait("part2_16x9.mp4", output_filename)
        sys.exit(0)

    print(f"Concaténation réussie → {raw_filename} (16s en 16:9)")

    # ── 7. Crop final 16:9 → 9:16 ────────────────────────────────────────────
    print("\nCrop ffmpeg 16:9 → 9:16 (centre)...")
    if not crop_to_portrait(raw_filename, output_filename):
        print("Erreur crop — fallback crop partie 1")
        crop_to_portrait("part1_16x9.mp4", output_filename)

    print(f"\nSuccès ! Vidéo 9:16 de ~16s → {output_filename}")


if __name__ == "__main__":
    generate_video_with_refs()
