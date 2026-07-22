import time
import sys
import json
import requests
import math
import io
import base64
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


def load_reference_images_b64(image_urls):
    """
    Retourne une liste de dicts {bytesBase64Encoded, mimeType}
    pour injection directe dans le payload REST.
    """
    refs = []
    if not isinstance(image_urls, list):
        return refs

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

            # Convertir en JPEG via PIL pour normaliser le format
            pil_img = Image.open(io.BytesIO(resp.content)).convert('RGB')
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG', quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            refs.append({
                'bytesBase64Encoded': b64,
                'mimeType': 'image/jpeg',
            })
            print(f"Image encodée en base64: {url[:60]}...")

        except Exception as e:
            print(f"Erreur image {url[:60]}: {e}")

    return refs


def generate_video_rest(api_key, model, prompt, ref_images_b64, duration=8, aspect_ratio="16:9", resolution="720p"):
    """
    Appel REST direct à l'API Veo — évite les bugs du SDK Python.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateVideos?key={api_key}"

    instances = [{"prompt": prompt}]

    if ref_images_b64:
        instances[0]["referenceImages"] = [
            {
                "referenceImage": {
                    "bytesBase64Encoded": ref['bytesBase64Encoded'],
                    "mimeType": ref['mimeType'],
                },
                "referenceType": "ASSET",
            }
            for ref in ref_images_b64
        ]

    payload = {
        "instances": instances,
        "parameters": {
            "aspectRatio": aspect_ratio,
            "durationSeconds": duration,
            "resolution": resolution,
        }
    }

    resp = requests.post(url, json=payload, timeout=60)

    if not resp.ok:
        print(f"Erreur API REST: {resp.status_code} — {resp.text[:500]}")
        return None

    data = resp.json()
    operation_name = data.get('name')
    if not operation_name:
        print(f"Pas de operation name dans la réponse: {data}")
        return None

    print(f"Opération démarrée: {operation_name}")
    return operation_name


def poll_operation(api_key, operation_name, timeout_seconds=700):
    """
    Polling de l'opération via REST.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/{operation_name}?key={api_key}"
    elapsed = 0

    while elapsed < timeout_seconds:
        time.sleep(20)
        elapsed += 20

        resp = requests.get(url, timeout=30)
        if not resp.ok:
            print(f"Erreur polling: {resp.status_code}")
            continue

        data = resp.json()
        print(f"En attente... {elapsed}s — done: {data.get('done', False)}")

        if data.get('done'):
            if 'error' in data:
                print(f"Erreur opération: {data['error']}")
                return None

            videos = data.get('response', {}).get('generatedVideos', [])
            if videos:
                return videos[0]
            print(f"Réponse sans vidéo: {data}")
            return None

    print(f"Timeout après {timeout_seconds}s")
    return None


def download_video(api_key, video_obj, output_path):
    """
    Télécharge la vidéo depuis l'URI retournée.
    """
    uri = video_obj.get('video', {}).get('uri') or video_obj.get('uri')
    if not uri:
        print(f"URI introuvable dans: {video_obj}")
        return False

    # Ajouter la clé API à l'URI
    sep = '&' if '?' in uri else '?'
    download_url = f"{uri}{sep}key={api_key}"

    resp = requests.get(download_url, timeout=120, stream=True)
    if not resp.ok:
        print(f"Erreur téléchargement: {resp.status_code}")
        return False

    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    size = __import__('os').path.getsize(output_path)
    print(f"Vidéo téléchargée: {output_path} ({size} bytes)")
    return size > 10000


def extend_video_rest(api_key, model, prompt, video_obj, duration=8, resolution="720p"):
    """
    Extension vidéo via REST — passe la vidéo source par URI.
    """
    video_uri = video_obj.get('video', {}).get('uri') or video_obj.get('uri')
    if not video_uri:
        print(f"URI vidéo source introuvable: {video_obj}")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateVideos?key={api_key}"

    payload = {
        "instances": [{
            "prompt": prompt,
            "video": {"uri": video_uri},
        }],
        "parameters": {
            "durationSeconds": duration,
            "resolution": resolution,
            # aspect_ratio absent — hérité de la vidéo source
        }
    }

    resp = requests.post(url, json=payload, timeout=60)

    if not resp.ok:
        print(f"Erreur extension REST: {resp.status_code} — {resp.text[:500]}")
        return None

    data = resp.json()
    operation_name = data.get('name')
    if not operation_name:
        print(f"Pas de operation name: {data}")
        return None

    print(f"Extension démarrée: {operation_name}")
    return operation_name


def crop_to_portrait(input_file, output_file):
    """
    Rogne 16:9 → 9:16 en gardant le centre. Crop pur, pas de resize.
    """
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
    print(f"Crop 9:16 réussi → {output_file}")
    return True


def generate_video_with_refs():
    if len(sys.argv) < 4:
        print("Usage: python generate_veo.py <api_key> <prompt> <image_urls_json> [aspect_ratio]")
        sys.exit(1)

    api_key        = sys.argv[1]
    full_prompt    = sys.argv[2]
    image_urls     = json.loads(sys.argv[3])
    output_file    = "final_video.mp4"

    # ── 1. Extraction ─────────────────────────────────────────────────────────
    extracted       = extract_parts(full_prompt)
    visual_scenario = extracted["scenario"] or full_prompt
    v1, v2          = split_text_into_two(extracted["voice_over"])

    print(f"Stratégie: génération 16:9 + crop ffmpeg → 9:16")
    print(f"Scénario: {visual_scenario[:120]}...")
    print(f"Voix-off partie 1 ({len(v1.split())} mots): {v1[:80]}...")
    print(f"Voix-off partie 2 ({len(v2.split())} mots): {v2[:80]}...")

    # ── 2. Images de référence en base64 ──────────────────────────────────────
    ref_images_b64 = load_reference_images_b64(image_urls)
    print(f"{len(ref_images_b64)} image(s) de référence chargée(s)")

    # ── 3. Génération partie 1 — 8s en 16:9 ──────────────────────────────────
    print(f"\nÉtape 1/2 — 8s en 16:9 (sera cropé en 9:16)")

    prompt_1 = (
        f"LANDSCAPE 16:9 FORMAT. "
        f"CRITICAL COMPOSITION: Keep ALL subjects and products STRICTLY CENTERED horizontally. "
        f"Left and right 25% of frame must stay empty/background only "
        f"(video will be cropped to 9:16 portrait, only center kept). "
        f"HOOK IN FIRST 2 SECONDS: immediate visual impact that stops scrolling. "
        f"VISUAL SCENARIO: {visual_scenario}. "
        f"AUDIO: narrator speaks ONLY this French text: '{v1}'. "
        f"Premium cinematic. Photorealistic. No floating text overlay. No watermark. "
        f"Slow sensual camera movements. All action stays center frame."
    )

    model = "veo-3.1-fast-generate-preview"
    op1_name = generate_video_rest(api_key, model, prompt_1, ref_images_b64, duration=8, aspect_ratio="16:9")

    if not op1_name:
        print("Échec lancement étape 1")
        sys.exit(1)

    video1_obj = poll_operation(api_key, op1_name)
    if not video1_obj:
        print("Échec étape 1")
        sys.exit(1)

    if not download_video(api_key, video1_obj, "part1_16x9.mp4"):
        print("Échec téléchargement partie 1")
        sys.exit(1)

    print("Étape 1 réussie. Pause 30s avant étape 2...")
    time.sleep(30)

    # ── 4. Extension partie 2 — 8s supplémentaires ───────────────────────────
    print(f"\nÉtape 2/2 — Extension 8s en 16:9")

    prompt_2 = (
        f"CONTINUE SEAMLESSLY from previous scene. LANDSCAPE 16:9 FORMAT. "
        f"CRITICAL COMPOSITION: All subjects STRICTLY CENTERED horizontally. "
        f"Left and right 25% of frame = background only. "
        f"Build to emotional climax and clear call-to-action. "
        f"End with brand name elegant reveal at bottom center. "
        f"VISUAL: continue — {visual_scenario}. "
        f"AUDIO: narrator concludes ONLY: '{v2}'. "
        f"No floating text overlay. Smooth invisible transition."
    )

    op2_name = extend_video_rest(api_key, model, prompt_2, video1_obj, duration=8)

    video2_obj = None
    if op2_name:
        video2_obj = poll_operation(api_key, op2_name)

    if not video2_obj or not download_video(api_key, video2_obj, "part2_16x9.mp4"):
        print("Étape 2 échouée — crop partie 1 uniquement")
        crop_to_portrait("part1_16x9.mp4", output_file)
        sys.exit(0)

    # ── 5. Concaténation ──────────────────────────────────────────────────────
    print("\nConcaténation 16:9...")
    with open("filelist.txt", "w") as f:
        f.write("file 'part1_16x9.mp4'\n")
        f.write("file 'part2_16x9.mp4'\n")

    concat = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "filelist.txt", "-c", "copy", "raw_16x9.mp4"],
        capture_output=True, text=True
    )

    if concat.returncode != 0:
        print(f"Erreur concat: {concat.stderr[-300:]}")
        print("Fallback : crop partie 2")
        crop_to_portrait("part2_16x9.mp4", output_file)
        sys.exit(0)

    print("Concaténation réussie — 16s en 16:9")

    # ── 6. Crop final → 9:16 ─────────────────────────────────────────────────
    print("\nCrop ffmpeg 16:9 → 9:16...")
    if not crop_to_portrait("raw_16x9.mp4", output_file):
        crop_to_portrait("part1_16x9.mp4", output_file)

    print(f"\nSuccès ! Vidéo 9:16 ~16s → {output_file}")


if __name__ == "__main__":
    generate_video_with_refs()
