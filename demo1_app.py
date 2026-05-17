from google import genai
from pydantic import BaseModel
from PIL import Image
from deepface import DeepFace
import cv2
import numpy as np
import pandas as pd
import os
import json
from difflib import SequenceMatcher
import re
import time
import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI
from dotenv import load_dotenv

# =========================================================
# ---------------- CONFIGURATION --------------------------
# =========================================================

load_dotenv()

AZURE_OPENAI_KEY  = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT    = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_DEPLOYMENT  = os.getenv("AZURE_DEPLOYMENT")

azure_client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL")
gemini_client  = genai.Client(api_key=GEMINI_API_KEY)

BASE_FOLDER = os.getenv("BASE_FOLDER")
if not BASE_FOLDER:
    raise EnvironmentError("BASE_FOLDER is not set in .env — add: BASE_FOLDER=C:\\path\\to\\your\\project")

EXCEL_PATH    = os.path.join(BASE_FOLDER, "Data.xlsx")
SHEET_NAME    = "Type_1"
ID_FOLDER     = os.path.join(BASE_FOLDER, "IDs", "1")
SELFIE_FOLDER = os.path.join(BASE_FOLDER, "IDs", "selfie")
OUTPUT_FILE   = os.path.join(BASE_FOLDER, "kyc_results.xlsx")
LOG_FOLDER    = os.path.join(BASE_FOLDER, "logs")

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# =========================================================
# ---------------- SCHEMAS --------------------------------
# =========================================================

class FIDDetails(BaseModel):
    id_no: str
    date_of_issue: str

class BIDDetails(BaseModel):
    name_en: str | None = None   # may be absent on old cards where name is Sinhala-only
    name_si: str | None = None   # handwritten Sinhala script name
    name_tl: str | None = None   # handwritten Tamil script name
    sex: str | None = None
    date_of_birth: str | None = None
    address: str | None = None

class FIDDetails_type2(BaseModel):
    id_no: str
    name_en: str
    name_si: str
    name_tl: str
    sex: str
    date_of_birth: str

class BIDDetails_type2(BaseModel):
    address: str
    date_of_issue: str

# =========================================================
# ---------------- HELPER FUNCTIONS -----------------------
# =========================================================

def is_type1_nic(nic: str) -> bool:
    """Old NIC format: 9 digits + V or X → Type 1. New 12-digit format → Type 2."""
    return bool(re.match(r'^\d{9}[VXvx]$', nic.strip()))

# =========================================================
# ------------ IMAGE PREPROCESSING FOR OCR ----------------
# =========================================================

def preprocess_for_handwritten(img: Image.Image) -> Image.Image:
    """Enhance a PIL image for better handwritten / faded-ink OCR.

    Pipeline:
      1. Grayscale — removes colour noise that confuses character recognition
      2. CLAHE    — balances contrast locally; recovers faded ink in dark areas
      3. Denoise  — smooths scan/photo grain without blurring character edges
      4. Sharpen  — makes handwritten strokes crisper for the vision model
    """
    bgr      = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(enhanced, h=8)

    kernel   = np.array([[0, -1, 0],
                          [-1,  5, -1],
                          [0, -1, 0]], dtype=np.float32)
    sharpened = np.clip(cv2.filter2D(denoised, -1, kernel), 0, 255).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))

# =========================================================
# ------------ SCHEMA-SPECIFIC OCR PROMPTS ----------------
# =========================================================

_OCR_PROMPTS: dict[str, str] = {

    "FIDDetails": """
You are reading the FRONT of a Sri Lankan National Identity Card — OLD FORMAT (Type 1).

WHAT THIS SIDE CONTAINS:
- NIC NUMBER (top area): exactly 9 digits followed by the letter V or X
  Examples: 952620274V  801234567X  760012345V
  • The suffix is almost always 'V'; 'X' is rare but valid.
  • The number is ONLY digits 0-9 — never contains the letter O.
- DATE OF ISSUE (lower area): date this card was issued.
  Common formats: 23.05.2005 / 23/05/2005 / 2005-05-23

HANDWRITING TIPS:
  • Digit '1' may look like lowercase 'l' — it is always a digit in the NIC number.
  • Digit '0' may look like the letter 'O' — always treat it as zero in the NIC number.
  • If a digit is ambiguous, choose the value that keeps the total at 9 digits + V/X.
  • For dates: keep whatever separator is on the card (. / -).

If a field is partially obscured or unclear, return your best possible interpretation.
Do NOT return null — always return a string even if uncertain.
""",

    "BIDDetails": """
You are reading the BACK of a Sri Lankan National Identity Card — OLD FORMAT (Type 1).

WHAT THIS SIDE CONTAINS (top to bottom):
1. NAME — a single line near the top (labelled "නම / பெயர்").
   CRITICAL: On many old cards the name is written ONLY in Sinhala cursive — there is
   NO separate English name line.
   • If English block letters are present → put them in name_en.
   • If the name is in Sinhala cursive ONLY → put the Sinhala text in name_si AND
     write a romanised transliteration in name_en.
     Example: "ශෙහාන් විජේසිංහ" → name_en = "Shehani Wijesinghe"
     Transliteration guide for common Sinhala characters:
       ශ=Sh, ෙ=e, හ=h, ා=a, න්=n, ව=W/V, ි=i, ජ=J, ේ=e, ස=S, ං=ng, ල=l, ක=k
   • Common Sri Lankan name patterns: initials.SURNAME  or  GIVENNAME SURNAME
     e.g. "K.A.S. PERERA", "W.M.D. FERNANDO", "RUWAN BANDARA"
2. OTHER NAMES (වෙනත් නම් / வேறு பெயர்கள்) — usually blank on old cards.
3. DATE OF BIRTH — format DD.MM.YYYY or YYYY.MM.DD.
4. SEX — MALE or FEMALE (also seen as M/F or ස්ත්‍රී/පුරුෂ).
5. OCCUPATION — often blank.
6. ADDRESS — residential place names.
   Common Sri Lankan districts: Colombo, Kandy, Galle, Kurunegala, Matara, Badulla,
   Ratnapura, Anuradhapura, Polonnaruwa, Ampara, Kalutara, Kegalle, Hambantota.

HANDWRITING TIPS:
  • The entire card may be handwritten in Sinhala cursive — read letter by letter.
  • For name_si: capture the Sinhala script characters exactly as written.
  • For name_en: if no English is on the card, provide your best phonetic transliteration.
  • Never return null. Return empty string only if a field is completely blank on the card.
""",

    "FIDDetails_type2": """
You are reading the FRONT of a Sri Lankan National Identity Card — NEW FORMAT (Type 2).

WHAT THIS SIDE CONTAINS:
- NIC NUMBER: exactly 12 digits. First 4 digits are the birth year (e.g. 1999, 2000, 2001).
  Examples: 200035500058  199923401234  200113200398
  • All 12 characters are digits (0-9) — no letters.
- NAME IN ENGLISH  — printed name (not handwritten on new cards).
- NAME IN SINHALA  — printed name in Sinhala script.
- NAME IN TAMIL    — printed name in Tamil script.
- SEX              — MALE or FEMALE.
- DATE OF BIRTH    — usually YYYY.MM.DD format.

New cards are printed, not handwritten, but ink may be faded or the card may be worn.
Return your best interpretation. Do NOT return null.
""",

    "BIDDetails_type2": """
You are reading the BACK of a Sri Lankan National Identity Card — NEW FORMAT (Type 2).

WHAT THIS SIDE CONTAINS:
- ADDRESS  — full residential address in Sri Lanka (printed text).
- DATE OF ISSUE — when this card was issued, usually YYYY.MM.DD format.

Return your best interpretation. Do NOT return null.
""",
}

def _get_ocr_prompt(schema_cls) -> str:
    """Return the detailed OCR prompt for the given schema class."""
    return _OCR_PROMPTS.get(
        schema_cls.__name__,
        "Extract all text fields from this Sri Lankan National Identity Card accurately. "
        "For any handwritten or unclear text, provide your best possible interpretation. "
        "Do NOT return null for any field.",
    )

def load_excel(excel_path: str, sheet_name: str) -> pd.DataFrame:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel not found: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = df.columns.str.strip()
    return df

def find_image(folder: str, base_name: str) -> str:
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(folder, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"No image found for {base_name} in {folder} "
        f"(expected one of {IMAGE_EXTENSIONS})"
    )

def extract_id_details(image_path: str, schema, max_retries: int = 3):
    """Extract structured fields from an ID card image using Gemini.

    Retry strategy (3 attempts):
      0 — original image  + detailed schema-specific prompt
      1 — CLAHE-enhanced  + detailed prompt  (helps faded/handwritten text)
      2 — original image  + detailed prompt  (final retry, fresh API call)
    Exponential backoff is applied only on API errors, not between normal attempts.
    """
    prompt = _get_ocr_prompt(schema)

    original = Image.open(image_path).convert("RGB")
    w, h = original.size
    if max(w, h) > 1600:
        scale = 1600 / max(w, h)
        original = original.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    preprocessed = preprocess_for_handwritten(original)

    for attempt in range(max_retries):
        img = preprocessed if attempt == 1 else original
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt, img],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
            return response.parsed
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)  # 5s → 10s
                print(f"   ⚠️  Gemini OCR error (attempt {attempt + 1}/{max_retries}): {e}. "
                      f"{'Retrying with enhanced image' if attempt == 0 else 'Retrying'} in {wait}s...")
                time.sleep(wait)
            else:
                raise

def verify_faces(id_image: str, selfie_image: str) -> dict:
    result = DeepFace.verify(
        img1_path=id_image,
        img2_path=selfie_image,
        model_name="ArcFace",
        detector_backend="retinaface",
    )
    return {
        "verified":  result["verified"],
        "distance":  result["distance"],
        "threshold": result["threshold"],
    }

def gpt_name_comparison(input_name: str, id_name: str) -> dict:
    prompt = f"""
    Compare the following two personal names and determine whether they
    refer to the same individual.

    Name 1: {input_name}
    Name 2: {id_name}

    Consider:
    1. Cultural name ordering (Western vs Eastern name formats)
    2. Initials vs full names (e.g., "J. Smith" vs "John Smith")
    3. Abbreviations (e.g., "Wm" for "William")
    4. Spelling variations and common misspellings
    5. Nicknames and common diminutives
    6. Middle names/initials inclusion/exclusion

    Return your response in the following JSON format:
    {{
        "same_entity": true/false,
        "confidence_score": number between 0 and 1,
        "explanation": "brief explanation of your reasoning"
    }}

    Only return the JSON object, no additional text.
    """

    try:
        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in name matching and verification. You only respond with valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )

        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()

        result = json.loads(result_text)
        if not all(k in result for k in ["same_entity", "confidence_score", "explanation"]):
            raise ValueError("Invalid response structure from GPT")
        return result

    except Exception as e:
        print(f"⚠️  GPT name comparison failed: {e}. Using fallback method.")
        similarity = SequenceMatcher(None, input_name.lower(), id_name.lower()).ratio()
        return {
            "same_entity":      similarity >= 0.8,
            "confidence_score": similarity,
            "explanation":      f"Fallback method used due to GPT error: {e}",
        }

def nic_confidence(input_nic: str, id_card_nic: str) -> float:
    clean_input = re.sub(r'[^A-Z0-9]', '', input_nic.upper())
    clean_id    = re.sub(r'[^A-Z0-9]', '', id_card_nic.upper())
    if clean_input == clean_id:
        return 100.0
    return round(SequenceMatcher(None, clean_input, clean_id).ratio() * 100, 2)

def save_log_to_file(log_content: str, nic: str = "batch") -> str:
    os.makedirs(LOG_FOLDER, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(LOG_FOLDER, f"kyc_log_{nic}_{timestamp}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"✅ Log saved to: {log_path}")
    return log_path

def generate_kyc_log(input_name, extracted_name, gpt_result, gpt_name_conf,
                     input_id_no, extracted_id_no, id_conf, face_result,
                     processing_time=None) -> str:
    lines = [
        "=" * 60,
        "KYC VERIFICATION RESULTS",
        "=" * 60, "",
        "===== NAME VERIFICATION =====",
        f"Input Name          : {input_name}",
        f"ID Card Name        : {extracted_name}", "",
        "--- GPT Analysis ---",
        f"Same Entity?        : {'✅ YES' if gpt_result['same_entity'] else '❌ NO'}",
        f"Confidence Score    : {gpt_name_conf}%",
    ]
    if gpt_result.get("explanation"):
        lines.append(f"Explanation         : {gpt_result['explanation']}")
    lines += [
        f"Status              : {'✅ PASS' if gpt_name_conf >= 80 else '❌ FAIL'}", "",
        "===== ID NUMBER VERIFICATION =====",
        f"Input ID No         : {input_id_no}",
        f"ID Card ID No       : {extracted_id_no}",
        f"Confidence          : {id_conf}%",
        f"Status              : {'✅ PASS' if id_conf >= 95 else '❌ FAIL'}", "",
        "===== FACE VERIFICATION =====",
        f"Verified            : {'✅ YES' if face_result['verified'] else '❌ NO'}",
        f"Distance            : {round(face_result['distance'], 4)}",
        f"Threshold           : {face_result['threshold']}",
    ]
    if processing_time is not None:
        lines.append(f"Processing Time     : {processing_time:.2f} seconds")
    overall_pass = (gpt_name_conf >= 80) and (id_conf >= 95) and face_result["verified"]
    lines += [
        "", "=" * 60,
        "✅ OVERALL KYC VERIFICATION: PASSED" if overall_pass else "❌ OVERALL KYC VERIFICATION: FAILED",
        "=" * 60,
    ]
    return "\n".join(lines)

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    return f"{seconds / 3600:.2f} hours"

# =========================================================
# ---------------- RECORD PROCESSOR ----------------------
# =========================================================

def _process_kyc_record(nic: str, excel_name: str,
                         id_front_path: str, id_back_path: str, selfie_path: str) -> dict:
    """Run OCR, face verification, and name comparison for one record.
    NIC type is auto-detected: 9-digit+V/X → Type 1, 12-digit → Type 2.

    OCR (front + back) and face verification are independent, so they run
    in parallel. GPT name comparison runs after OCR since it needs the name.
    """
    front_schema = FIDDetails      if is_type1_nic(nic) else FIDDetails_type2
    back_schema  = BIDDetails      if is_type1_nic(nic) else BIDDetails_type2

    # --- Stage 1: parallel (OCR front + OCR back + face verification) ---
    parallel_start = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        front_future = executor.submit(extract_id_details, id_front_path, front_schema)
        back_future  = executor.submit(extract_id_details, id_back_path,  back_schema)
        face_future  = executor.submit(verify_faces,       id_front_path, selfie_path)

        front_data  = front_future.result()
        back_data   = back_future.result()
        face_result = face_future.result()
    parallel_time = time.time() - parallel_start

    if is_type1_nic(nic):
        extracted_name = back_data.name_en
        extracted_sex  = back_data.sex
    else:
        extracted_name = front_data.name_en
        extracted_sex  = front_data.sex

    # --- Stage 2: GPT name comparison (needs OCR result from stage 1) ---
    name_start   = time.time()
    name_compare = gpt_name_comparison(excel_name, extracted_name)
    name_time    = time.time() - name_start

    return {
        "front_data":     front_data,
        "extracted_name": extracted_name,
        "extracted_sex":  extracted_sex,
        "face_result":    face_result,
        "name_compare":   name_compare,
        "nic_conf":       nic_confidence(nic, front_data.id_no),
        "parallel_time":  parallel_time,
        "name_time":      name_time,
    }

# =========================================================
# ---------------- MAIN PIPELINE --------------------------
# =========================================================

def run_kyc_pipeline():
    overall_start = time.time()
    batch_log = [
        "KYC BATCH PROCESSING LOG",
        f"Process Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60, "",
    ]

    df            = load_excel(EXCEL_PATH, SHEET_NAME)
    total_records = len(df)
    results       = []
    successful_count = 0
    failed_count     = 0

    print(f"\n🚀 Starting KYC verification for {total_records} records...")
    print("-" * 60)

    for index, row in df.iterrows():
        record_start = time.time()
        nic      = str(row["NIC No"]).strip()
        progress = ((index + 1) / total_records) * 100  # defined before try so except can use it

        try:
            first_name = str(row["First Name"]).strip()
            last_name  = str(row["Last Name"]).strip()
            excel_name = f"{first_name} {last_name}"

            header = [
                f"\n{'=' * 60}",
                f"PROCESSING NIC: {nic}",
                f"PROCESSING NAME: {excel_name}",
                f"RECORD {index + 1}/{total_records}",
                f"{'=' * 60}", "",
            ]

            print(f"\n📋 [{index + 1}/{total_records}] Processing NIC: {nic}")
            print(f"   Name: {excel_name}")

            id_front_path = find_image(ID_FOLDER,     f"{nic}_front")
            id_back_path  = find_image(ID_FOLDER,     f"{nic}_back")
            selfie_path   = find_image(SELFIE_FOLDER, f"{nic}_selfie")
            print("   📁 Image paths found")

            r = _process_kyc_record(nic, excel_name, id_front_path, id_back_path, selfie_path)

            record_time  = time.time() - record_start
            nic_conf     = r["nic_conf"]
            nic_match    = nic_conf >= 95
            name_compare = r["name_compare"]
            face_result  = r["face_result"]

            individual_log = generate_kyc_log(
                input_name=excel_name,
                extracted_name=r["extracted_name"],
                gpt_result=name_compare,
                gpt_name_conf=name_compare["confidence_score"] * 100,
                input_id_no=nic,
                extracted_id_no=r["front_data"].id_no,
                id_conf=nic_conf,
                face_result=face_result,
                processing_time=record_time,
            )
            individual_log += (
                f"\n⏱️  TIMING BREAKDOWN:\n"
                f"OCR + Face (parallel): {r['parallel_time']:.2f}s\n"
                f"Name Comparison:       {r['name_time']:.2f}s\n"
                f"Total Processing:      {record_time:.2f}s\n"
            )

            batch_log.extend(header)
            batch_log.append(individual_log)
            save_log_to_file("\n".join(header) + "\n" + individual_log, nic)

            biometric_pass = nic_match and face_result["verified"]
            name_pass      = name_compare["same_entity"] and name_compare["confidence_score"] >= 0.7

            if is_type1_nic(nic):
                # Old handwritten cards: NIC + face are the hard gates.
                # Name OCR is unreliable on cursive Sinhala, so a name mismatch
                # triggers REQUIRES REVIEW rather than an outright FAILED.
                final_status = biometric_pass
                if biometric_pass and not name_pass:
                    record_status = "REQUIRES REVIEW"
                elif biometric_pass:
                    record_status = "VERIFIED"
                else:
                    record_status = "FAILED"
            else:
                # New printed cards: all three checks required.
                final_status  = biometric_pass and name_pass
                record_status = "VERIFIED" if final_status else "FAILED"

            if final_status:
                successful_count += 1
                status_icon = "✅"
            else:
                failed_count += 1
                status_icon = "❌"

            results.append({
                "ID Name":             r["extracted_name"],
                "NIC No(Extracted)":   r["front_data"].id_no,
                "NIC Confidence":      nic_conf,
                "NIC Match":           nic_match,
                "Face Verified":       face_result["verified"],
                "Face Confidence":     face_result["distance"],
                "Name Match":          name_compare["same_entity"],
                "Name Confidence":     name_compare["confidence_score"] * 100,
                "Sex":                 r["extracted_sex"],
                "Processing Time (s)": round(record_time, 2),
                "Final Status":        record_status,
            })

            print(f"   {status_icon} Record completed in {record_time:.2f} seconds")
            print(f"   📊 Progress: {progress:.1f}% ({successful_count} passed, {failed_count} failed)")

            time.sleep(2)

        except Exception as e:
            failed_count += 1
            print(f"   ❌ Error processing NIC {nic}: {e}")
            print(f"   📊 Progress: {progress:.1f}% ({successful_count} passed, {failed_count} failed)")

            batch_log += [
                f"\n❌ ERROR PROCESSING NIC {nic}:",
                f"   Error: {e}",
                traceback.format_exc(),
            ]
            results.append({
                "ID Name":             "ERROR",
                "NIC No(Extracted)":   "ERROR",
                "NIC Confidence":      0,
                "NIC Match":           False,
                "Face Verified":       False,
                "Face Confidence":     0,
                "Name Match":          False,
                "Name Confidence":     0,
                "Sex":                 "",
                "Processing Time (s)": round(time.time() - record_start, 2),
                "Final Status":        "ERROR",
            })

    overall_time        = time.time() - overall_start
    avg_time_per_record = overall_time / total_records if total_records > 0 else 0

    pd.DataFrame(results).to_excel(OUTPUT_FILE, index=False)

    summary = [
        "\n" + "=" * 60,
        "📊 PROCESSING SUMMARY",
        "=" * 60,
        f"Total Records:           {total_records}",
        f"Successfully Processed:  {successful_count}",
        f"Failed:                  {failed_count}",
        f"Success Rate:            {successful_count / total_records * 100:.1f}%" if total_records > 0 else "Success Rate: N/A",
        f"Total Processing Time:   {format_time(overall_time)}",
        f"Average Time per Record: {avg_time_per_record:.2f} seconds",
    ]
    times = [r.get("Processing Time (s)", 0) for r in results
             if isinstance(r.get("Processing Time (s)"), (int, float))]
    if times:
        summary += [
            f"Fastest Record:          {min(times):.2f} seconds",
            f"Slowest Record:          {max(times):.2f} seconds",
        ]
    summary.append("=" * 60)

    batch_log.extend(summary)
    batch_log += [
        "\n📈 FINAL RESULTS:",
        f"Excel Results Saved to: {OUTPUT_FILE}",
        f"Logs Saved to: {LOG_FOLDER}",
        f"Process Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
    ]
    save_log_to_file("\n".join(batch_log), "batch_summary")

    print("\n" + "=" * 60)
    print("✅ KYC BATCH PROCESSING COMPLETED")
    for line in summary:
        print(line)
    print(f"\n📁 Logs saved to: {LOG_FOLDER}")
    print(f"📊 Results saved to: {OUTPUT_FILE}")
    print("=" * 60)


def warmup_deepface():
    """Load ArcFace + RetinaFace weights into memory before processing begins.
    Without this, the first record pays a 20-40s cold-start penalty."""
    import numpy as np
    print("🔥 Warming up DeepFace models (ArcFace + RetinaFace)...")
    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    try:
        DeepFace.represent(img_path=dummy, model_name="ArcFace", detector_backend="skip", enforce_detection=False)
    except Exception:
        pass
    print("✅ DeepFace models ready.")


if __name__ == "__main__":
    warmup_deepface()
    run_kyc_pipeline()
