import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import re
from concurrent.futures import ThreadPoolExecutor
from google import genai
from pydantic import BaseModel
from PIL import Image, ImageTk
from deepface import DeepFace
from difflib import SequenceMatcher
import cv2
import numpy as np
import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# ---------------- CONFIG ----------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL")
client         = genai.Client(api_key=GEMINI_API_KEY)

AZURE_OPENAI_KEY  = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT    = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_DEPLOYMENT  = os.getenv("AZURE_DEPLOYMENT")

azure_client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
)

def is_type1_nic(nic: str) -> bool:
    """Old NIC format: 9 digits + V or X → Type 1. New 12-digit format → Type 2."""
    return bool(re.match(r'^\d{9}[VXvx]$', nic.strip()))

# ---------------- IMAGE PREPROCESSING ----------------

def preprocess_for_handwritten(img: Image.Image) -> Image.Image:
    """Enhance a PIL image for better handwritten / faded-ink OCR.
    Pipeline: grayscale → CLAHE → denoise → sharpen."""
    bgr      = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=8)
    kernel   = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = np.clip(cv2.filter2D(denoised, -1, kernel), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))

# ---------------- OCR PROMPTS ----------------

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

If a field is partially obscured or unclear, return your best possible interpretation.
Do NOT return null — always return a string.
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
     Transliteration guide: ශ=Sh, ෙ=e, හ=h, ා=a, න්=n, ව=W/V, ි=i, ජ=J, ේ=e, ස=S, ල=l
   • Common patterns: initials.SURNAME  or  GIVENNAME SURNAME
     e.g. "K.A.S. PERERA", "W.M.D. FERNANDO", "RUWAN BANDARA"
2. OTHER NAMES — usually blank on old cards.
3. DATE OF BIRTH — format DD.MM.YYYY or YYYY.MM.DD.
4. SEX — MALE or FEMALE.
5. OCCUPATION — often blank.
6. ADDRESS — place names. Common districts: Colombo, Kandy, Galle, Kurunegala,
   Matara, Badulla, Ratnapura, Anuradhapura, Polonnaruwa, Ampara, Kalutara, Kegalle.

HANDWRITING TIPS:
  • The entire card may be handwritten in Sinhala cursive — read letter by letter.
  • For name_si: capture the Sinhala script characters as written.
  • For name_en: if no English is on the card, provide your best phonetic transliteration.
  • Never return null. Return empty string only if a field is completely blank on the card.
""",
    "FIDDetails_type2": """
You are reading the FRONT of a Sri Lankan National Identity Card — NEW FORMAT (Type 2).

WHAT THIS SIDE CONTAINS:
- NIC NUMBER: exactly 12 digits. First 4 digits are the birth year (e.g. 1999, 2000, 2001).
  Examples: 200035500058  199923401234  200113200398
  • All 12 characters are digits 0-9 — no letters.
- NAME IN ENGLISH  — printed name.
- NAME IN SINHALA  — printed name in Sinhala script.
- NAME IN TAMIL    — printed name in Tamil script.
- SEX              — MALE or FEMALE.
- DATE OF BIRTH    — usually YYYY.MM.DD format.

New cards are printed but ink may be faded or card worn.
Return your best interpretation. Do NOT return null.
""",
    "BIDDetails_type2": """
You are reading the BACK of a Sri Lankan National Identity Card — NEW FORMAT (Type 2).

WHAT THIS SIDE CONTAINS:
- ADDRESS  — full residential address in Sri Lanka.
- DATE OF ISSUE — when this card was issued, usually YYYY.MM.DD format.

Return your best interpretation. Do NOT return null.
""",
}

def _get_ocr_prompt(schema_cls) -> str:
    return _OCR_PROMPTS.get(
        schema_cls.__name__,
        "Extract all text fields from this Sri Lankan National Identity Card accurately. "
        "For handwritten or unclear text, provide your best possible interpretation. "
        "Do NOT return null for any field.",
    )

# ---------------- SCHEMAS ----------------

class FIDDetails(BaseModel):
    id_no: str
    date_of_issue: str

class BIDDetails(BaseModel):
    name_en: str | None = None
    name_si: str | None = None
    name_tl: str | None = None
    sex: str | None = None
    date_of_birth: str | None = None
    address: str | None = None

class FIDDetails_type2(BaseModel):
    id_no: str
    name_en: str
    name_si: str | None = None
    name_tl: str | None = None
    sex: str | None = None
    date_of_birth: str | None = None

class BIDDetails_type2(BaseModel):
    address: str | None = None
    date_of_issue: str | None = None

# ---------------- CORE LOGIC ----------------

def extract_id_details(image_path: str, schema, max_retries: int = 3):
    import time
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
            response = client.models.generate_content(
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
                wait = 5 * (2 ** attempt)
                print(f"⚠️  Gemini OCR error (attempt {attempt + 1}/{max_retries}): {e}. "
                      f"{'Retrying with enhanced image' if attempt == 0 else 'Retrying'} in {wait}s...")
                time.sleep(wait)
            else:
                raise

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

def id_confidence(input_id: str, id_card_id: str) -> float:
    input_id_clean   = "".join(input_id.upper().split())
    id_card_id_clean = "".join(id_card_id.upper().split())
    if input_id_clean == id_card_id_clean:
        return 100.0
    return round(SequenceMatcher(None, input_id_clean, id_card_id_clean).ratio() * 100, 2)

def face_confidence(distance: float) -> float:
    """Convert ArcFace cosine distance to a 0–100% confidence score.
    confidence = (1 - distance) × 100
    e.g. distance=0.00 → 100%, distance=0.68 → 32%, distance=1.0 → 0%"""
    return round(max(0.0, (1 - distance) * 100), 1)

# Pass threshold = (1 - ArcFace threshold) × 100 = (1 - 0.68) × 100 = 32%
# Any confidence above 32% means distance was below 0.68 (DeepFace verified = True)
FACE_CONFIDENCE_PASS = 32.0

def confidence_remark(avg: float) -> str:
    """Return a human-readable remark based on average confidence across all three checks."""
    if avg >= 80:
        return "HIGH CONFIDENCE — Strong identity match"
    elif avg >= 60:
        return "MODERATE CONFIDENCE — Acceptable identity match"
    elif avg >= 40:
        return "LOW CONFIDENCE — Manual review recommended"
    else:
        return "VERY LOW CONFIDENCE — Identity could not be verified"

def verify_faces(id_image: str, selfie_image: str) -> dict:
    result     = DeepFace.verify(
        img1_path=id_image,
        img2_path=selfie_image,
        model_name="ArcFace",
        detector_backend="retinaface",
    )
    threshold  = result["threshold"]          # e.g. 0.68 from ArcFace
    distance   = result["distance"]
    confidence = face_confidence(distance)
    return {
        "verified":   result["verified"],
        "distance":   distance,
        "threshold":  threshold,
        "confidence": confidence,             # 0–100%, relative to threshold
    }

# ---------------- GUI ----------------

class KYCApp:
    def _warmup_deepface(self):
        """Load ArcFace + RetinaFace weights in the background at startup."""
        try:
            import numpy as np
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            DeepFace.represent(img_path=dummy, model_name="ArcFace",
                               detector_backend="skip", enforce_detection=False)
        except Exception:
            pass

    def show_face(self, label, face_array):
        img = Image.fromarray((face_array * 255).astype("uint8"))
        img = img.resize((180, 180))
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        label.image = photo

    def __init__(self, root):
        self.root = root
        root.title("KYC Verification System")
        root.geometry("1200x800")

        threading.Thread(target=self._warmup_deepface, daemon=True).start()

        self.id_front_path = None
        self.id_back_path  = None
        self.selfie_path   = None

        # ========== SCROLLABLE CANVAS ==========
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.main_frame, bg="white")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.scrollable_frame = tk.Frame(self.canvas, bg="white")
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        self.canvas.bind_all("<Button-4>",   lambda _: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>",   lambda _: self.canvas.yview_scroll(1, "units"))

        # ========== TWO-COLUMN LAYOUT ==========
        columns_frame = tk.Frame(self.scrollable_frame, bg="white")
        columns_frame.pack(fill="both", expand=True, padx=10, pady=10)

        left_column  = tk.Frame(columns_frame, bg="white", width=500)
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right_column = tk.Frame(columns_frame, bg="white", width=500)
        right_column.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # ========== LEFT: INPUT FORM ==========
        tk.Label(left_column, text="Customer Information",
                 font=("Arial", 12, "bold"), bg="white", fg="#2c3e50").pack(pady=(0, 15))

        form_frame = tk.Frame(left_column, bg="white")
        form_frame.pack()

        tk.Label(form_frame, text="First Name *", font=("Arial", 10, "bold"), bg="white").grid(row=0, column=0, sticky="w", pady=8)
        self.first_name_entry = tk.Entry(form_frame, width=35, font=("Arial", 10))
        self.first_name_entry.grid(row=0, column=1, padx=15, pady=8)

        tk.Label(form_frame, text="Last Name", font=("Arial", 10), bg="white").grid(row=1, column=0, sticky="w", pady=8)
        self.last_name_entry = tk.Entry(form_frame, width=35, font=("Arial", 10))
        self.last_name_entry.grid(row=1, column=1, padx=15, pady=8)

        tk.Label(form_frame, text="NIC No *", font=("Arial", 10, "bold"), bg="white").grid(row=2, column=0, sticky="w", pady=8)
        self.nic_no_entry = tk.Entry(form_frame, width=35, font=("Arial", 10))
        self.nic_no_entry.grid(row=2, column=1, padx=15, pady=8)

        tk.Label(form_frame, text="Telephone No", font=("Arial", 10), bg="white").grid(row=3, column=0, sticky="w", pady=8)
        self.telephone_entry = tk.Entry(form_frame, width=35, font=("Arial", 10))
        self.telephone_entry.grid(row=3, column=1, padx=15, pady=8)

        ttk.Separator(left_column, orient="horizontal").pack(fill="x", pady=20)

        upload_frame = tk.Frame(left_column, bg="white")
        upload_frame.pack(pady=10)

        tk.Button(upload_frame, text="Upload ID Front", command=self.upload_id_front,
                  bg="#3498db", fg="white", font=("Arial", 10, "bold"), width=20, height=1).pack(side="left", padx=5, pady=1)
        tk.Button(upload_frame, text="Upload ID Back", command=self.upload_id_back,
                  bg="#3498db", fg="white", font=("Arial", 10, "bold"), width=20, height=1).pack(side="left", padx=5, pady=1)
        tk.Button(upload_frame, text="Upload Selfie", command=self.upload_selfie,
                  bg="#3498db", fg="white", font=("Arial", 10, "bold"), width=20, height=1).pack(side="left", padx=5, pady=1)

        btn_row = tk.Frame(left_column, bg="white")
        btn_row.pack(pady=10)

        self.run_btn = tk.Button(btn_row, text="Run KYC Verification", command=self.run_kyc,
                                 bg="#27ae60", fg="white", font=("Arial", 10, "bold"), width=20, height=1)
        self.run_btn.pack(side="left", padx=5)

        tk.Button(btn_row, text="Clear", command=self.clear_all,
                  bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), width=10, height=1).pack(side="left", padx=5)

        ttk.Separator(left_column, orient="horizontal").pack(fill="x", pady=20)

        tk.Label(left_column, text="Face Comparison",
                 font=("Arial", 12, "bold"), bg="white", fg="#2c3e50").pack(pady=(0, 10))

        self.image_frame = tk.Frame(left_column, bg="white")
        self.image_frame.pack(pady=10)

        self.id_img_label     = tk.Label(self.image_frame, text="ID Face",     font=("Arial", 9), bg="white")
        self.id_img_label.grid(row=0, column=0, padx=15)
        self.selfie_img_label = tk.Label(self.image_frame, text="Selfie Face", font=("Arial", 9), bg="white")
        self.selfie_img_label.grid(row=0, column=1, padx=15)

        # ========== RIGHT: RESULTS ==========
        tk.Label(right_column, text="Verification Results",
                 font=("Arial", 12, "bold"), bg="white", fg="#2c3e50").pack(pady=(0, 10))

        output_frame = tk.Frame(right_column, bg="white")
        output_frame.pack(fill="both", expand=True)

        self.output = tk.Text(output_frame, height=35, width=65, font=("Consolas", 9), wrap="word")
        output_scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output.yview)
        self.output.configure(yscrollcommand=output_scrollbar.set)
        self.output.pack(side="left", fill="both", expand=True)
        output_scrollbar.pack(side="right", fill="y")

        self.output.insert(tk.END, "KYC Verification Results will appear here...\n\n")
        self.output.insert(tk.END, "=" * 60 + "\n")
        self.output.insert(tk.END, "Instructions:\n")
        self.output.insert(tk.END, "=" * 60 + "\n")
        self.output.insert(tk.END, "1. Fill in First Name, Last Name, and NIC No\n")
        self.output.insert(tk.END, "2. Upload ID Front, ID Back (optional), and Selfie\n")
        self.output.insert(tk.END, "3. Click 'Run KYC Verification'\n")
        self.output.insert(tk.END, "4. ID type is auto-detected from the NIC number\n")
        self.output.insert(tk.END, "5. Results will appear here\n")
        self.output.insert(tk.END, "=" * 60 + "\n")

    # ---- Clear ----

    def clear_all(self):
        # Reset form fields
        for entry in [self.first_name_entry, self.last_name_entry,
                      self.nic_no_entry, self.telephone_entry]:
            entry.delete(0, tk.END)

        # Reset uploaded image paths
        self.id_front_path = None
        self.id_back_path  = None
        self.selfie_path   = None

        # Reset face preview labels
        self.id_img_label.configure(image="", text="ID Face")
        self.id_img_label.image = None
        self.selfie_img_label.configure(image="", text="Selfie Face")
        self.selfie_img_label.image = None

        # Reset output area
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "KYC Verification Results will appear here...\n\n")
        self.output.insert(tk.END, "=" * 60 + "\n")
        self.output.insert(tk.END, "Instructions:\n")
        self.output.insert(tk.END, "=" * 60 + "\n")
        self.output.insert(tk.END, "1. Fill in First Name, Last Name, and NIC No\n")
        self.output.insert(tk.END, "2. Upload ID Front, ID Back (optional), and Selfie\n")
        self.output.insert(tk.END, "3. Click 'Run KYC Verification'\n")
        self.output.insert(tk.END, "4. ID type is auto-detected from the NIC number\n")
        self.output.insert(tk.END, "5. Results will appear here\n")
        self.output.insert(tk.END, "=" * 60 + "\n")

    # ---- Upload handlers ----

    def upload_id_front(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            self.id_front_path = path
            messagebox.showinfo("Uploaded", "ID Front uploaded successfully")
            self.output.insert(tk.END, f"✓ ID Front: {os.path.basename(path)}\n")

    def upload_id_back(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            self.id_back_path = path
            messagebox.showinfo("Uploaded", "ID Back uploaded successfully")
            self.output.insert(tk.END, f"✓ ID Back: {os.path.basename(path)}\n")

    def upload_selfie(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            self.selfie_path = path
            messagebox.showinfo("Uploaded", "Selfie uploaded successfully")
            self.output.insert(tk.END, f"✓ Selfie: {os.path.basename(path)}\n")

    # ---- KYC verification (runs in background thread) ----

    def run_kyc(self):
        if not self.id_front_path or not self.selfie_path:
            messagebox.showerror("Error", "Please upload ID Front and Selfie")
            return

        input_first_name = self.first_name_entry.get().strip()
        input_last_name  = self.last_name_entry.get().strip()
        input_id_no      = self.nic_no_entry.get().strip()
        input_phone      = self.telephone_entry.get().strip()
        input_name       = f"{input_first_name} {input_last_name}".strip()

        if not input_first_name:
            messagebox.showerror("Error", "First Name is required")
            return
        if not input_id_no:
            messagebox.showerror("Error", "NIC No is required")
            return

        selected_id_type = "Type 1" if is_type1_nic(input_id_no) else "Type 2"
        id_front_path    = self.id_front_path
        id_back_path     = self.id_back_path
        selfie_path      = self.selfie_path

        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "🔍 KYC Verification Started\n")
        self.output.insert(tk.END, f"ID Type: {selected_id_type} (auto-detected from NIC)\n")
        self.output.insert(tk.END, "=" * 60 + "\n\n")
        self.output.insert(tk.END, "⏳ Processing... please wait.\n")
        self.run_btn.config(state="disabled", text="Processing...")

        def worker():
            try:
                front_schema = FIDDetails      if selected_id_type == "Type 1" else FIDDetails_type2
                back_schema  = BIDDetails      if selected_id_type == "Type 1" else BIDDetails_type2

                with ThreadPoolExecutor(max_workers=5) as executor:
                    front_f    = executor.submit(extract_id_details, id_front_path, front_schema)
                    back_f     = executor.submit(extract_id_details, id_back_path, back_schema) if id_back_path else None
                    verify_f   = executor.submit(verify_faces, id_front_path, selfie_path)
                    id_face_f  = executor.submit(DeepFace.extract_faces, id_front_path, detector_backend="retinaface")
                    sel_face_f = executor.submit(DeepFace.extract_faces, selfie_path,   detector_backend="retinaface")

                    fid         = front_f.result()
                    bid         = back_f.result() if back_f else None
                    face_result = verify_f.result()
                    id_face     = id_face_f.result()[0]["face"]
                    selfie_face = sel_face_f.result()[0]["face"]

                extracted_id_no = fid.id_no
                extracted_name  = (bid.name_en if bid else None) if selected_id_type == "Type 1" else fid.name_en
                if not extracted_name:
                    extracted_name = "Name not found"

                gpt_result    = gpt_name_comparison(input_name, extracted_name)
                gpt_name_conf = gpt_result["confidence_score"]
                id_conf_val   = id_confidence(input_id_no, extracted_id_no)

                self.root.after(0, lambda: self._display_results(
                    selected_id_type,
                    input_first_name, input_last_name, input_id_no, input_phone,
                    extracted_name, extracted_id_no,
                    gpt_result, gpt_name_conf, id_conf_val,
                    face_result, id_face, selfie_face,
                ))

            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda: self._handle_kyc_error(err_msg))
            finally:
                self.root.after(0, lambda: self.run_btn.config(state="normal", text="Run KYC Verification"))

        threading.Thread(target=worker, daemon=True).start()

    def _display_results(self, selected_id_type,
                          input_first_name, input_last_name, input_id_no, input_phone,
                          extracted_name, extracted_id_no,
                          gpt_result, gpt_name_conf, id_conf, face_result, id_face, selfie_face):
        """Update UI with verification results — always called on the main thread."""
        self.show_face(self.id_img_label,     id_face)
        self.show_face(self.selfie_img_label, selfie_face)

        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "=" * 60 + "\n")
        self.output.insert(tk.END, "KYC VERIFICATION RESULTS\n")
        self.output.insert(tk.END, "=" * 60 + "\n\n")
        self.output.insert(tk.END, f"📋 ID Type (auto-detected): {selected_id_type}\n\n")

        self.output.insert(tk.END, "===== INPUT DATA =====\n")
        self.output.insert(tk.END, f"First Name    : {input_first_name}\n")
        self.output.insert(tk.END, f"Last Name     : {input_last_name}\n")
        self.output.insert(tk.END, f"NIC No        : {input_id_no}\n")
        self.output.insert(tk.END, f"Phone         : {input_phone}\n\n")

        self.output.insert(tk.END, "===== ID CARD EXTRACTED DATA =====\n")
        self.output.insert(tk.END, f"ID Card Name  : {extracted_name}\n")
        self.output.insert(tk.END, f"ID Card No    : {extracted_id_no}\n\n")

        is_old_card    = selected_id_type == "Type 1"
        name_label     = "(advisory for old cards)" if is_old_card else ""
        name_pass      = gpt_name_conf >= 0.8
        face_conf      = face_result["confidence"]
        biometric_pass = (id_conf >= 95) and (face_conf >= FACE_CONFIDENCE_PASS)

        self.output.insert(tk.END, f"===== NAME VERIFICATION {name_label} =====\n")
        self.output.insert(tk.END, f"Same Entity?        : {'✅ YES' if gpt_result['same_entity'] else '❌ NO'}\n")
        self.output.insert(tk.END, f"Confidence Score    : {gpt_name_conf * 100:.1f}%\n")
        if gpt_result.get("explanation"):
            self.output.insert(tk.END, f"Explanation         : {gpt_result['explanation']}\n")
        if is_old_card:
            self.output.insert(tk.END, f"Status              : {'✅ PASS' if name_pass else '⚠️  ADVISORY (handwritten OCR may be inaccurate)'}\n\n")
        else:
            self.output.insert(tk.END, f"Status              : {'✅ PASS' if name_pass else '❌ FAIL'}\n\n")

        self.output.insert(tk.END, "===== ID NUMBER VERIFICATION =====\n")
        self.output.insert(tk.END, f"Input ID No         : {input_id_no}\n")
        self.output.insert(tk.END, f"ID Card ID No       : {extracted_id_no}\n")
        self.output.insert(tk.END, f"Confidence          : {id_conf}%\n")
        self.output.insert(tk.END, f"Status              : {'✅ PASS' if id_conf >= 95 else '❌ FAIL'}\n\n")

        self.output.insert(tk.END, "===== FACE VERIFICATION =====\n")
        self.output.insert(tk.END, f"Confidence          : {face_conf}%\n")
        self.output.insert(tk.END, f"Distance            : {round(face_result['distance'], 4)}\n")
        self.output.insert(tk.END, f"Threshold           : {face_result['threshold']}\n")
        self.output.insert(tk.END, f"Status              : {'✅ PASS' if face_conf >= FACE_CONFIDENCE_PASS else '❌ FAIL'}\n")

        avg_confidence = round((gpt_name_conf * 100 + id_conf + face_conf) / 3, 1)

        if is_old_card:
            if biometric_pass and not name_pass:
                overall_label = "⚠️  OVERALL KYC: REQUIRES REVIEW (name unclear — manual check needed)\n"
            elif biometric_pass:
                overall_label = "✅ OVERALL KYC VERIFICATION: PASSED\n"
            else:
                overall_label = "❌ OVERALL KYC VERIFICATION: FAILED\n"
        else:
            overall_pass  = biometric_pass and name_pass
            overall_label = "✅ OVERALL KYC VERIFICATION: PASSED\n" if overall_pass else "❌ OVERALL KYC VERIFICATION: FAILED\n"

        self.output.insert(tk.END, "\n" + "=" * 60 + "\n")
        self.output.insert(tk.END, "===== FINAL REMARKS =====\n")
        self.output.insert(tk.END, f"Name Confidence     : {gpt_name_conf * 100:.1f}%\n")
        self.output.insert(tk.END, f"NIC Confidence      : {id_conf}%\n")
        self.output.insert(tk.END, f"Face Confidence     : {face_conf}%\n")
        self.output.insert(tk.END, f"Overall Confidence  : {avg_confidence}%\n")
        self.output.insert(tk.END, f"Remark              : {confidence_remark(avg_confidence)}\n")
        self.output.insert(tk.END, "=" * 60 + "\n")
        self.output.insert(tk.END, overall_label)
        self.output.insert(tk.END, "=" * 60 + "\n")
        self.output.see("1.0")

    def _handle_kyc_error(self, error_msg: str):
        """Display error on the main thread."""
        messagebox.showerror("Error", f"An error occurred: {error_msg}")
        self.output.insert(tk.END, f"❌ Error: {error_msg}\n")


# ---------------- RUN ----------------

if __name__ == "__main__":
    root = tk.Tk()
    app  = KYCApp(root)
    root.mainloop()
