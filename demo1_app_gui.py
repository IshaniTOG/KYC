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
import pandas as pd
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

EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH")
if not EXCEL_FILE_PATH:
    raise EnvironmentError("EXCEL_FILE_PATH is not set in .env — add: EXCEL_FILE_PATH=C:\\path\\to\\Data.xlsx")

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
    name_en: str | None = None   # may be absent on old cards where name is Sinhala-only
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
    """Extract structured fields from an ID card image using Gemini.

    Retry strategy (3 attempts):
      0 — original image  + detailed schema-specific prompt
      1 — CLAHE-enhanced  + detailed prompt  (helps faded/handwritten text)
      2 — original image  + detailed prompt  (final retry)
    """
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

def verify_faces(id_image: str, selfie_image: str):
    return DeepFace.verify(
        img1_path=id_image,
        img2_path=selfie_image,
        model_name="ArcFace",
        detector_backend="retinaface",
    )

def load_excel_data(file_path):
    try:
        if not os.path.exists(file_path):
            base_path = os.path.splitext(file_path)[0]
            for ext in [".xlsx", ".xls", ".csv"]:
                alt_path = base_path + ext
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Excel file not found at: {file_path}")

        df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        print(f"Columns available: {list(df.columns)}")

        def find_col(df, candidates):
            for col in df.columns:
                if col in candidates:
                    return col
            for col in df.columns:
                if any(c in col for c in candidates):
                    return col
            return None

        first_name_col = find_col(df, ["first_name", "firstname", "first", "fname", "given_name"]) or df.columns[0]
        last_name_col  = find_col(df, ["last_name", "lastname", "last", "lname", "surname", "family_name"])
        nic_col        = find_col(df, ["nic", "nic_no", "nic_number", "id_no", "id_number", "national_id", "national_id_number"])
        email_col      = find_col(df, ["email", "email_address", "email_id", "e_mail"])
        phone_col      = find_col(df, ["telephone", "telephone_no", "phone", "phone_number", "mobile", "mobile_number", "contact_number"])

        first_names = sorted(df[first_name_col].dropna().astype(str).str.strip().unique())
        print(f"Loaded {len(first_names)} records from Excel")

        return df, first_names, {
            "first_name": first_name_col,
            "last_name":  last_name_col,
            "nic":        nic_col,
            "email":      email_col,
            "phone":      phone_col,
        }

    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None, ["Select name from Excel"], {}

# ---------------- GUI ----------------

class KYCApp:
    def _warmup_deepface(self):
        """Load ArcFace + RetinaFace weights in the background at startup.
        Without this, the first 'Run KYC' pays a 20-40s cold-start penalty."""
        import numpy as np
        try:
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            DeepFace.represent(img_path=dummy, model_name="ArcFace",
                               detector_backend="skip", enforce_detection=False)
        except Exception:
            pass  # warmup failure is non-fatal

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

        self.df, self.first_names_list, self.column_mapping = load_excel_data(EXCEL_FILE_PATH)

        # Pre-load ArcFace weights in the background so the first verification
        # doesn't pay a 20-40s cold-start penalty
        threading.Thread(target=self._warmup_deepface, daemon=True).start()

        self.selected_name = tk.StringVar(value="Select a name")
        self.input_mode    = tk.StringVar(value="dropdown")

        self.id_front_path = None
        self.id_back_path  = None
        self.selfie_path   = None

        # ========== SCROLLABLE FRAME ==========
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

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))

        # ========== INPUT MODE SECTION ==========
        mode_frame = tk.Frame(self.scrollable_frame, relief="groove", bd=2, bg="#f0f0f0")
        mode_frame.pack(fill="x", pady=(0, 20), padx=10)

        tk.Label(mode_frame, text="Input Mode:", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(side="left", padx=(10, 5), pady=10)
        ttk.Radiobutton(mode_frame, text="Select from Excel", variable=self.input_mode, value="dropdown",
                        command=self.toggle_input_mode).pack(side="left", padx=10)
        ttk.Radiobutton(mode_frame, text="Manual Entry", variable=self.input_mode, value="manual",
                        command=self.toggle_input_mode).pack(side="left", padx=10)

        tk.Label(mode_frame, text="(ID type is auto-detected from NIC number)",
                 font=("Arial", 9), bg="#f0f0f0", fg="#666666").pack(side="left", padx=20, pady=10)

        # ========== DROPDOWN SECTION ==========
        self.dropdown_frame = tk.Frame(self.scrollable_frame, relief="groove", bd=2, bg="#f0f0f0")
        self.dropdown_frame.pack(fill="x", pady=(0, 20), padx=10)

        tk.Label(self.dropdown_frame, text="Select Customer:", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(side="left", padx=(10, 5), pady=10)
        self.name_dropdown = ttk.Combobox(
            self.dropdown_frame, textvariable=self.selected_name,
            values=self.first_names_list, state="readonly", width=30, font=("Arial", 10),
        )
        self.name_dropdown.pack(side="left", padx=5, pady=10)

        # ========== TWO-COLUMN LAYOUT ==========
        columns_frame = tk.Frame(self.scrollable_frame, bg="white")
        columns_frame.pack(fill="both", expand=True, padx=10, pady=5)

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

        tk.Label(form_frame, text="Email", font=("Arial", 10), bg="white").grid(row=3, column=0, sticky="w", pady=8)
        self.email_entry = tk.Entry(form_frame, width=35, font=("Arial", 10))
        self.email_entry.grid(row=3, column=1, padx=15, pady=8)

        tk.Label(form_frame, text="Telephone No", font=("Arial", 10), bg="white").grid(row=4, column=0, sticky="w", pady=8)
        self.telephone_entry = tk.Entry(form_frame, width=35, font=("Arial", 10))
        self.telephone_entry.grid(row=4, column=1, padx=15, pady=8)

        ttk.Separator(left_column, orient="horizontal").pack(fill="x", pady=20)

        upload_frame = tk.Frame(left_column, bg="white")
        upload_frame.pack(pady=10)

        tk.Button(upload_frame, text="Upload ID Front", command=self.upload_id_front,
                  bg="#3498db", fg="white", font=("Arial", 10, "bold"), width=20, height=1).pack(side="left", padx=5, pady=1)
        tk.Button(upload_frame, text="Upload ID Back", command=self.upload_id_back,
                  bg="#3498db", fg="white", font=("Arial", 10, "bold"), width=20, height=1).pack(side="left", padx=5, pady=1)
        tk.Button(upload_frame, text="Upload Selfie", command=self.upload_selfie,
                  bg="#3498db", fg="white", font=("Arial", 10, "bold"), width=20, height=1).pack(side="left", padx=5, pady=1)

        self.run_btn = tk.Button(left_column, text="Run KYC Verification", command=self.run_kyc,
                                 bg="#27ae60", fg="white", font=("Arial", 10, "bold"), width=20, height=1)
        self.run_btn.pack(pady=3)

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
        self.output.insert(tk.END, "1. Select input mode (Excel dropdown or manual entry)\n")
        self.output.insert(tk.END, "2. If using dropdown, select a customer\n")
        self.output.insert(tk.END, "3. Upload ID front, ID back (optional), and selfie\n")
        self.output.insert(tk.END, "4. Click 'Run KYC Verification'\n")
        self.output.insert(tk.END, "5. ID type is auto-detected from the NIC number\n")
        self.output.insert(tk.END, "6. Results will appear here\n")
        self.output.insert(tk.END, "=" * 60 + "\n")

        self.name_dropdown.bind("<<ComboboxSelected>>", lambda e: self.auto_fill_all())
        self.toggle_input_mode()

    def toggle_input_mode(self):
        if self.input_mode.get() == "dropdown":
            self.name_dropdown.config(state="readonly")
            self.dropdown_frame.config(bg="#f0f0f0")
            self.clear_all_fields()
            if self.selected_name.get() != "Select a name":
                self.auto_fill_all()
        else:
            self.name_dropdown.config(state="disabled")
            self.dropdown_frame.config(bg="#e0e0e0")
            self.selected_name.set("Manual Entry Mode")
            self.clear_all_fields()

    def auto_fill_all(self):
        if self.input_mode.get() != "dropdown":
            return
        selected_first_name = self.selected_name.get()
        if not selected_first_name or selected_first_name == "Select a name" or self.df is None:
            return
        try:
            first_name_col = self.column_mapping.get("first_name")
            if not first_name_col:
                messagebox.showwarning("Warning", "First name column not found in Excel")
                return

            mask = self.df[first_name_col].astype(str).str.strip().str.lower() == selected_first_name.lower().strip()
            matching_rows = self.df[mask]
            if len(matching_rows) == 0:
                messagebox.showwarning("Not Found", f"No data found for: {selected_first_name}")
                return

            row = matching_rows.iloc[0]
            self.clear_all_fields()
            self.first_name_entry.insert(0, str(row[first_name_col]).strip())

            for attr, col_key in [("last_name_entry", "last_name"), ("nic_no_entry", "nic"),
                                   ("email_entry", "email"), ("telephone_entry", "phone")]:
                col = self.column_mapping.get(col_key)
                if col and col in row:
                    getattr(self, attr).insert(0, str(row[col]).strip())

            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, f"✓ Auto-filled data for: {selected_first_name}\n")
            self.output.insert(tk.END, "=" * 60 + "\n\nReady for KYC verification...\n")

        except Exception as e:
            messagebox.showerror("Error", f"Error auto-filling fields: {e}")

    def clear_all_fields(self):
        for entry in [self.first_name_entry, self.last_name_entry,
                      self.nic_no_entry, self.email_entry, self.telephone_entry]:
            entry.delete(0, tk.END)

    def refresh_data(self):
        self.df, self.first_names_list, self.column_mapping = load_excel_data(EXCEL_FILE_PATH)
        self.name_dropdown["values"] = self.first_names_list
        messagebox.showinfo("Refreshed", f"Loaded {len(self.first_names_list)} records from Excel")

    def upload_id_front(self):
        self.id_front_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if self.id_front_path:
            messagebox.showinfo("Uploaded", "ID Front uploaded successfully")
            self.output.insert(tk.END, f"✓ ID Front uploaded: {os.path.basename(self.id_front_path)}\n")

    def upload_id_back(self):
        self.id_back_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if self.id_back_path:
            messagebox.showinfo("Uploaded", "ID Back uploaded successfully")
            self.output.insert(tk.END, f"✓ ID Back uploaded: {os.path.basename(self.id_back_path)}\n")

    def upload_selfie(self):
        self.selfie_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if self.selfie_path:
            messagebox.showinfo("Uploaded", "Selfie uploaded successfully")
            self.output.insert(tk.END, f"✓ Selfie uploaded: {os.path.basename(self.selfie_path)}\n")

    # ---- KYC verification (runs in background thread) ----

    def run_kyc(self):
        if not self.id_front_path or not self.selfie_path:
            messagebox.showerror("Error", "Please upload both ID card and selfie")
            return

        # Read all inputs on the main thread before handing off
        input_first_name = self.first_name_entry.get().strip()
        input_last_name  = self.last_name_entry.get().strip()
        input_id_no      = self.nic_no_entry.get().strip()
        input_email      = self.email_entry.get().strip()
        input_phone      = self.telephone_entry.get().strip()
        input_name       = f"{input_first_name} {input_last_name}".strip()

        if not input_first_name:
            messagebox.showerror("Error", "First Name is required")
            return
        if not input_id_no:
            messagebox.showerror("Error", "NIC No is required")
            return

        input_mode             = self.input_mode.get()
        selected_dropdown_name = self.selected_name.get() if input_mode == "dropdown" else "Manual Entry"
        selected_id_type       = "Type 1" if is_type1_nic(input_id_no) else "Type 2"
        id_front_path          = self.id_front_path
        id_back_path           = self.id_back_path
        selfie_path            = self.selfie_path

        # Show processing state, disable button to prevent double-clicks
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"🔍 KYC Verification Started\n")
        self.output.insert(tk.END, f"Input Mode: {input_mode.upper()}\n")
        self.output.insert(tk.END, f"Customer: {selected_dropdown_name}\n")
        self.output.insert(tk.END, f"ID Type: {selected_id_type} (auto-detected from NIC)\n")
        self.output.insert(tk.END, "=" * 60 + "\n\n")
        self.output.insert(tk.END, "⏳ Processing... please wait (this may take ~1–2 minutes).\n")
        self.run_btn.config(state="disabled", text="Processing...")

        def worker():
            try:
                front_schema = FIDDetails     if selected_id_type == "Type 1" else FIDDetails_type2
                back_schema  = BIDDetails     if selected_id_type == "Type 1" else BIDDetails_type2

                # --- Stage 1: OCR front + OCR back + face verify + face thumbnails — all parallel ---
                with ThreadPoolExecutor(max_workers=5) as executor:
                    front_f     = executor.submit(extract_id_details, id_front_path, front_schema)
                    back_f      = executor.submit(extract_id_details, id_back_path, back_schema) if id_back_path else None
                    verify_f    = executor.submit(verify_faces, id_front_path, selfie_path)
                    id_face_f   = executor.submit(DeepFace.extract_faces, id_front_path, detector_backend="retinaface")
                    sel_face_f  = executor.submit(DeepFace.extract_faces, selfie_path,   detector_backend="retinaface")

                    fid          = front_f.result()
                    bid          = back_f.result() if back_f else None
                    face_result  = verify_f.result()
                    id_face      = id_face_f.result()[0]["face"]
                    selfie_face  = sel_face_f.result()[0]["face"]

                extracted_id_no = fid.id_no
                if selected_id_type == "Type 1":
                    extracted_name = bid.name_en if bid else "Name not found"
                else:
                    extracted_name = fid.name_en

                # --- Stage 2: GPT name comparison (needs extracted_name from stage 1) ---
                gpt_result    = gpt_name_comparison(input_name, extracted_name)
                gpt_name_conf = gpt_result["confidence_score"]
                id_conf_val   = id_confidence(input_id_no, extracted_id_no)

                self.root.after(0, lambda: self._display_results(
                    input_mode, selected_dropdown_name, selected_id_type,
                    input_first_name, input_last_name, input_id_no, input_email, input_phone,
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

    def _display_results(self, input_mode, selected_dropdown_name, selected_id_type,
                          input_first_name, input_last_name, input_id_no, input_email, input_phone,
                          extracted_name, extracted_id_no,
                          gpt_result, gpt_name_conf, id_conf, face_result, id_face, selfie_face):
        """Update UI with verification results — always called on the main thread."""
        self.show_face(self.id_img_label,     id_face)
        self.show_face(self.selfie_img_label, selfie_face)

        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "=" * 60 + "\n")
        self.output.insert(tk.END, "KYC VERIFICATION RESULTS\n")
        self.output.insert(tk.END, "=" * 60 + "\n\n")

        self.output.insert(tk.END, f"📋 Input Mode: {input_mode.upper()}\n")
        if input_mode == "dropdown":
            self.output.insert(tk.END, f"📋 Customer from Excel: {selected_dropdown_name}\n")
        self.output.insert(tk.END, f"📋 ID Type (auto-detected): {selected_id_type}\n\n")

        self.output.insert(tk.END, "===== INPUT DATA =====\n")
        self.output.insert(tk.END, f"First Name    : {input_first_name}\n")
        self.output.insert(tk.END, f"Last Name     : {input_last_name}\n")
        self.output.insert(tk.END, f"NIC No        : {input_id_no}\n")
        self.output.insert(tk.END, f"Email         : {input_email}\n")
        self.output.insert(tk.END, f"Phone         : {input_phone}\n\n")

        self.output.insert(tk.END, "===== ID CARD EXTRACTED DATA =====\n")
        self.output.insert(tk.END, f"ID Card Name  : {extracted_name}\n")
        self.output.insert(tk.END, f"ID Card No    : {extracted_id_no}\n\n")

        is_old_card    = selected_id_type == "Type 1"
        name_label     = "(advisory for old cards)" if is_old_card else ""
        name_pass      = gpt_name_conf >= 0.8
        biometric_pass = (id_conf >= 95) and face_result["verified"]

        self.output.insert(tk.END, f"===== NAME VERIFICATION {name_label} =====\n")
        self.output.insert(tk.END, "--- Semantic Analysis ---\n")
        self.output.insert(tk.END, f"Same Entity?        : {'✅ YES' if gpt_result['same_entity'] else '❌ NO'}\n")
        self.output.insert(tk.END, f"Confidence Score    : {gpt_name_conf * 100:.1f}%\n")
        if gpt_result.get("explanation"):
            self.output.insert(tk.END, f"Explanation         : {gpt_result['explanation']}\n")
        if is_old_card:
            self.output.insert(tk.END, f"Status              : {'✅ PASS' if name_pass else '⚠️  ADVISORY (handwritten Sinhala OCR may be inaccurate)'}\n\n")
        else:
            self.output.insert(tk.END, f"Status              : {'✅ PASS' if name_pass else '❌ FAIL'}\n\n")

        self.output.insert(tk.END, "===== ID NUMBER VERIFICATION =====\n")
        self.output.insert(tk.END, f"Input ID No         : {input_id_no}\n")
        self.output.insert(tk.END, f"ID Card ID No       : {extracted_id_no}\n")
        self.output.insert(tk.END, f"Confidence          : {id_conf}%\n")
        self.output.insert(tk.END, f"Status              : {'✅ PASS' if id_conf >= 95 else '❌ FAIL'}\n\n")

        self.output.insert(tk.END, "===== FACE VERIFICATION =====\n")
        self.output.insert(tk.END, f"Verified            : {'✅ YES' if face_result['verified'] else '❌ NO'}\n")
        self.output.insert(tk.END, f"Distance            : {round(face_result['distance'], 4)}\n")
        self.output.insert(tk.END, f"Threshold           : {face_result['threshold']}\n")

        if is_old_card:
            # Old cards: NIC + face are hard gates; name mismatch → REQUIRES REVIEW
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
