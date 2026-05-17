#  KYC ID Verification System

An automated KYC verification system that validates user identity using:

-  Excel data (`Data.xlsx`)
-  ID card images
-  Selfie images
-  Face verification
-  Automated result generation (`kyc_results.xlsx`)

The system is available in two versions:

- **demo1_app.py** → Script (CLI) version
- **demo1_app_gui.py** → GUI version

---

#  Installation

## Prerequisites

- Python 3.9 or higher

## Install Dependencies

pip install -r requirements.txt

# File configuration

#BASE_FOLDER = r"C:\Your\Path\To\Project"

EXCEL_PATH    = os.path.join(BASE_FOLDER, "Data.xlsx")
SHEET_NAME    = "Type_1"

ID_FOLDER     = os.path.join(BASE_FOLDER, "IDs", "1")
SELFIE_FOLDER = os.path.join(BASE_FOLDER, "IDs", "selfie")

OUTPUT_FILE   = os.path.join(BASE_FOLDER, "kyc_results.xlsx")
LOG_FOLDER    = os.path.join(BASE_FOLDER, "logs")

# Run Script Versions

python demo1_app.py
python demo1_app_gui.py

# Output
-kyc_results.xlsx will be generated in the project folder
-Logs will be stored inside the logs/ directory

# File Structure  

Project Folder/
├── demo1_app.py       # CLI Version
├── demo1_app_gui.py        # GUI Version
├── requirements.txt      # Dependencies
├── Data.xlsx             # Input data
├── kyc_results.xlsx      # Output (Generated after execution)
├── logs/                 # System logs (Auto-created)
└── IDs/                  # Image storage
    ├── 1/                # Put ID card images here

    └── selfie/           # Put selfie images here


