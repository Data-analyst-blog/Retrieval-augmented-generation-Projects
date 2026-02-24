import pdfplumber
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def parse_pdf(file_path):
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract text
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    text += " | ".join([str(cell) for cell in row]) + "\n"

            # OCR for scanned PDFs
            if not page_text:
                image = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(image)
                text += ocr_text + "\n"

    return text