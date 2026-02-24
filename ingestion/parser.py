from ingestion.pdf_parser import parse_pdf
from ingestion.excel_parser import parse_excel
from ingestion.word_parser import parse_word

def parse_file(file_path):

    if file_path.endswith(".pdf"):
        return parse_pdf(file_path)

    elif file_path.endswith(".xlsx"):
        return parse_excel(file_path)

    elif file_path.endswith(".docx"):
        return parse_word(file_path)

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        return ""