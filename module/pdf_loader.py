# pdf_loader.py
import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from PIL import ImageOps
import json

pytesseract.pytesseract.tesseract_cmd =  r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def extract_text_from_pdfs(folder_path, cache_file="extracted_text.json"):
    documents = []
    error_files = []

    # Check if cached data exists
    if os.path.exists(cache_file):
        print("Loading text from cache...")
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        return cached_data["documents"], cached_data["error_files"]

    # Process PDFs
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")
            
            try:
                doc = fitz.open(file_path)
                text_content = ''
                for page_num in range(len(doc)):
                    try:
                        page = doc.load_page(page_num)
                        text = page.get_text()
                        
                        if not text.strip():
                            pix = page.get_pixmap()
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            img = img.convert("L")
                            img = ImageOps.autocontrast(img)
                            text = pytesseract.image_to_string(img, lang="vie", config="--psm 6")
                        
                        text_content += text + '\n'
                    
                    except Exception as page_error:
                        print(f"Error reading page {page_num + 1} of {file_name}: {page_error}")

                doc.close()
                documents.append({'file_name': file_name, 'text': text_content})
            
            except Exception as file_error:
                print(f"Error processing file {file_name}: {file_error}")
                error_files.append((file_name, str(file_error)))

    # Save to cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"documents": documents, "error_files": error_files}, f, ensure_ascii=False, indent=4)

    return documents, error_files