import PyPDF2

file_path = "kampus.pdf"  # ganti sesuai nama file PDF-mu

text = ""
with open(file_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for page in reader.pages:
        text += page.extract_text()

print(text)
