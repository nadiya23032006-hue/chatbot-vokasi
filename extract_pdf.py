import PyPDF2

file_path = "kampus.pdf"
output_file = "kampus.txt"

with open(file_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

# Simpan ke file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Teks berhasil disimpan ke {output_file}")
