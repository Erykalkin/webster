try:
    import pdf2image
    print("pdf2image is available")
except ImportError:
    print("pdf2image is NOT available")

try:
    import fitz # PyMuPDF
    print("fitz is available")
except ImportError:
    print("fitz is NOT available")

try:
    import pypdfum2
    print("pypdfum2 is available")
except ImportError:
    print("pypdfum2 is NOT available")

try:
    import pdfplumber
    print("pdfplumber is available")
except ImportError:
    print("pdfplumber is NOT available")
