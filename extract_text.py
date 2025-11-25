import fitz  # PyMuPDF
import sys
from pathlib import Path


def extract_text_from_pdf(pdf_path: str, output_path: str = None) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file
        output_path (str, optional): Path to save the extracted text. If None, only returns the text.
    Returns:
        str: Extracted text from the PDF
    """
    # Check if file exists
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        
        full_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n"
            full_text += text
        
        doc.close()
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"Text extracted and saved to: {output_path}")
        
        return full_text
    
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_text.py <pdf_file> [output_file]")
        print("\nExample:")
        print("  python extract_text.py document.pdf")
        print("  python extract_text.py document.pdf output.txt")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        text = extract_text_from_pdf(pdf_file, output_file)
        
        if not output_file:
            print("\n Extracted Text : \n")
            print(text)
        
        print(f"\nSuccessfully extracted text from {len(text)} characters")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
