from agentic_doc.parse import parse
import os

class PDFScraper:
    def extract_text_from_pdf_landingai(self, file_path: str):
        """Extract text from PDF using Landing AI"""
        try:
            if parse is None:
                raise ImportError("agentic_doc.parse not available")

            result = parse(file_path)
            if result and len(result) > 0:
                parsed_doc = result[0]
                print("Extracted content successfully")
                print(f"Content preview: {str(result)[:500]}...")
                return result[0].markdown
            else:
                print("No content extracted from PDF")
                return ""
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
