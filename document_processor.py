import os
import PyPDF2
from docx import Document as DocxDocument
from typing import List
import re

class DocumentProcessor:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def read_pdf(self, file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"--- Page {page_num + 1} ---\n"
                        text += page_text + "\n\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Failed to read PDF: {str(e)}")
    
    def read_docx(self, file_path: str) -> str:
        text = ""
        try:
            doc = DocxDocument(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Failed to read DOCX: {str(e)}")
    
    def read_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except:
                raise Exception("Failed to read text file")
        except Exception as e:
            raise Exception(f"Failed to read text file: {str(e)}")
    
    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def process_document(self, file_path: str) -> List[str]:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        print(f"Processing {file_path} ({ext})")
        
        if ext == '.pdf':
            text = self.read_pdf(file_path)
        elif ext == '.docx':
            text = self.read_docx(file_path)
        elif ext in ['.txt', '.md']:
            text = self.read_txt(file_path)
        else:
            raise Exception(f"Unsupported file type: {ext}")
        
        if not text:
            raise Exception("No text could be extracted from the document")
        
        chunks = self.chunk_text(text)
        
        print(f"Created {len(chunks)} chunks from document")
        return chunks