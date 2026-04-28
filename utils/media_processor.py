"""
Media Processing Module
Handles images and documents (PDFs)
"""

from PIL import Image
import PyPDF2
import os
import logging

logger = logging.getLogger(__name__)

class MediaProcessor:
    """Process images and documents"""
    
    async def process_image(self, image_path: str) -> str:
        """
        Process and describe image
        Returns basic image info (can be enhanced with vision API)
        """
        try:
            with Image.open(image_path) as img:
                # Get basic info
                width, height = img.size
                format_name = img.format
                mode = img.mode
                
                description = (
                    f"Image details: {width}x{height}px, "
                    f"Format: {format_name}, Mode: {mode}"
                )
                
                # You can integrate Google Vision API or similar here
                # For now, returning basic info
                
                logger.info(f"Processed image: {description}")
                return description
                
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return "Could not process image"
    
    async def process_document(self, doc_path: str, filename: str) -> str:
        """
        Extract text from PDF or text documents
        Returns: Extracted text content
        """
        try:
            # Handle PDF files
            if filename.lower().endswith('.pdf'):
                return await self._extract_pdf_text(doc_path)
            
            # Handle text files
            elif filename.lower().endswith(('.txt', '.md', '.py', '.json')):
                with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return content
            
            else:
                return f"Unsupported file type: {filename}"
                
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return None
    
    async def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text_content = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                # Limit to first 10 pages to avoid huge content
                max_pages = min(num_pages, 10)
                
                for page_num in range(max_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    text_content.append(text)
                
                full_text = '\n'.join(text_content)
                
                if num_pages > 10:
                    full_text += f"\n\n[Note: Only first 10 of {num_pages} pages shown]"
                
                logger.info(f"Extracted {len(full_text)} characters from PDF")
                return full_text
                
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return None
