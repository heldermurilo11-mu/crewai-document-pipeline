"""In-memory and page-level PDF rotation helpers using pypdf / PyMuPDF."""

import io
import logging
from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Applies page rotations cleanly without modifying original source files."""

    @staticmethod
    def rotate_pdf(pdf_bytes: bytes, degrees: int, page_index: int | None = None) -> bytes:
        """Rotate all pages (or a single page) by *degrees* clockwise."""
        if degrees == 0:
            return pdf_bytes
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        for idx, page in enumerate(reader.pages):
            if page_index is None or idx == page_index:
                page.rotate(degrees)
            writer.add_page(page)
        output = io.BytesIO()
        writer.write(output)
        logger.debug("PDF rotated in-memory by %s degrees (page_index=%s)", degrees, page_index)
        return output.getvalue()
