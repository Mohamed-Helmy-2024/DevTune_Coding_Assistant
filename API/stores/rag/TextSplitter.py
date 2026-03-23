from typing import List
import re
import logging
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None


class TextChunk:
    """Class to represent a text chunk with metadata"""
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}


class TextSplitter:
    """Class for splitting text into chunks for embeddings"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize TextSplitter

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        if RecursiveCharacterTextSplitter is not None:
            self.lc_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
        else:
            self.lc_splitter = None

    def split_text(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """
        Split text into chunks

        Args:
            text: Text to split
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of TextChunk objects
        """
        if not text or len(text) == 0:
            return []

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        chunks = []
        start = 0

        # If langchain splitter is available, use it for more robust splitting
        if self.lc_splitter is not None:
            # LangChain returns string chunks; we convert to TextChunk
            chunks = []
            for i, chunk_text in enumerate(self.lc_splitter.split_text(text)):
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    'chunk_index': i,
                    'chunk_size': len(chunk_text)
                })
                chunks.append(TextChunk(content=chunk_text, metadata=chunk_metadata))
            return chunks

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # If not at the end, try to break at sentence or word boundary
            if end < len(text):
                # Try to find sentence boundary (. ! ?)
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )

                if sentence_end != -1 and sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Try to find word boundary (space)
                    space_pos = text.rfind(' ', start, end)
                    if space_pos != -1 and space_pos > start:
                        end = space_pos

            # Extract chunk
            chunk_text = text[start:end].strip()

            if chunk_text:
                # Create metadata for this chunk
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    'chunk_index': len(chunks),
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_size': len(chunk_text)
                })

                chunks.append(TextChunk(content=chunk_text, metadata=chunk_metadata))

            # Move to next chunk with overlap
            start = end - self.chunk_overlap if end < len(text) else end

            # Prevent infinite loop
            if start <= 0:
                start = 1

        return chunks

    def split_by_sentences(self, text: str, metadata: dict = None, max_sentences: int = 5) -> List[TextChunk]:
        """
        Split text by sentences

        Args:
            text: Text to split
            metadata: Optional metadata
            max_sentences: Maximum number of sentences per chunk

        Returns:
            List of TextChunk objects
        """
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence exceeds limits, save current chunk
            if len(current_chunk) >= max_sentences or current_size + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update({
                        'chunk_index': len(chunks),
                        'num_sentences': len(current_chunk)
                    })
                    chunks.append(TextChunk(content=chunk_text, metadata=chunk_metadata))

                # Start new chunk
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)

        # Add remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                'chunk_index': len(chunks),
                'num_sentences': len(current_chunk)
            })
            chunks.append(TextChunk(content=chunk_text, metadata=chunk_metadata))

        return chunks

    def split_by_paragraphs(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """
        Split text by paragraphs

        Args:
            text: Text to split
            metadata: Optional metadata

        Returns:
            List of TextChunk objects
        """
        # Split by double newline (paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph exceeds chunk size, save current chunk
            if current_size + len(paragraph) > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    'chunk_index': len(chunks),
                    'num_paragraphs': len(current_chunk)
                })
                chunks.append(TextChunk(content=chunk_text, metadata=chunk_metadata))

                # Start new chunk
                current_chunk = [paragraph]
                current_size = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_size += len(paragraph)

        # Add remaining paragraphs
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                'chunk_index': len(chunks),
                'num_paragraphs': len(current_chunk)
            })
            chunks.append(TextChunk(content=chunk_text, metadata=chunk_metadata))

        return chunks
