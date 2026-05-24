from typing import List

from pypdf import PdfReader

def get_all_text_from_pdf(path) -> str:
    reader = PdfReader(path)
    text = ""
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        text += page.extract_text()

    return text

def split_text_into_by_sentences(text) -> List[str]:
    sentences = text.split('.')
    return sentences

def join_sentences_into_chunks(sentences: List[str], chunk_size=500, overlap_percentage=0.1) -> List[str]:
    chunks = []
    current_chunk = ""
    if overlap_percentage < 0 or overlap_percentage > 1:
        raise ValueError("Overlap percentage must be between 0 and 1.")

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + '.'
        else:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[int(-overlap_percentage*len(current_chunk)):] + sentence + '.'

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

