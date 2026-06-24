import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
@dataclass
class Document:
    doc_id: str
    category: str
    filename: str
    raw_text: str
    clean_text: str = ""
  def load_dataset(data_dir: str, max_docs: Optional[int] = None) -> List[Document]:
    """
    Load all documents from the dataset.
    """
    pass