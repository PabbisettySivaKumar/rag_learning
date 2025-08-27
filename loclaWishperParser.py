from langchain.schema import Document
from langchain_community.document_loaders.base import BaseLoader
from faster_whisper import WhisperModel
import os

class  FasterwhisperParser(BaseLoader):
    def __init__(self, file_path, model_size="tiny", device="metal", compute_type="int8"):
        self.file_path = file_path
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def lazy_parse(self,blob):
        segments, _ = self.model.transcribe(blob.path)
        text = " ".join([segment.text for segment in segments])
        return [Document(page_content=text, metadata={"source": os.path.basename(self.file_path)})]
