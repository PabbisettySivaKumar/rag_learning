import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader, FileSystemBlobLoader
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from loclaWishperParser import FasterwhisperParser
from faster_whisper import WhisperModel



#from PDF

loader= PyPDFLoader('/Users/sivakumar/projects/rag_learning/docs/andrew_ng_ml_lnotes.pdf')
pages= loader.load()

print(len(pages))
page= pages[1]
print(page.page_content)
print(page.metadata)

#from youtube

url= 'https://www.youtube.com/watch?v=1km00uTv1aA'
save_dir= '/Users/sivakumar/projects/rag_learning/docs/youtube/'
y_loader= YoutubeAudioLoader([url], save_dir)
blobs= list(y_loader.yield_blobs())

blob_loader= FileSystemBlobLoader(save_dir, glob="*.m4a")

loader= GenericLoader(
    blob_loader,
    FasterwhisperParser(model_size='small', device='cpu', file_path=save_dir) 
    )

docs= loader.load()
print(docs[0].page_content[:500])
