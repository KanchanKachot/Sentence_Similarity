from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from Sentence_Similarity import sent_similarity
import traceback
app = FastAPI()




class Sent(BaseModel):
   sentences:str


@app.post("/sentence_similarity")
def sentence_similarity(body:Sent):
  
   result=sent_similarity(body.sentences)

   return result