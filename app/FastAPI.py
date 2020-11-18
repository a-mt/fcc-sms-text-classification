#!/usr/bin/env python
# coding: utf-8

#+---------------------------------------------------------
#| LOAD LIBRARIES
#+---------------------------------------------------------

import cloudpickle as pickle
import re
import numpy as np

from tflite_runtime.interpreter import Interpreter
from keras_preprocessing.text import Tokenizer
from keras_preprocessing import sequence
from wn.morphy import _morphy

#+---------------------------------------------------------
#| LOAD MODEL & DEPENDENCIES
#+---------------------------------------------------------

path = "resources/"

# Load model
interpreter = Interpreter(
    model_path=path + 'fcc_sms_classification.tflite'
)
interpreter.allocate_tensors()

class objectview(object):
    """
    Trick to access dictionary items as object attributes
    """
    def __init__(self, d):
        self.__dict__ = d

with open(path + "utils.pkl", "rb") as f:
    utils = objectview(pickle.load(f))

#+---------------------------------------------------------
#| PREPROCESSING
#+---------------------------------------------------------

def lemmatize(word, pos='n'):
    '''
    Parts of speech constants:
    ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    '''
    lemmas = _morphy(word, pos)
    return min(lemmas, key=len) if lemmas else word

def cleanup(txt):
    txt = re.sub(r'([^\s\w])+', ' ', txt)
    txt = " ".join([lemmatize(word) for word in txt.split()
                    if not word in utils.stopwords_eng])
    txt = txt.lower()
    return txt

max_len = 500

def preprocessing(X):
    return utils.pad_sequences(
        utils.texts_to_sequences([cleanup(x) for x in X]),
        maxlen=max_len)

#+---------------------------------------------------------
#| PREDICT
#+---------------------------------------------------------

def predict(X):
    input_index  = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    input_data = np.array(X, dtype=np.float32)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    return interpreter.get_tensor(output_index)

#+---------------------------------------------------------
#| FASTAPI SETTINGS
#+---------------------------------------------------------

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Using Pydantic BaseModel class for automatic data validation
class Data(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": "you have won £1000 cash! call to claim your prize."
            }
        }

# Defining Response format for documentation
class Response(BaseModel):
    p: float
    prediction: str

    class Config:
        schema_extra = {
            "example": {
                "prediction": "ham",
                "p": 0
            }
        }

app = FastAPI(debug=True)

@app.post("/predict", response_model=Response)
def app_predict(data: Data):
    try:
        text = data.text
        res  = predict(preprocessing([text]))[0]
        p    = float(res[0])

        return {"prediction": ("ham" if p<0.5 else "spam"), "p": p}
    except Exception as e:
        print(e)
        return {"prediction" : "error"}
    
@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'OK'

