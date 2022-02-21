#!/usr/bin/env python

import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='transformer_onnx',  
     version='0.0.1',
     author="NLP Connect",
     author_email="",
     description="onnx model for transformers pipeline",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/nlpconnect/transformer_onnx",
     packages=['transformer_onnx'],
     install_requires=["transformers[onnx]"], 
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent"
     ]
 )