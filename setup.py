#!/usr/bin/env python

import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='transformers_onnx',  
     version='0.0.2',
     author="NLP Connect",
     author_email="ankur310794@gmail.com",
     description="onnx model for transformers pipeline",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/nlpconnect/transformer_onnx",
     packages=['transformers_onnx'],
     install_requires=["transformers[onnx]"], 
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent"
     ]
 )