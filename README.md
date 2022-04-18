# transformer_onnx

`transformers_onnx` is a simple package which can use inside transformers pipeline.

# Install

    pip install transformers_onnx

# Convert model into Onnx format

    #for question-answering
    python -m transformers.onnx --feature "question-answering" -m nlpconnect/roberta-base-squad2-nq ./qa/

    #for text-classification or zeroshot classification
    python -m transformers.onnx --feature "sequence-classification" -m cross-encoder/nli-roberta-base ./classifier/

    #for feature-extraction (last_hidden_state or pooler_output)
    python -m transformers.onnx --feature "default" -m nlpconnect/dpr-ctx_encoder_bert_uncased_L-2_H-128_A-2 ./feature/

    #for token-classification
    python -m transformers.onnx --feature "token-classification" -m dslim/bert-base-NER ./ner/
    
# Use transformers_onnx to run transformers pipeline

## Question Answering

    from transformers import pipeline, AutoTokenizer, AutoConfig
    from transformer_onnx import OnnxModel

    model = OnnxModel("qa/model.onnx", task="question-answering")
    model.config = AutoConfig.from_pretrained("nlpconnect/roberta-base-squad2-nq")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/roberta-base-squad2-nq")
    qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Input data
    context = ["Released on 6/03/2021",
            "Release delayed until the 11th of August",
            "Documentation can be found here: huggingface.com"]
    # Define column queries
    queries = ["What is Released date?", "till when delayed?", "What is the url?"]
    qa(context=context, question=queries)

## Text Classification/ Zero shot classification

    from transformers import pipeline, AutoTokenizer, AutoConfig
    from transformer_onnx import OnnxModel

    model = OnnxModel("classifier/model.onnx", task="sequence-classification")
    model.config = AutoConfig.from_pretrained("cross-encoder/nli-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-roberta-base")
    zero_shot = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
    zero_shot(sequences=["Hello Hiiii", "I am playing football"], candidate_labels=["Greeting", "Sports"])

## Feature Extraction

    from transformers import pipeline, AutoTokenizer, AutoConfig
    from transformer_onnx import OnnxModel

    # for last_hidden_state
    model = OnnxModel("feature/model.onnx", task="last_hidden_state")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/dpr-ctx_encoder_bert_uncased_L-2_H-128_A-2")
    feature_extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    feature_extractor(["Hello Hiiii", "I am playing football"])

    # for pooler_output
    model = OnnxModel("feature/model.onnx", task="pooler_output")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/dpr-ctx_encoder_bert_uncased_L-2_H-128_A-2")
    feature_extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    feature_extractor(["Hello Hiiii", "I am playing football"])


# NER

    from transformers import pipeline, AutoTokenizer, AutoConfig
    from transformer_onnx import OnnxModel

    model = OnnxModel("ner/model.onnx", task="token-classification")
    model.config = AutoConfig.from_pretrained("dslim/bert-base-NER")
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    ner = pipeline("token-classification", model=model, tokenizer=tokenizer)
    ner("My name is transformers and I live in github/huggingface")