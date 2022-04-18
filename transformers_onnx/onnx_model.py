from transformers import AutoModel, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

"""
Inspired by: https://github.com/neuml/txtai/blob/master/src/python/txtai/models/onnx.py
"""

# Conditional import
try:
    import onnxruntime as ort

    ONNX_RUNTIME = True
except ImportError:
    ONNX_RUNTIME = False

import numpy as np
import torch

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel

# pylint: disable=W0223
class OnnxModel(PreTrainedModel):
    """
    Provides a Transformers/PyTorch compatible interface for ONNX models. Handles casting inputs
    and outputs with minimal to no copying of data.
    """

    def __init__(self, model, config=None, task=None):
        """
        Creates a new OnnxModel.

        Args:
            model: path to model or InferenceSession
            config: path to model configuration
        """

        if not ONNX_RUNTIME:
            raise ImportError('onnxruntime is not available - install "model" extra to enable')

        super().__init__(AutoConfig.from_pretrained(config) if config else OnnxConfig())

        # Create ONNX session
        self.model = ort.InferenceSession(model, ort.SessionOptions(), self.providers())

        # set pooler or sequence output flag
        self.task = task
        # Add references for this class to supported AutoModel classes
        Registry.register(self)

    def providers(self):
        """
        Returns a list of available and usable providers.

        Returns:
            list of available and usable providers
        """

        # Create list of providers, prefer CUDA provider if available
        # CUDA provider only available if GPU is available and onnxruntime-gpu installed
        if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Default when CUDA provider isn't available
        return ["CPUExecutionProvider"]

    def forward(self, **inputs):
        """
        Runs inputs through an ONNX model and returns outputs. This method handles casting inputs
        and outputs between torch tensors and numpy arrays as shared memory (no copy).

        Args:
            inputs: model inputs

        Returns:
            model outputs
        """

        inputs = self.parse(inputs)
        if "attention_mask" not in inputs and self.task=="token-classification":
          inputs['attention_mask'] = np.array([[1]*len(i) for i in inputs['input_ids']])

        # Run inputs through ONNX model
        results = self.model.run(None, inputs)

        # pylint: disable=E1101
        # Detect if logits is an output and return classifier output in that case
        model_keys = [x.name for x in self.model.get_outputs()]
        if self.task=="sequence-classification":
            return SequenceClassifierOutput(logits=torch.from_numpy(np.array(results[0])))

        elif self.task in ["feature-extraction-pooler_output", "pooler_output"]:
          results = results[1]
        
        elif self.task in ["feature-extraction" ,"feature-extraction-last_hidden_state", "last_hidden_state"]:
          results = results[0]

        results = torch.from_numpy(np.array(results))
        #print("results :", results.shape, results)
        return results

    def parse(self, inputs):
        """
        Parse model inputs and handle converting to ONNX compatible inputs.

        Args:
            inputs: model inputs

        Returns:
            ONNX compatible model inputs
        """

        features = {}

        # Select features from inputs
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in inputs:
                value = inputs[key]

                # Cast torch tensors to numpy
                if hasattr(value, "cpu"):
                    value = value.cpu().numpy()

                # Cast to numpy array if not already one
                features[key] = np.asarray(value)

        return features


class OnnxConfig(PretrainedConfig):
    """
    Configuration for ONNX models.
    """

class Registry:
    """
    Methods to register models and fully support pipelines.
    """

    @staticmethod
    def register(model, config=None):
        """
        Registers a model with auto model and tokenizer configuration to fully support pipelines.

        Args:
            model: model to register
            config: config class name
        """

        # Default config class name to model name if not provided
        name = model.__class__.__name__
        if not config:
            config = name

        # Default model config_class if empty
        if hasattr(model.__class__, "config_class") and not model.__class__.config_class:
            model.__class__.config_class = config

        # Add references for this class to supported AutoModel classes
        for mapping in [AutoModel, AutoModelForQuestionAnswering, AutoModelForSequenceClassification]:
            mapping.register(config, model.__class__)

        # Add references for this class to support pipeline AutoTokenizers
        if hasattr(model, "config") and type(model.config) not in TOKENIZER_MAPPING:
            TOKENIZER_MAPPING.register(type(model.config), type(model.config).__name__)
