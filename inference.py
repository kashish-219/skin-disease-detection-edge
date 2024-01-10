# inference.py

import os
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np


class Network:

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as an IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return

    def get_input_shape(self):
        # Gets the input shape of the network
        return self.network.inputs[self.input_blob].shape

    def preprocessing(self, input_image, height, width):
        # Implement your preprocessing steps here
        # Example: Black-hat transformation, image inpainting, Grab Cut segmentation
        # ...

        image = np.copy(input_image)
        image = cv2.resize(image, (width, height))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, height, width)

        return image

    def async_inference(self, image):
        # Makes an asynchronous inference request, given an input image.
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        return

    def wait(self):
        # Checks the status of the inference request.
        status = self.exec_network.requests[0].wait(-1)
        return status

    def extract_output(self, apply_softmax=True):
        # Returns a list of the results for the output layer of the network.

        raw_scores = self.exec_network.requests[0].outputs[self.output_blob]

        if apply_softmax:
            # Apply softmax to convert raw scores to probabilities
            softmax_scores = self.softmax(raw_scores)
            return softmax_scores
        else:
            return raw_scores

    def softmax(self, x):
        # Applies the softmax function to a vector x
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
