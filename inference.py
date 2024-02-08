# network.py
import os
from openvino.inference_engine import IENetwork, IECore
from openvino.inference_engine import IECore
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

    def load_model(self, model_xml, device="CPU", cpu_extension=None):
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.plugin = IECore()

        # if cpu_extension and "CPU" in device:
        #     self.load_extension(cpu_extension, device)

        try:
            self.network = IENetwork(model=model_xml, weights=model_bin)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        try:
            self.exec_network = self.plugin.load_network(network=self.network, device_name=device)
        except Exception as e:
            print(f"Error loading network into the plugin: {e}")
            return

        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))


    # def load_model(self, model_xml, device="CPU", cpu_extension=None):
    #     model_bin = os.path.splitext(model_xml)[0] + ".bin"
    #     self.plugin = IECore()

    #     if cpu_extension and "CPU" in device:
    #         self.load_extension(cpu_extension, device)

    #     try:
    #         self.network = IENetwork(model=model_xml, weights=model_bin)
    #     except Exception as e:
    #         print(f"Error loading model: {e}")
    #         return

    #     try:
    #         self.exec_network = self.plugin.load_network(network=self.network, device_name=device)
    #     except Exception as e:
    #         print(f"Error loading network into the plugin: {e}")
    #         return

    #     self.input_blob = next(iter(self.network.inputs))
    #     self.output_blob = next(iter(self.network.outputs))


    # def load_model(self, model_xml, device="CPU", cpu_extension=None):
    #     model_bin = os.path.splitext(model_xml)[0] + ".bin"
    #     self.plugin = IECore()

    #     if cpu_extension and "CPU" in device:
    #         self.load_extension(cpu_extension, device)

    #     # try:
    #     #     self.network = IENetwork(model=model_xml, weights=model_bin)
    #     # except Exception as e:
    #     #     print(f"Error loading model: {e}")
    #     #     return
            
    #     try:
    #         self.network = IENetwork(model=model_xml)
    #     except Exception as e:
    #         print(f"Error loading model: {e}")
    #         return

    #     try:
    #         self.network.load_model(model=model_bin)
    #     except Exception as e:
    #         print(f"Error loading model weights: {e}")
    #         return
            
    #     # try:
    #     #     self.network = IENetwork(model=model_xml)
    #     #     self.network.load_model(model=model_bin)
    #     # except Exception as e:
    #     #     print(f"Error loading model: {e}")
    #     #     return


    #     try:
    #         self.exec_network = self.plugin.load_network(network=self.network, device_name=device)
    #     except Exception as e:
    #         print(f"Error loading network into the plugin: {e}")
    #         return

    #     self.input_blob = next(iter(self.network.inputs))
    #     self.output_blob = next(iter(self.network.outputs))

    def load_extension(self, extension, device):
        if extension and "CPU" in device:
            try:
                self.plugin.add_extension(extension, device)
                print(f"Extension {extension} loaded for device {device}")
            except Exception as e:
                print(f"Error loading extension: {e}")
    
    def is_network_loaded(self):
        return self.exec_network is not None

    def get_input_shape(self):
        return self.network.inputs[self.input_blob].shape
    
    def preprocessing(self, input_image, height, width):
        image = np.copy(input_image)

        # Ensure that the image has three channels (for RGB images)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = cv2.resize(image, (width, height))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, height, width)
        return image


    # def preprocessing(self, input_image, height, width):
    #     image = np.copy(input_image)
    #     image = cv2.resize(image, (width, height))
    #     image = image.transpose((2, 0, 1))
    #     image = image.reshape(1, 3, height, width)
    #     return image

    def async_inference(self, image):
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})

    def wait(self):
        status = self.exec_network.requests[0].wait(-1)
        return status

    def extract_output(self, apply_softmax=True):
        raw_scores = self.exec_network.requests[0].outputs[self.output_blob]

        if apply_softmax:
            softmax_scores = self.softmax(raw_scores)
            return softmax_scores
        else:
            return raw_scores

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)


# # inference.py

# import os
# from openvino.inference_engine import IENetwork, IECore
# import cv2
# import numpy as np


# class Network:
#     # Initialize variables for Inference Engine components
#     def __init__(self):
#         # Inference Engine plugin : attribute of the Network class 
#         self.plugin = None                  

#         # Neural network model ... sets the self.network attribute to None when an instance of the Network class is create
#         self.network = None      

#         self.input_blob = None              # Input layer of the network
#         self.output_blob = None             # Output layer of the network
#         self.exec_network = None            # Executable network
#         self.infer_request = None           # Inference request

#     # updated
#     def load_model(self, model_xml, device="CPU", cpu_extension=None):
#         # Construct paths for the model XML and BIN files
#         model_bin = os.path.splitext(model_xml)[0] + ".bin"

#         # Initialize the Inference Engine plugin
#         self.plugin = IECore()

#         # Add a CPU extension, if applicable
#         if cpu_extension and "CPU" in device:
#             self.plugin.add_extension(cpu_extension, device)

#         # Read the IR (Intermediate Representation) as an IENetwork
#         try:
#             self.network = IENetwork(model=model_xml, weights=model_bin)
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             return

#         # Load the IENetwork into the plugin
#         try:
#             self.exec_network = self.plugin.load_network(network=self.network, device_name=device)
#         except Exception as e:
#             print(f"Error loading network into the plugin: {e}")
#             return

#         # Get the input and output layer
#         self.input_blob = next(iter(self.network.inputs))
#         self.output_blob = next(iter(self.network.outputs))

#         return


#     # def load_model(self, model, device="CPU", cpu_extension=None):
#     #     # Load the specified model into the Inference Engine

#     #     # Construct paths for the model XML and BIN files
#     #     # Path of xml file stored in model in edge_app.py which is assigned to model_xml here
#     #     model_xml = model

#     #     #  binary weights file has the same filename as the XML file.
#     #     # os.path.splitext is a method in the os.path module in Python that splits the given path into a tuple containing the root and extension.
#     #     # The [0] index is used to access the first element of the tuple returned by os.path.splitext(model_xml), which is the root part of the path.
#     #     # The binary weights file extension (".bin") is appended to the root part of the path obtained from the XML file.
#     #     model_bin = os.path.splitext(model_xml)[0] + ".bin"

#     #     # Initialize the Inference Engine plugin
#     #     # IECore is a class from the Inference Engine API, and it is used to initialize the Inference Engine.
#     #     self.plugin = IECore()

#     #     # Add a CPU extension, if applicable
#     #     # Extensions are used to enable specific hardware optimizations or support custom layers.
#     #     if cpu_extension and "CPU" in device:
#     #         self.plugin.add_extension(cpu_extension, device)

#     #     # Read the IR (Intermediate Representation) as an IENetwork
#     #     # An instance of IENetwork is created by reading the model's XML and BIN files (Intermediate Representation) using the provided paths (model_xml and model_bin).
#     #     # This  used to create an instance of the IENetwork class from the Inference Engine (IE) API. The IENetwork class represents a neural network model in the Intermediate Representation (IR) format, which consists of a model XML file and a binary weights file.
#     #     # constructor is provided with two arguments:
#     #     # model: The path to the XML file of the neural network model.
#     #     # weights: The path to the binary weights file associated with the model.
#     #     # self.network = IENetwork(model=model_xml, weights=model_bin)
#     #     self.network = IENetwork(model=model_xml)
#     #     self.network.read_weights(model_bin)
        

#     #     # Load the IENetwork into the plugin
#     #     # The IENetwork instance is loaded into the Inference Engine plugin using the load_network method. This creates an executable network (self.exec_network) that can be used for inference.
#     #     # The newly created IENetwork instance is assigned to the self.network attribute of the class instance.
#     #     # self.network is an instance attribute used to store the IENetwork representation of the loaded model.
#     #     # self.exec_network = self.plugin.load_network(self.network, device)
#     #     # self.exec_network = self.plugin.load_network(network=self.network, device_name=device)

#     #     try:
#     #         self.exec_network = self.plugin.load_network(network=self.network, device_name=device)
#     #     except Exception as e:
#     #         print(f"Error loading network: {e}")
#     #         return

#     #     # Get the input and output layer 
#     #     # input and output layer names of the neural network are obtained using next(iter(self.network.inputs)) and next(iter(self.network.outputs))
#     #     # self.network.inputs is a dictionary that maps input layer names to their corresponding data.
#     #     # iter(self.network.inputs) creates an iterator over the keys (layer names) of the inputs dictionary.
#     #     # next(...) retrieves the first element (input layer name) from the iterator.
#     #     # self.input_blob is then assigned the name of the first input layer of the neural network.
#     #     self.input_blob = next(iter(self.network.inputs))

#     #     # self.network.outputs is a dictionary mapping output layer names to their corresponding data.
#     #     # iter(self.network.outputs) creates an iterator over the keys (output layer names) of the outputs dictionary.
#     #     # next(...) retrieves the first element (output layer name) from the iterator.
#     #     # self.output_blob is then assigned the name of the first output layer of the neural network.
#     #     self.output_blob = next(iter(self.network.outputs))

#     #     return

#     def get_input_shape(self):
#         # Gets the input shape of the network/The method returns the input shape of the network.
#         # self.network.inputs is a dictionary that maps input layer names to their corresponding data.
#         # self.input_blob is the name of the input layer obtained earlier (e.g., by next(iter(self.network.inputs))).
#         # self.network.inputs[self.input_blob] retrieves information about the input layer, including its shape.
#         # The .shape attribute is used to extract the shape of the input layer.
#         # The shape typically includes dimensions such as batch size, number of channels, height, and width.
#         return self.network.inputs[self.input_blob].shape

#     def preprocessing(self, input_image, height, width):
#         # Preprocess the input image before inference
#         # Implement your preprocessing steps here
#         # Example: Black-hat transformation, image inpainting, Grab Cut segmentation
#         # Example: Resize the image and transpose dimensions

#         # Create a copy of the input image to avoid modifying the original
#         # A copy of the input image is created using NumPy's np.copy() function.
#         image = np.copy(input_image)   

#         # Resize the image to the specified height and width using OpenCV      
#         # Resizing the image is a common preprocessing step to ensure that it matches the dimensions expected by the neural network.               
#         image = cv2.resize(image, (width, height))             
        
#         # Transpose the image dimensions to match the expected format
#         # The original format is usually H x W x C (height x width x channels)
#         # Transposing to C x H x W (channels x height x width)
#         image = image.transpose((2, 0, 1))    

#         # Add a batch dimension to the image to match the network input shape
#         # The resulting shape becomes 1 x C x H x W ,where 3 represents the number of color channels (common for RGB images).                 
#         image = image.reshape(1, 3, height, width)

#         # The preprocessed image is returned.
#         return image

#     def async_inference(self, image):
#         # Makes an asynchronous inference request, given an input image.
#         # self.exec_network is an instance attribute representing the executable network, which is obtained after loading the neural network model into the Inference Engine.
#         # start_async is a method of the executable network that initiates an asynchronous inference request.
#         # request_id=0 specifies the identifier for the inference request. In this case, it's set to 0.
#         # inputs={self.input_blob: image} specifies the input data for the inference request. self.input_blob is the name of the input layer, and image is the input data.
#         self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
#         return

#     def wait(self):
#         # Wait for the completion of the asynchronous inference request.
#         # Check the status of the inference request
#         # self.exec_network.requests is a list of inference requests associated with the executable network (self.exec_network).
#         # [0] is used to access the first (and typically only) inference request in the list.
#         # wait(-1) is a method that blocks until the inference request is complete or until the specified timeout is reached.
#         # The timeout value of -1 indicates an indefinite wait, meaning the method will block until the inference is complete.
#         # The status variable holds information about whether the inference request completed successfully or if there were any errors.
#         status = self.exec_network.requests[0].wait(-1)
#         return status

#     def extract_output(self, apply_softmax=True):
#         # Extract and post-process the output from the network

#         # Returns a list of the results for the output layer of the network.
#         # Get the raw scores from the output layer of the network

#         #         self.exec_network.requests is a list of inference requests associated with the executable network (self.exec_network).
#         # [0] is used to access the first (and typically only) inference request in the list.
#         # .outputs[self.output_blob] accesses the output of the neural network associated with the specified output layer (self.output_blob).
#         # The variable raw_scores holds the raw output scores from the specified output layer.
#         # These scores are the numerical values produced by the neural network before any post-processing.
#         raw_scores = self.exec_network.requests[0].outputs[self.output_blob]

#         # This conditional checks whether to apply softmax to the raw scores.
#         # Softmax is a mathematical function that converts raw scores into probabilities, making it easier to interpret the output as a probability distribution over classes.
#         # If apply_softmax is True, the softmax method is called to apply softmax to the raw scores.
#         # The method returns either the softmax-transformed scores (probabilities) or the raw scores, depending on the value of apply_softmax.
#         if apply_softmax:
#             # Apply softmax to convert raw scores to probabilities
#             softmax_scores = self.softmax(raw_scores)
#             return softmax_scores
#         else:
#             return raw_scores

#     def softmax(self, x):
#         # Applies the softmax function to a vector x
#         # np.exp calculates the element-wise exponential of a NumPy array.
#         # x - np.max(x) is used for numerical stability. Subtracting the maximum value of x ensures that large values don't result in extremely large exponentials that might lead to numerical overflow.
#         # exp_x is the array of exponentials calculated in the previous step.
#         # exp_x.sum(axis=-1, keepdims=True) calculates the sum along the last axis while keeping the dimensions. This sum is necessary for the normalization step in softmax.
#         # The method returns the result of the softmax transformation.
#         exp_x = np.exp(x - np.max(x))
#         return exp_x / exp_x.sum(axis=-1, keepdims=True)
