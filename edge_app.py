# edge_app.py
import cv2
import numpy as np
from inference import Network
import sys

# from openvino.inference_engine import IECore
from openvino.inference_engine import IENetwork, IECore
# from openvino import inference_engine as ie

# CPU extension path
import os

# Assuming you have sourced setupvars.sh
# openvino_dir = '/opt/intel/openvino_2023.2.0/'
# openvino_dir = '/Users/kashishkhatri/Desktop/Research/research_skin/openvino/'
openvino_dir  = '/opt/intel/openvino_2023/'

# Check if the INTEL_OPENVINO_DIR environment variable is set
# if 'INTEL_OPENVINO_DIR' in os.environ:
#     openvino_dir = os.environ['INTEL_OPENVINO_DIR']
#     CPU_EXTENSION = os.path.join(openvino_dir, 'deployment_tools', 'inference_engine', 'lib', 'intel64', 'libcpu_extension_sse4.so')
# else:
#     # Use a default path if the environment variable is not set
#     CPU_EXTENSION = "/default/path/to/libcpu_extension_sse4.so"

# Update the CPU_EXTENSION path
# plugin.load_model(MODEL, "CPU")
# net_input_shape = plugin.get_input_shape()
# CPU_EXTENSION = "/opt/intel/openvino_2023.2.0/python/openvino"
# CPU_EXTENSION = "/Users/kashishkhatri/Desktop/Research/research_skin/openvino/scripts/setupvars"
# CPU_EXTENSION = "/Users/kashishkhatri/openvino_env/lib/python3.10/site-packages/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
# CPU_EXTENSION = os.path.abspath(cpu_extension_path)
# CPU_EXTENSION = "/Users/kashishkhatri/anaconda3/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_batch_plugin.so "
CPU_EXTENSION = "/Users/kashishkhatri/anaconda3/lib/python3.10/site-packages/openvino/libs/libopenvino_auto_batch_plugin.so"
# CPU_EXTENSION = "python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm"

# Path of converted skin disease model in XML
MODEL = "model/model_tf.xml"
# MODEL = "/Users/kashishkhatri/Desktop/Research/research_skin/skin-disease-detection-edge/model/model_tf.xml"

# mapping numerical class indices to human-readable skin disease names.
SKIN_CLASSES = {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'
}

def preprocessing(input_image, height, width):
    # Implement your preprocessing steps here
    # Example: Black-hat transformation, image inpainting, Grab Cut segmentation
    # ...

    # Create a copy of the input image
    image = np.copy(input_image)

    # Resize the image to the specified height and width
    image = cv2.resize(image, (width, height))

    # Transpose the image dimensions to match the expected format
    image = image.transpose((2, 0, 1))

    # Add a batch dimension to the image
    image = image.reshape(1, 3, height, width)

    return image


# performs image inference using a neural network
def pred_at_edge(input_img):

    # Initialize the Inference Engine
    # create an instance of the Network class and assign it to the variable plugin. 
    plugin = Network()                  #network() input from package

    # Load the network model into the IE
    # loads a pre-trained neural network model specified by the MODEL variable onto the Inference Engine(plugin)
    # plugin.load_model(MODEL, "CPU", CPU_EXTENSION)

    # code
    # plugin.load_model(MODEL, "CPU")
    # plugin.load_extension(CPU_EXTENSION, "CPU")
    # plugin.load_model(MODEL, "CPU")

    if not plugin.is_network_loaded():
        print("Error loading the network. Exiting.")
        sys.exit(1)



    #  this line retrieves the shape of the input expected by the neural network
    net_input_shape = plugin.get_input_shape()

    # Reading input image
    # Reads the input image (input_img) using OpenCV's cv2.imread function, assuming it is a color image (cv2.IMREAD_COLOR).
    # img = cv2.imread(input_img, cv2.IMREAD_COLOR)

     # Retrieve the network input shape
    net_input_shape = plugin.get_input_shape()
    height, width = net_input_shape[2], net_input_shape[3]

    # Reading input image
    img = cv2.imread(input_img, cv2.IMREAD_COLOR)

    # Pre-process the image
    expand_img = preprocessing(img, height, width)


    # Pre-process the image
    # Calls the preprocessing function to prepare the input image for inference. It resizes, transposes, and adds a batch dimension to the image. The result is stored in final_img.
    # expand_img = preprocessing(img, net_input_shape[2], net_input_shape[3])

    # Expands the dimensions of the pre-processed image to add a batch dimension, making it suitable for inference.
    # function from NumPy to add a new dimension to the array 
    # axis=0 argument specifies that the new dimension should be added at the beginning of the array, effectively creating a batch dimension.
    #eg: if image is 3D it is converted to 4D
    final_img = np.expand_dims(expand_img, axis=0)

    # Perform inference on the preprocessed image
    plugin.async_inference(final_img)

    # Get the output probabilities after applying softmax
    # Waits for the inference to complete using the wait method.
    if plugin.wait() == 0:

        # Extracts the output probabilities after applying softmax activation.
        softmax_probs = plugin.extract_output()

        # Find the index of the class with the highest probability
        # using NumPy to find the index of the element with the maximum value in the array softmax_probs
        pred_class = np.argmax(softmax_probs)

        # Get the predicted disease and its associated accuracy
        # Retrieves the predicted disease class based on the index.
        # maps class indices to class labels. 
        # translating the numerical index of the predicted class into a human-readable class label.
        disease = SKIN_CLASSES[pred_class]                  
        accuracy = softmax_probs[0][pred_class]

        print(disease, accuracy)
        return disease, accuracy
