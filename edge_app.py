# edge_app.py

import cv2
import numpy as np
from inference import Network

# from openvino.inference_engine import IECore
from openvino.inference_engine import IENetwork, IECore

# CPU extension path
import os

# Assuming you have sourced setupvars.sh
openvino_dir = '/opt/intel/openvino_2023.2.0/'


# Check if the INTEL_OPENVINO_DIR environment variable is set
if 'INTEL_OPENVINO_DIR' in os.environ:
    openvino_dir = os.environ['INTEL_OPENVINO_DIR']
    CPU_EXTENSION = os.path.join(openvino_dir, 'deployment_tools', 'inference_engine', 'lib', 'intel64', 'libcpu_extension_sse4.so')
else:
    # Use a default path if the environment variable is not set
    CPU_EXTENSION = "/default/path/to/libcpu_extension_sse4.so"

# Update the CPU_EXTENSION path
# plugin.load_model(MODEL, "CPU")
# net_input_shape = plugin.get_input_shape()
# CPU_EXTENSION = "/opt/intel/openvino_2023.2.0/python/openvino"
# CPU_EXTENSION = "/Users/kashishkhatri/Desktop/Research/research_skin/openvino/scripts/setupvars"
# CPU_EXTENSION = "/Users/kashishkhatri/openvino_env/lib/python3.10/site-packages/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
# CPU_EXTENSION = os.path.abspath(cpu_extension_path)

# Path of converted skin disease model in XML
MODEL = "model/model_tf.xml"

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

    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, 3, height, width)

    return image

def pred_at_edge(input_img):
    # Initialize the Inference Engine
    plugin = Network()

    # Load the network model into the IE
    plugin.load_model(MODEL, "CPU", CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Reading input image
    img = cv2.imread(input_img, cv2.IMREAD_COLOR)

    # Pre-process the image
    expand_img = preprocessing(img, net_input_shape[2], net_input_shape[3])
    final_img = np.expand_dims(expand_img, axis=0)

    # Perform inference on the image
    plugin.async_inference(final_img)

    # Get the output probabilities after applying softmax
    if plugin.wait() == 0:
        softmax_probs = plugin.extract_output()

        # Find the index of the class with the highest probability
        pred_class = np.argmax(softmax_probs)
        disease = SKIN_CLASSES[pred_class]
        accuracy = softmax_probs[0][pred_class]

        print(disease, accuracy)
        return disease, accuracy
