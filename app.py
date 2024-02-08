# app.py

# Import necessary modules
from flask import render_template, Flask, request
from edge_app import pred_at_edge
import time

# Initialize Flask application
 #creates a Flask web application instance, and the __name__ argument is used to set up the proper paths and configurations based on the location of the script in which this line is present.
app = Flask(__name__)             

# Define human-readable class names
# A dictionary mapping numerical class indices to human-readable skin disease names.
# Dictionary
SKIN_CLASSES = {
  0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
  1: 'Basal Cell Carcinoma',
  2: 'Benign Keratosis',
  3: 'Dermatofibroma',
  4: 'Melanoma',
  5: 'Melanocytic Nevi',
  6: 'Vascular skin lesion'
}

# Define route for the home page
@app.route('/')
def index():
    return render_template('index.html', title='Home')

# Define route for handling file uploads
@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    # Record the start time for measuring the processing time
    start = time.time()

    # Check if the request method is POST
    if request.method == 'POST':
        # Retrieve the uploaded skin image file from the request
        skin_image = request.files['file']

        # Save the skin image to a specified path
        path = 'static/data/' + skin_image.filename
        skin_image.save(path)
        
        # Call the pred_at_edge function to get predictions
        disease, accuracy = pred_at_edge(path)

    # Record the end time after processing
    end = time.time()

    # Render the predictions using human-readable class names
    disease_name = SKIN_CLASSES.get(disease, 'Unknown Skin Disease')
    return render_template('uploaded.html', title='Success', predictions=disease_name, acc=accuracy, img_file=skin_image.filename, time_diff=end - start)

# Run the Flask application if this script is the main entry point
if __name__ == "__main__":
    app.run()
