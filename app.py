# app.py

from flask import render_template, Flask, request
from edge_app import pred_at_edge
import time

app = Flask(__name__)

# Define human-readable class names
SKIN_CLASSES = {
  0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
  1: 'Basal Cell Carcinoma',
  2: 'Benign Keratosis',
  3: 'Dermatofibroma',
  4: 'Melanoma',
  5: 'Melanocytic Nevi',
  6: 'Vascular skin lesion'
}

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    start = time.time()
    if request.method == 'POST':
        skin_image = request.files['file']
        path = 'static/data/' + skin_image.filename
        skin_image.save(path)
        
        # Call the pred_at_edge function to get predictions
        disease, accuracy = pred_at_edge(path)

    end = time.time()

    # Render the predictions using human-readable class names
    disease_name = SKIN_CLASSES.get(disease, 'Unknown Skin Disease')
    return render_template('uploaded.html', title='Success', predictions=disease_name, acc=accuracy, img_file=skin_image.filename, time_diff=end - start)

if __name__ == "__main__":
    app.run()
