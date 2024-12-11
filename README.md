# CoinRepository
This repository contains a PyTorch-based training script for detecting coins in images using a Faster R-CNN model pretrained on ResNet-50 FPN. The script uses a custom dataset of annotated images, along with bounding box coordinates for the coins. The model is trained to localize and classify multiple coin types.

Contents

coin_counter_v2.py (The main training script)
data/ (Directory containing train, valid, and test image sets along with their annotations)
coin_detector.pth (The trained model weights, generated after running the script)

Requirements

Python 3.7 or higher
PyTorch 1.10 or higher
torchvision 0.11 or higher
Pillow
Pandas
Matplotlib
tqdm
NumPy
You can install most dependencies using:

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install pillow pandas matplotlib tqdm numpy
(Adjust the PyTorch installation command above depending on your environment.)

Directory Structure and Data Preparation

Before running the script, ensure your project directory is set up as follows:

project_root/
|-- coin_counter_v2.py
|-- data/
|   |-- train/
|   |   |-- image1.jpg
|   |   |-- image2.jpg
|   |   |-- ...
|   |   |-- _annotations.csv
|   |
|   |-- valid/
|   |   |-- image1.jpg
|   |   |-- ...
|   |   |-- _annotations.csv
|   |
|   |-- test/
|       |-- image1.jpg
|       |-- ...
|       |-- _annotations.csv
|
|-- coin_detector.pth (will be saved here after training)
_annotations.csv format:

Each _annotations.csv file should contain bounding box annotations for each image. Typical columns are:

filename: The image file name.
xmin, ymin, xmax, ymax: Coordinates of the bounding box.
class: The class name of the object (coin type).
Image Formats:
The code supports .jpg, .jpeg, and .png files. Make sure all your images referenced in _annotations.csv are in one of these formats.

How to Run the Script

Activate Your Environment
Make sure you have a Python environment with all the required packages. For example:

source venv/bin/activate
Set File Paths (If Necessary)
In the train.py file, you may need to adjust the following lines if your data structure differs:
train_annotations_file = 'data/train/_annotations.csv'
train_img_dir = 'data/train'

valid_annotations_file = 'data/valid/_annotations.csv'
valid_img_dir = 'data/valid'

test_annotations_file = 'data/test/_annotations.csv'
test_img_dir = 'data/test'

Make sure these paths point to the correct directories and files in your environment.
Run the Training Script
Simply run:
python train.py
This will load the datasets, initialize the model, and begin training for the number of epochs specified in the code (num_epochs variable).
The model weights will be saved as coin_detector.pth after training completes.

Training Output
During training, a progress bar will show the current epoch, iteration, and loss values.
After each epoch, an average loss for that epoch is printed.
Once training completes, you will see some visualization of predictions on training images in popup windows (if running locally with GUI support) or inline if using a Jupyter environment.

Inspecting the Results
After training, coin_detector.pth contains the trained model weights. You can load this model into another script for inference on new images. Just ensure you use the same model architecture and classes.

Troubleshooting
If you run out of GPU memory, try reducing the batch size (default: 8).
If you see file not found errors, ensure all paths to CSV and image directories are correct.
For performance issues, consider using fewer workers in the DataLoader by setting num_workers=0.
Customization

Hyperparameters: You can modify learning rates, weight decay, and number of epochs in train.py directly.
Transforms: If you want to add data augmentation or preprocessing steps, customize the transforms in the dataset initialization.
Model Architecture: This code uses fasterrcnn_resnet50_fpn. You can load a different model from torchvision.models.detection if you prefer.
