import os
import random
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import warnings
import pillow_heif  # Import pillow-heif to enable HEIC support

warnings.filterwarnings('ignore', category=FutureWarning)

# Register HEIF format with Pillow
pillow_heif.register_heif_opener()

def get_model_instance(num_classes):
    # Initialize the model architecture (without pre-trained weights)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    
    # Replace the classifier with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def resize_with_padding(image, target_size):
    # Resize image while maintaining aspect ratio and pad to target_size
    ratio = min(target_size[0] / image.width, target_size[1] / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    image = image.resize(new_size, resample=Image.BILINEAR)
    
    # Create a new image and paste the resized on it
    new_image = Image.new("RGB", target_size)
    paste_position = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    new_image.paste(image, paste_position)
    return new_image

def open_image(path):
    # Open image using PIL
    try:
        image = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {path}: {e}")
        return None
    return image

def visualize_predictions(model, images, device, class_names, coin_values, annotations_df=None, threshold=0.5, output_dir=None):
    model.eval()
    target_size = (640, 640)  # Target size for resizing
    
    for img_path in images:
        img_name = os.path.basename(img_path)
        img = open_image(img_path)
        if img is None:
            continue  # Skip if the image could not be opened
        img_resized = resize_with_padding(img, target_size)
        img_tensor = T.ToTensor()(img_resized).to(device)
        
        with torch.no_grad():
            prediction = model([img_tensor])
        
        img_np = img_resized.copy()
        plt.figure(figsize=(12, 8))
        plt.imshow(img_np)
        ax = plt.gca()
        
        # Predictions in red
        predicted_total = 0.0
        for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
            if score > threshold:
                xmin, ymin, xmax, ymax = box
                xmin = xmin.item()
                ymin = ymin.item()
                xmax = xmax.item()
                ymax = ymax.item()
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
                # Adjust label index (subtract 1, as 0 is background)
                if label >= 1 and label <= len(class_names):
                    label_name = class_names[label - 1]
                else:
                    label_name = 'unknown'
                coin_value = coin_values.get(label_name, 0.0)
                predicted_total += coin_value
                ax.text(
                    xmin, ymin,
                    f"{label_name}: {score:.2f}",
                    bbox=dict(facecolor='yellow', alpha=0.5)
                )
        
        # Display predicted total amount
        ax.text(
            10, 20,
            f"Predicted Total: ${predicted_total:.2f}",
            bbox=dict(facecolor='yellow', alpha=0.5),
            fontsize=14,
            color='black'
        )
        
        # Ground truth in green (if annotations are available)
        actual_total = None
        if annotations_df is not None:
            actual_total = 0.0
            gt_annots = annotations_df[annotations_df['filename'] == img_name]
            for _, row in gt_annots.iterrows():
                # Adjust bounding boxes according to resizing
                xmin = row['xmin']
                ymin = row['ymin']
                xmax = row['xmax']
                ymax = row['ymax']
                # Apply same resizing and padding to bounding boxes
                ratio = min(target_size[0] / img.width, target_size[1] / img.height)
                xmin = xmin * ratio + (target_size[0] - img.width * ratio) / 2
                xmax = xmax * ratio + (target_size[0] - img.width * ratio) / 2
                ymin = ymin * ratio + (target_size[1] - img.height * ratio) / 2
                ymax = ymax * ratio + (target_size[1] - img.height * ratio) / 2
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='green', linewidth=2)
                ax.add_patch(rect)
                label_name = row['class']
                coin_value = coin_values.get(label_name, 0.0)
                actual_total += coin_value
                ax.text(
                    xmin, ymax,
                    f"{label_name}",
                    bbox=dict(facecolor='green', alpha=0.5)
                )
            # Display actual total amount
            ax.text(
                10, 50,
                f"Actual Total: ${actual_total:.2f}",
                bbox=dict(facecolor='green', alpha=0.5),
                fontsize=14,
                color='black'
            )
        
        plt.axis('off')
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, img_name)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

def main():
    # Hardcoded paths
    image_dir = 'CoinRepository/data/train'  # Replace with your actual path
    annotations_file = 'CoinRepository/data/train/_annotations.csv'  # Replace with your actual path
    model_path = 'v2_coin_detector.pth'  # Replace with your actual model path
    num_images = 5  # Number of random images to process
    threshold = 0.5  # Detection threshold
    output_dir = None  # Set to a directory path if you want to save the images

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define class names and coin values directly
    class_names = ['Penny', 'Nickel', 'Dime', 'Quarter']  # Adjust based on your training classes
    num_classes = len(class_names) + 1  # +1 for background

    # Define coin values for each class name
    coin_values = {
        'Penny': 0.01,
        'Nickel': 0.05,
        'Dime': 0.10,
        'Quarter': 0.25
    }

    # Initialize the model and load the state_dict
    model = get_model_instance(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Check if annotations file exists
    if os.path.exists(annotations_file):
        # Load annotations dataframe
        annotations_df = pd.read_csv(annotations_file)
    else:
        annotations_df = None
        print("Annotations file not found. Proceeding without ground truth annotations.")
    
    # Collect images from the directory, including HEIC files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.heif']
    all_images = [
        os.path.join(image_dir, filename)
        for filename in os.listdir(image_dir)
        if os.path.splitext(filename)[1].lower() in valid_extensions
    ]
    
    if not all_images:
        print("No images found in the specified directory.")
        return
    
    # Randomly select images
    images = random.sample(all_images, min(num_images, len(all_images)))
    
    # Visualize predictions
    visualize_predictions(model, images, device, class_names, coin_values, annotations_df, threshold=threshold, output_dir=output_dir)

if __name__ == '__main__':
    main()
