import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision.transforms.functional as F

# Define collate_fn at the top level
def collate_fn(batch):
    return tuple(zip(*batch))

class CoinDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transforms=None):
        """
        Args:
            annotations_file (str): Path to the CSV file with annotations.
            img_dir (str): Directory with all the images.
            transforms (callable, optional): Optional transforms to be applied on a sample.
            target_size (tuple, optional): Desired output size of the images (width, height).
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transforms = transforms
        
        # Define valid image extensions
        self.valid_extensions = ['.jpg', '.jpeg', '.png']
        
        # Filter out annotations for non-image files
        self.img_labels = self.img_labels[self.img_labels['filename'].apply(
            lambda x: os.path.splitext(x)[1].lower() in self.valid_extensions
        )]
        
        self.image_names = self.img_labels['filename'].unique()
        self.class_names = sorted(self.img_labels['class'].unique())
        self.class_to_idx = {cls_name: idx + 1 for idx, cls_name in enumerate(self.class_names)}  # 0 is background

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(str(self.img_dir), str(img_name))
        img = Image.open(img_path).convert("RGB")

        annots = self.img_labels[self.img_labels['filename'] == img_name]

        boxes = []
        labels = []
        for _, row in annots.iterrows():
            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            labels.append(self.class_to_idx[row['class']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img = self.transforms(img)

        return img, target


def get_model(num_classes):
    # Load pre-trained Faster R-CNN model with default weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Replace the pre-trained head with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def visualize_predictions(model, subset, device, num_images=5):
    model.eval()
    for idx in range(num_images):
        img, target = subset[idx]
        img = img.to(device)
        with torch.no_grad():
            prediction = model([img])

        img = img.cpu().permute(1, 2, 0).numpy()
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        ax = plt.gca()

        # Ground truth in green
        for box in target["boxes"]:
            xmin, ymin, xmax, ymax = box
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='green', linewidth=2)
            ax.add_patch(rect)

        # Predictions in red
        for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
            if score > 0.5:
                xmin, ymin, xmax, ymax = box
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
                ax.text(
                    xmin, ymin,
                    f"{subset.dataset.class_names[label -1]}: {score:.2f}",
                    bbox=dict(facecolor='yellow', alpha=0.5)
                )

        plt.axis('off')
        plt.show()

def set_seed(seed=42):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train():
    # Set random seed for reproducibility
    set_seed(42)

    # Directories
    # Dataset for training
    train_annotations_file = 'data/train/_annotations.csv'
    train_img_dir = 'data/train'
    # Data set for testing
    test_annotations_file = 'data/test/_annotations.csv'
    test_img_dir = 'data/test'

    transforms = T.Compose([
        T.ToTensor(),
    ])

    # Create datasets
    train_dataset = CoinDataset(train_annotations_file, train_img_dir, transforms=transforms)
    test_dataset = CoinDataset(test_annotations_file, test_img_dir, transforms=transforms)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = len(train_dataset.class_names) + 1  # +1 for background
    model = get_model(num_classes)
    model.to(device)

    # Parameters to control dataset sizes
    # Ensure that the sizes do not exceed the dataset
    train_fraction = 1
    total_size = len(train_dataset)
    train_size = int(train_fraction * total_size)
    train_size = min(train_size, total_size)

    test_fraction = 1
    total_size = len(test_dataset)
    test_size = int(test_fraction * total_size)
    test_size = min(test_size, total_size)

    # Create random indices for training for speed and efficiency
    # Create subsets of the data for randomization
    rand_train_indices = torch.randperm(len(train_dataset)).tolist()
    train_indices = rand_train_indices[:train_size]
    rand_test_indices = torch.randperm(len(test_dataset)).tolist()
    test_indices = rand_test_indices[:test_size]

    # Redefine datasets
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    # Initialize DataLoader with collate_fn and num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,  # Set to 0 for Jupyter compatibility
        collate_fn=collate_fn
    )


    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-4, weight_decay=0.0005)

    # Number of Epochs
    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
        for i, (images, targets) in progress_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            progress_bar.set_postfix({'Loss': losses.item()})

        # Update Learning Rate
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    print("Training complete.")

    # Save the model
    torch.save(model.state_dict(), 'coin_detector.pth')

    # Visualize some testing predictions
    visualize_predictions(model, test_dataset, device, num_images=5)

if __name__ == "__main__":
    train()
