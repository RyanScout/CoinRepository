# Coin Detection with Faster R-CNN

This repository contains a PyTorch-based script for detecting coins in images using a Faster R-CNN model pretrained on ResNet-50 FPN.

## How to Run the Code

### 1. Install Dependencies

Install all the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Directory Structure

Ensure your project directory is organized as follows:

```plaintext
project_root/
│-- coin_counter_v2.py         # The main training script
│-- requirements.txt           # Dependencies list
│-- data/                      # Dataset directory
│   ├-- train/                 # Training images and annotations
│   │   ├-- image1.jpg
│   │   └-- _annotations.csv
│   └-- test/                  # Testing images and annotations
│       ├-- image2.jpg
│       └-- _annotations.csv
```

### 3. Run the Training Script

Execute the training script:

```bash
python coin_counter_v2.py
```

The trained model weights will be saved as `coin_detector.pth`.

## Installing and Running from a ZIP File

1. **Download the ZIP File**:  
   Download the repository as a ZIP file from the GitHub page by clicking the **"Download ZIP"** button.

2. **Extract the ZIP File**:  
   Unzip the file into your desired directory.

3. **Navigate to the Project Directory**:  
   Open a terminal and navigate to the extracted folder:

   ```bash
   cd path/to/extracted/folder
   ```

4. **Install Dependencies**:  
   Install the required packages using `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Script**:  
   Execute the training script:

   ```bash
   python coin_counter_v2.py
   ```

## Adjusting Learning Parameters

You can modify learning parameters in `coin_counter_v2.py`:

- **Learning Rate**:  
  ```python
  optimizer = optim.Adam(params, lr=1e-4, weight_decay=0.0005)
  ```

- **Number of Epochs**:  
  ```python
  num_epochs = 5
  ```

- **Batch Size**:  
  ```python
  train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
  ```

That's it! Adjust the parameters as needed and run the script to train your model.
