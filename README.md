# CNN Image Classification - Intel Dataset (Exercise 4)

This project implements a convolutional neural network (CNN) classifier for the Intel Image Classification dataset. The work is contained in `Ex04_CNN_Image_Classification.ipynb` and covers data preparation, model design, training, evaluation, and visualization.

## Project Highlights
- Dataset: Intel Image Classification (6 classes: buildings, forest, glacier, mountain, sea, street)
- Input size: images resized to 64x64
- Model: 3 convolutional layers + 2 fully connected layers with dropout
- Training: Adam optimizer, CrossEntropyLoss, 8 epochs
- Reported test accuracy: 82.77%

## Repository Structure
- `Ex04_CNN_Image_Classification.ipynb`: main notebook
- `seg_train/`: training data (ImageFolder format)
- `seg_test/`: validation/test data (ImageFolder format)
- `seg_pred/`: prediction data (optional)

Expected dataset layout:
```
seg_train/
  buildings/
  forest/
  glacier/
  mountain/
  sea/
  street/
seg_test/
  buildings/
  forest/
  glacier/
  mountain/
  sea/
  street/
```

## How It Works
1. **Data preparation**
   - Compute dataset mean/std for normalization.
   - Apply augmentation (random horizontal flip, color jitter) on training data.
2. **Model architecture**
   - Conv(3->32) + MaxPool
   - Conv(32->64) + MaxPool
   - Conv(64->64) + MaxPool
   - FC(64*8*8 -> 128) + Dropout
   - FC(128 -> 6)
3. **Training & evaluation**
   - Track training/validation loss.
   - Compute test accuracy and confusion matrix.
4. **Visualizations**
   - Loss curves
   - Confusion matrix
   - Feature maps and filters

## Requirements
Install the main dependencies:
```
pip install torch torchvision matplotlib seaborn scikit-learn
```
Note: GPU is optional but speeds up training.

## Running the Notebook
Open and run the notebook in order:
```
Ex04_CNN_Image_Classification.ipynb
```
Make sure the dataset folders (`seg_train`, `seg_test`) are in the same directory as the notebook, or update the paths in the Data Preparation section.

## Results (from notebook)
- Final test accuracy: 82.77%
- Trainable parameters: 581,510
- Best validation loss: 0.5317 (epoch 7)

## Notes
- If you change image size or augmentations, recompute mean/std.
- Reproducibility can be improved by setting random seeds.

## References
- Intel Image Classification dataset
- PyTorch and torchvision
# Simple-CNN-Example
