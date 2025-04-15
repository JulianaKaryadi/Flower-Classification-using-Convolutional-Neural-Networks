# Flower-Classification-using-Convolutional-Neural-Networks
Flower Classification with Deep CNNs: A comparative study of ResNet50, DenseNet121, and MobileNetV3Small for classifying dahlia, daisy, rose, lily, and sunflower images using transfer learning in TensorFlow. Features custom data pipeline, performance analysis, and visualization tools for model evaluation.

## Data Collection and Preparation

### Image Scraper (`images_scrapper.ipyb`)

This script automates the process of collecting flower images from free online repositories. Key features include:

- **Web Scraping**: Utilizes BeautifulSoup to extract image URLs from FreeImages.com
- **Robust Error Handling**: Implements retry mechanism with exponential backoff for handling network issues
- **Rate Limiting**: Incorporates delays to respect website terms of service
- **Customizable Parameters**: Configurable target count (2000 images per flower type) and search queries

The script systematically collects high-quality images of different flower types (dahlias, daisies, roses, lilies, and sunflowers) to build a comprehensive dataset for training our classification models. A total of 10,000 images were collected (2,000 per flower category).

### Dataset Organization (`flowers_dataset.ipynb`)

This script structures the collected images into a proper machine learning dataset:

- **Train/Validation/Test Split**: Implements a 70/15/15 random split to ensure unbiased evaluation
- **Directory Structure**: Creates organized folder hierarchy for TensorFlow/Keras compatibility
- **Class Balance**: Ensures equal representation of each flower class (2,000 images per class)

The final dataset distribution:
- Training set: 7,000 images (1,400 per class)
- Validation set: 1,500 images (300 per class)
- Test set: 1,500 images (300 per class)

## Performance Visualization and Analysis

The project includes comprehensive visualization tools to evaluate model performance:

- **Training Metrics**: Plots accuracy and loss curves for both training and validation sets
- **Confusion Matrices**: Visualizes classification performance across flower categories
- **Performance Metrics**: Calculates and compares training time, test accuracy, and mean Average Precision (mAP)
- **Model Comparison**: Implements a weighted performance heuristic that combines accuracy and mAP to determine the optimal model

The visualization code provides key insights into model behavior, highlighting strengths and weaknesses of each architecture when classifying the different flower types.

## Dataset Access

The complete flower classification dataset (10,000 images across 5 categories) is available via Google Drive:
[FlowersDataset (10,000 images)](https://drive.google.com/drive/folders/1I3pplF-375N1Xiq7afCCE0PyVwi2sGIN?usp=drive_link)
