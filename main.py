import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model

# Configuration
IMG_WIDTH, IMG_HEIGHT = 224, 224
LAST_CONV_LAYER_NAME = "conv5_block3_out"
CLASSIFIER_LAYER_NAME = "predictions"

# 1. Data Loading and Preprocessing
def load_and_preprocess_image(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Loads and preprocesses an image for ResNet50."""
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded), img_array

def load_or_create_segmentation_mask(img_path, dataset_root, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Loads a segmentation mask if exists, otherwise creates a dummy zero mask."""
    img_name = os.path.basename(img_path)
    mask_name = img_name.rsplit('.', 1)[0] + "_mask.png"
    class_name = os.path.basename(os.path.dirname(img_path))
    mask_dir = os.path.join(dataset_root, "masks", class_name)
    mask_path = os.path.join(mask_dir, mask_name)
    
    if os.path.exists(mask_path):
        mask = keras_image.load_img(mask_path, target_size=target_size, color_mode="grayscale")
        mask_array = keras_image.img_to_array(mask) / 255.0
        return (mask_array > 0.5).astype(np.uint8).squeeze()
    else:
        print(f"Warning: No mask found for {img_name}. Using dummy zero mask.")
        return np.zeros((IMG_WIDTH, IMG_HEIGHT), dtype=np.uint8)

# 2. Classification Model
def get_classification_model():
    """Loads a pre-trained ResNet50 model."""
    return ResNet50(weights="imagenet", include_top=True)

# 3. Grad-CAM Generation
def make_gradcam_heatmap(img_array_preprocessed, model, last_conv_layer_name, classifier_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap."""
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array_preprocessed)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_index.numpy() if hasattr(pred_index, 'numpy') else pred_index

def binarize_heatmap(heatmap, threshold=0.5):
    """Converts a heatmap to a binary mask."""
    return (heatmap > threshold).astype(np.uint8)

# 4. Alignment Metric
def calculate_iou(mask1, mask2):
    """Calculates IoU for two binary masks."""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

# 5. Main Workflow
def process_image_and_validate_focus(image_path, dataset_root, model, threshold=0.5):
    """Processes an image, generates Grad-CAM, and calculates IoU."""
    preprocessed_img, original_img_array = load_and_preprocess_image(image_path)
    true_segmentation_mask = load_or_create_segmentation_mask(image_path, dataset_root)
    
    # Get predictions
    predictions = model.predict(preprocessed_img)
    decoded_preds = decode_predictions(predictions, top=3)[0]
    print(f"Image: {os.path.basename(image_path)}")
    print("Top predictions:")
    for i, (_, label, score) in enumerate(decoded_preds):
        print(f"{i+1}: {label} ({score:.2f})")
    
    # Generate Grad-CAM
    top_pred_index = np.argmax(predictions[0])
    heatmap, _ = make_gradcam_heatmap(
        preprocessed_img, model, LAST_CONV_LAYER_NAME, CLASSIFIER_LAYER_NAME, top_pred_index
    )
    
    # Resize and binarize heatmap
    heatmap_resized = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    binarized_grad_cam_mask = binarize_heatmap(heatmap_resized, threshold)
    
    # Ensure masks are 2D
    if len(true_segmentation_mask.shape) == 3:
        true_segmentation_mask = true_segmentation_mask.squeeze()
    if len(binarized_grad_cam_mask.shape) == 3:
        binarized_grad_cam_mask = binarized_grad_cam_mask.squeeze()
    
    # Calculate IoU
    iou_score = calculate_iou(binarized_grad_cam_mask, true_segmentation_mask)
    print(f"IoU: {iou_score:.4f}\n")
    
    # Visualize
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(original_img_array / 255.0)
    axs[0].set_title(f"Original Image\nPred: {decoded_preds[0][1]}")
    axs[0].axis('off')
    axs[1].imshow(true_segmentation_mask, cmap='gray')
    axs[1].set_title("Ground Truth/Dummy Mask")
    axs[1].axis('off')
    axs[2].imshow(original_img_array / 255.0)
    axs[2].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    axs[2].set_title("Grad-CAM Heatmap")
    axs[2].axis('off')
    axs[3].imshow(binarized_grad_cam_mask, cmap='gray')
    axs[3].set_title(f"Binarized Grad-CAM\nIoU: {iou_score:.4f}")
    axs[3].axis('off')
    plt.savefig(f"results/{os.path.basename(image_path)}_visualization.png")
    plt.close()
    
    return iou_score, decoded_preds

# 6. Batch Processing
def process_dataset(dataset_root, threshold=0.5):
    """Processes all images in the dataset and saves results."""
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join(dataset_root, "masks"), exist_ok=True)  # Create masks dir if needed
    
    model = get_classification_model()
    results = []
    
    for class_name in os.listdir(dataset_root):
        class_dir = os.path.join(dataset_root, class_name)
        if os.path.isdir(class_dir) and class_name.lower() not in ['masks', 'results']:
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    iou_score, preds = process_image_and_validate_focus(img_path, dataset_root, model, threshold)
                    results.append({
                        "image": img_name,
                        "class": class_name,
                        "iou": iou_score,
                        "top_prediction": preds[0][1],
                        "top_score": preds[0][2]
                    })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("results/iou_results.csv", index=False)
    print(f"Average IoU: {df['iou'].mean():.4f}")

# Main Execution
if __name__ == "__main__":
    dataset_root = dataset_root = "datset"  # Update this path
    process_dataset(dataset_root, threshold=0.5)