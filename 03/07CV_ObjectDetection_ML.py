import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# COCO数据集的类别名
COCO_INSTANCE_CATEGORY_NAMES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def load_model():
    # Load the pre-trained Faster R-CNN model with the new API
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()  # Set the model to evaluation mode
    return model

def detect_objects(model, img_path):
    # Load and convert the image to RGB to ensure 3 color channels
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts the image to type torch.Tensor and scales to [0, 1]
    ])
    img_tensor = transform(img)

    # Ensure the tensor is in the correct shape [C, H, W] with C=3 for RGB images
    if img_tensor.shape[0] != 3:
        raise ValueError("Image does not have 3 channels")

    # Perform object detection
    with torch.no_grad():  # No need to compute gradients (for inference)
        prediction = model([img_tensor])
    return prediction

def plot_image(image_path, predictions, save_path=None):
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    boxes = predictions[0]['boxes'].numpy()
    labels = predictions[0]['labels'].numpy()
    scores = predictions[0]['scores'].numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # only plot if the score is greater than 0.5
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label-1]
            ax.text(box[0], box[1], f'{label}: {label_name} {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, format='png')  # Save the figure to a file
        plt.close(fig)  # Close the figure window to free up resources
    else:
        plt.show()  # Display the figure as usual


def main():
    model = load_model()
    img_path = '0701.png'  # Replace with the path to your image file
    predictions = detect_objects(model, img_path)
    save_path = 'output_image.png'  # Specify the path where you want to save the image
    plot_image(img_path, predictions, save_path)

    # Print the predictions
    print(predictions)

if __name__ == '__main__':
    main()
