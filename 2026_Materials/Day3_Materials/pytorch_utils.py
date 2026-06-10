import random
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# View examples
def image_grid(category, dataset):
    # Get index for category

    # Select images for the given category
    images = [img for img, label in dataset if label == category]
    images = [img.resize((64, 64)) for img in images]

    # Randomly sample the images
    num_images = min(48, len(images))
    sampled_images = random.sample(images, num_images)

    # Create a new image to hold the grid of sampled images
    grid = Image.new("RGB", (64 * 12, 64 * 4))

    # Store the image in the grid
    for i, img in enumerate(sampled_images):
        row = i // 12
        col = i % 12

        grid.paste(img, (col * 64, row * 64))

    # Display
    plt.figure(figsize=(12, 4))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.show()

def draw_class_prediction(model, valset, class_id=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create a new image to hold the grid of images
    grid = Image.new("RGB", (128 * 5, 128 * 4))

    # Lists to hold prediction, probability, actual class, and img
    max_predictions = []
    prediction_prob = []
    actual_class = []
    imgs = []

    # Get model predictions
    for i in range(len(valset)):
        x, y = valset[i]
        # xp = transforms(x)[None, :].to(device)  
        xp = x[None, :].to(device)
        predictions = model(xp).softmax(dim=1)  

        max_val, max_id = predictions.max(dim=1) 

        # If the predicted class equals the class id save the prediction, probability, actual class, and img
        if max_id.item() == class_id:
            max_predictions.append(max_id.item())
            prediction_prob.append(max_val.item())
            actual_class.append(y)
            imgs.append(x)

    # Convert lists to numpy arrays
    max_predictions = np.array(max_predictions)
    prediction_prob = np.array(prediction_prob)

    # Sort indices by highest probability
    sorted_indices = np.argsort(prediction_prob)[::-1]

    # Loop through the top 20 predictions with highest probabaility
    for idx, i in enumerate(sorted_indices[:20]):
        # Get the image
        img = imgs[i].numpy().transpose((1, 2, 0))
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = img.resize((128, 128))  

        # Determine box color -- green if prediction is correct, red if incorrect
        color = "green" if actual_class[i] == class_id else "red"

        # Draw the box around the image
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (img.width - 1, img.height - 1)], outline=color, width=3)

        # Draw text with probability of the class
        text = f"P({class_id}): {prediction_prob[i]:.2f}"
        font = ImageFont.load_default()
        draw.text((5, 5), text, font=font, fill=(0, 0, 0))

        # Place image in grid
        row = idx // 5  
        col = idx % 5
        grid.paste(img, (col * 128, row * 128))

    # Display final grid
    plt.figure(figsize=(12, 8))
    plt.imshow(np.array(grid))
    plt.axis('off')
    plt.show()