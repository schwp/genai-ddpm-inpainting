import matplotlib.pyplot as plt

def visualize_fashion_mnist_classes(tmp_loader, label_to_name_map):
    found_images = {}

    for images, labels in tmp_loader:
        for i in range(len(labels)):
            label_id = labels[i].item()
            if label_id not in found_images:
                found_images[label_id] = images[i]
            
            if len(found_images) == 10:
                break
        if len(found_images) == 10:
            break

    fig = plt.figure(figsize=(15, 6))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1)
        img = found_images[i]
        
        ax.set_title(label_to_name_map[i])
        ax.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    plt.tight_layout()
    plt.show()