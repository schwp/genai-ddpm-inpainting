import matplotlib.pyplot as plt
import pickle

def visualize_fashion_mnist_classes(tmp_loader, label_to_name_map):
    # Get one sample image for each label from the loader
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

    # Display images in a 2x5 grid
    fig = plt.figure(figsize=(15, 6))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1)
        img = found_images[i]
        
        ax.set_title(label_to_name_map[i])
        ax.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    plt.tight_layout()
    plt.show()


def plot_loss_evolution(pkl_file_path, save_file = False):
    # Load training and test loss history from pickle file
    try:
        with open(pkl_file_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File '{pkl_file_path}' not found.")
        return

    train_loss = data.get("train_epoch_loss", [])
    test_loss = data.get("test_epoch_loss", [])
    
    plt.figure(figsize=(10, 6))
    
    # Plot both loss curves
    plt.plot(train_loss, label='Train Loss', color='blue', linestyle='-')
    
    if test_loss:
        plt.plot(test_loss, label='Test Loss', color='orange', linestyle='--')
    
    plt.title('Loss Evolution per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_file:
        plt.savefig("loss_evolution.png") 
    plt.show()