from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics(run_dir):
    """Plot Training vs Validation Loss and Validation mAP50"""
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"No results.csv found in {run_dir}")
        return

    df = pd.read_csv(csv_path)
    epochs = df['epoch']

    # Training loss: sum of YOLO components
    train_loss = df['box_loss'] + df['cls_loss'] + df['dfl_loss']

    # Validation loss: sum of components if available
    if 'val_loss' in df.columns:
        val_loss = df['val_loss']
    else:
        val_loss = df['box_loss_val'] + df['cls_loss_val'] + df['dfl_loss_val']

    # Validation mAP50
    mAP50 = df['mAP50']

    # Combined plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs, train_loss, label="Training Loss", color='blue', linestyle='-')
    ax1.plot(epochs, val_loss, label="Validation Loss", color='red', linestyle='--')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True)

    # Secondary y-axis for mAP50
    ax2 = ax1.twinx()
    ax2.plot(epochs, mAP50, label="Validation mAP50", color='green', linestyle='-.', linewidth=2)
    ax2.set_ylabel("mAP50")
    ax2.tick_params(axis='y', labelcolor='green')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title("YOLO Training: Losses and Validation mAP50")
    plot_path = os.path.join(run_dir, "training_dashboard.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Training dashboard saved to {plot_path}")


def main():
    # Load YOLO model
    model = YOLO("./yolo11n.pt")  # pre-trained YOLO model

    # Custom folder to save results
    save_dir = "./DataOutput/Logs"
    os.makedirs(save_dir, exist_ok=True)

    # Train the model
    model.train(
        data="./data.yaml",       # dataset config
        epochs=300,               # number of epochs
        batch=16,                 # batch size
        imgsz=640,                # image size
        device="cuda:0",          # GPU
        project=save_dir,         # folder to save runs
        name="Pwang2-300e",       # subfolder for this run
        exist_ok=True             # overwrite if exists
    )

    # After training, plot metrics
    run_dir = os.path.join(save_dir, "tennis_ball_run")
    plot_metrics(run_dir)


if __name__ == "__main__":
    main()
