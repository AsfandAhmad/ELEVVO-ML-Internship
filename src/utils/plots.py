import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os



def plot_history(history, out_path):
    h = history.history
    plt.figure()
    plt.plot(h.get('accuracy', []), label='train_acc')
    plt.plot(h.get('val_accuracy', []), label='val_acc')
    plt.plot(h.get('loss', []), label='train_loss')
    plt.plot(h.get('val_loss', []), label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
