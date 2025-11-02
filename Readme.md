---

# Deepfake Detection — Training & Inference

This repository contains two Kaggle notebooks for training and inference of a **Deepfake Detection** model. The project leverages **PyTorch**, **EfficientNet-B1**, and **Discrete Cosine Transform (DCT)** features to detect fake images.

---


| Notebook                                                                                                  | Description                                                                                                     |
| --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [🔗 Deepfake Detection (Training)](https://www.kaggle.com/code/harshitr9671/deepfake-detection)           | Contains the full training pipeline — dataset loading, feature extraction, model definition, and training loop. |
| [🔗 Deepfake Detection (Inference)](https://www.kaggle.com/code/harshitr9671/deepfake-detection-infrence) | Loads the trained model checkpoint and performs inference on unseen data to generate predictions.               |


## Environment Setup

Before running the notebooks, make sure you have the required libraries installed.
You can install all dependencies using:

```bash
pip install numpy pandas pillow scipy torch torchvision scikit-learn optuna tqdm
```


## Required Libraries

| Library                                                        | Purpose                                                 |
| -------------------------------------------------------------- | ------------------------------------------------------- |
| `os`, `json`, `argparse`                                       | File and argument management                            |
| `numpy`, `pandas`                                              | Data manipulation and analysis                          |
| `PIL (Pillow)`                                                 | Image processing                                        |
| `scipy.fft.dct`                                                | Discrete Cosine Transform feature extraction            |
| `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`         | Deep learning framework (PyTorch)                       |
| `torchvision.transforms`, `torchvision.models.efficientnet_b1` | Data augmentations and pretrained EfficientNet backbone |
| `sklearn.model_selection`, `sklearn.metrics`                   | Dataset splitting and evaluation metrics                |
| `scipy.stats.entropy`                                          | Statistical analysis for feature diversity              |
| `optuna`                                                       | Hyperparameter optimization                             |

---

## 🚀 How to Run the Notebooks

### 1. **Training Notebook**

**Notebook:** [Deepfake Detection (Training)](https://www.kaggle.com/code/harshitr9671/deepfake-detection)

**Steps:**

1. Open the notebook on Kaggle.
2. Attach the dataset:

   * Path used: `/kaggle/input/deepfake-ml-challenge/DATASET`
3. Run all cells sequentially:

   * Data loading & preprocessing
   * Model definition (EfficientNet + DCT)
   * Training loop with validation
   * Model checkpoint saving

**Output:** Trained model weights

---

### 2. **Inference Notebook**

**Notebook:** [Deepfake Detection (Inference)](https://www.kaggle.com/code/harshitr9671/deepfake-detection-infrence)

**Steps:**

1. Open the notebook on Kaggle.
2. Attach:

   * The **test dataset**
   * The **trained model weights** from the training notebook
3. Run all cells sequentially:

   * Load model and weights
   * Perform inference
   * Generate predictions (0 for real, 1 for fake)

**Output:** A Json or  list of predictions.



## References

* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [EfficientNet Architecture](https://arxiv.org/abs/1905.11946)
* [Optuna Hyperparameter Optimization](https://optuna.org/)






