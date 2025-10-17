# Unsupervised Structural Probe for Finding Syntax in Word Representations

This project implements and evaluates structural probes, particularly unsupervised methods, for discovering syntactic structures within word embeddings. The work is inspired by and builds upon the findings of Hewitt & Manning in "A Structural Probe for Finding Syntax in Word Representations."

-----

## Key Experiments and Features

This repository contains two main experiments that explore different aspects of structural probing and tree decoding.

### Experiment 1: Tree Decoding with MST and ILP

This experiment compares two primary methods for decoding tree structures from distance matrices:

  * **Minimum Spanning Tree (MST):** A fast, greedy algorithm that is guaranteed to produce a valid tree.
  * **Integer Linear Programming (ILP):** An optimization-based approach that aims to find the optimal tree structure.

The experiment evaluates these methods on their ability to reconstruct single and mixed tree structures, measuring both edge accuracy and distance distortion (Frobenius Norm).

**Results from Experiment 1:**

### Experiment 2: Unsupervised Structural Probe Training

This experiment implements and trains an unsupervised structural probe to learn a projection matrix that reveals syntactic distances in BERT embeddings. Two novel unsupervised training methods are introduced and evaluated:

  * **MST Proper:** Uses the Minimum Spanning Tree of the predicted distances to create a full target distance matrix, providing a more stable training signal.
  * **Distance Weighted Walk:** Constructs a tree using a random walk where the probability of traversing an edge is inversely proportional to the distance predicted by the probe.

These methods are compared against an **untrained baseline** and a **supervised upper bound** to gauge their effectiveness. The performance is measured by the Undirected Unlabeled Attachment Score (UUAS).

**Results from Experiment 2:**

The "MST Proper" method shows a significant improvement over the baseline, achieving a UUAS of **0.600**, which is a substantial step towards the supervised upper bound of **0.872**. The "Distance Weighted Walk" method, however, did not perform as well, with a UUAS of **0.224**.

**Training and Development Performance:**

The training curves below illustrate the learning progress and development set performance for both the MST Proper and Distance Weighted Walk methods over 50 epochs.

-----

## How to Run the Experiments

The experiments are contained within Jupyter notebooks in the `notebook/` directory.

1.  **Experiment 1:** To replicate the tree decoding comparison, run the `exp1.py` notebook.
2.  **Experiment 2:** To train and evaluate the unsupervised structural probes, run the `exp2.py` notebook.

-----

## Requirements

To run the code in this repository, you will need to install the following dependencies:

```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
torch>=1.10.0
nltk>=3.6
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
networkx>=2.6.0
transformers>=4.0.0
```

You can install these packages using pip:

```bash
pip install -r requirements.txt
```