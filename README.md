# CSA4702 - DEEP LEARNING LABORATORY

**Department of Computer Science and Engineering**  
**SIMATS Engineering, Saveetha Institute of Medical and Technical Sciences**  
**Course Code:** CSA4702 / CSA47 – Deep Learning  
**Course Faculty:** Dr. T. Vincent Gnanaraj  

---

## 📋 Course Outcomes (CO) Mapping

- **CO1:** Formulate mathematical foundations, vector/matrix algebra, probability distributions, information theory metrics, and optimization routines for deep learning networks.
- **CO2:** Design, implement, and regularize multilayer perceptrons and convolutional neural network architectures for visual recognition tasks.
- **CO3:** Construct recurrent architectures (RNN, LSTM, GRU), analyze backpropagation through time (BPTT), and optimize sequential dependencies.
- **CO4:** Apply dimensionality reduction techniques (PCA, PPCA, FA, ICA) and probabilistic graphical models (Bayesian Networks) for latent factor discovery.
- **CO5:** Synthesize energy-based generative models (RBM, DBN, DBM) and deep generative networks (VAEs, GANs) for representation learning and sample synthesis.

---

## 🧪 Index of Experiments

| S.No | CO | Experiment Title | Packages / Technologies | Status |
| :---: | :---: | :--- | :--- | :---: |
| 1 | CO1 | Vector and Matrix Operations for Deep Learning | NumPy, Matplotlib | Completed |
| 2 | CO1 | Covariance, Eigenvalues and Eigenvectors Analysis | NumPy, Pandas, Matplotlib | Completed |
| 3 | CO1 | Probability Distributions and Bayes' Theorem | NumPy, SciPy, Matplotlib | Completed |
| 4 | CO1 | Entropy, Cross-Entropy and KL Divergence | NumPy, SciPy, Matplotlib | Completed |
| 5 | CO1 | Bias–Variance, Overfitting and Underfitting Analysis | Scikit-learn, NumPy, Matplotlib | Completed |
| 6 | CO1 | Gradient Descent, SGD, Momentum and Adam Comparison | NumPy, TensorFlow, Keras, Matplotlib | Completed |
| 7 | CO2 | Perceptron and Forward Propagation | NumPy, Matplotlib | Completed |
| 8 | CO2 | Two-Layer Neural Network with Backpropagation | NumPy, Matplotlib | Completed |
| 9 | CO2 | Multilayer Perceptron Implementation | TensorFlow, Keras, NumPy, Matplotlib | Completed |
| 10 | CO2 | L2, Dropout and Early Stopping Comparison | TensorFlow, Keras, NumPy, Matplotlib | Completed |
| 11 | CO2 | CNN for Image Classification | TensorFlow, Keras, NumPy, Matplotlib | Completed |
| 12 | CO2 | CNN Architecture and Hyperparameter Comparison | TensorFlow, Keras, NumPy, Matplotlib | Completed |
| 13 | CO3 | Sequential Dataset Preparation and RNN Prediction | TensorFlow, Keras, NumPy, Pandas, Matplotlib | Completed |
| 14 | CO3 | Unfolded RNN Architecture Visualization | TensorFlow, Keras, NetworkX, Matplotlib | Completed |
| 15 | CO3 | BPTT Flow and Gradient Analysis | TensorFlow, Keras, NumPy, Matplotlib | Completed |
| 16 | CO3 | LSTM and RNN Performance Comparison | TensorFlow, Keras, NumPy, Pandas, Matplotlib | Completed |
| 17 | CO3 | RNN Gradient Analysis and Gradient Clipping | TensorFlow, Keras, NumPy, Matplotlib | Completed |
| 18 | CO3 | RNN, LSTM and GRU Hyperparameter Experimentation | TensorFlow, Keras, NumPy, KerasTuner, Matplotlib | Completed |
| 19 | CO4 | PCA and Explained Variance Analysis | NumPy, Scikit-learn, Matplotlib | Completed |
| 20 | CO4 | Probabilistic PCA and Reconstruction Analysis | NumPy, Scikit-learn, Matplotlib | Completed |
| 21 | CO4 | Factor Analysis and Latent Factor Interpretation | Scikit-learn, NumPy, Pandas, Matplotlib | Completed |
| 22 | CO4 | Independent Component Analysis and Component Visualization | Scikit-learn, NumPy, Matplotlib | Completed |
| 23 | CO4 | Bayesian Network Construction and Visualization | pgmpy, NetworkX, Matplotlib | Completed |
| 24 | CO4 | Graphical Model Sampling and Probabilistic Inference | pgmpy, NumPy, Pandas, NetworkX, Matplotlib | Completed |
| 25 | CO5 | Restricted Boltzmann Machine Energy and Reconstruction | NumPy, SciPy, Matplotlib | Completed |
| 26 | CO5 | Gibbs Sampling and Contrastive Divergence Analysis | NumPy, SciPy, Matplotlib | Completed |
| 27 | CO5 | Deep Belief Network Representation Learning | NumPy, SciPy, TensorFlow, Keras, Matplotlib | Completed |
| 28 | CO5 | Deep Boltzmann Machine / Convolutional Boltzmann Machine Simulation | NumPy, SciPy, TensorFlow, Keras, Matplotlib | Completed |
| 29 | CO5 | VAE/GAN Sample Generation | TensorFlow, Keras, NumPy, Matplotlib | Completed |
| 30 | CO5 | Generative Model Output Comparison and Evaluation | TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib | Completed |
| Bonus | CO5 | Samurai vs. Ninja Tactical Simulation via Conditional GAN (cGAN) | PyTorch, NumPy, Pandas, Matplotlib | Completed |

---

## 📂 Repository Structure

- **`Experiments/Exp. <n>/` (Experiments 1 to 30)**:
  - `p<n>.py`: Comprehensive, modular Python implementation script.
  - `plot1_<name>.png`, `plot2_<name>.png`: Publication-grade 300 DPI visualizations.
  - `table 1.png` - `table <k>.png`: Exact terminal output tables rendered with high legibility.

- **`samurai/` (Special Capstone Experiment)**:
  - `samurai_ninja_gan.py`: Conditional Generative Adversarial Network (cGAN) for synthetic battle profile generation.
  - `samurai_ninja_trap_escape_1000.csv`: Conditioned tactical dataset with 1,000 synthesized encounters (`escape` vs. `trap`).
  - `plot1_cgan_training_dynamics.png`, `plot2_tactical_escape_distribution.png`: 300 DPI adversarial loss and tactical radar profiles.
  - `table 1.png` - `table 5.png`: High-legibility terminal output tables for combat influence, architecture, and tactical outcomes.
