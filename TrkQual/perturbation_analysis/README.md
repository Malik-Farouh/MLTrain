This repository documents an exhaustive machine learning investigation focused on engineering an independent **XGBoost gradient-boosted decision tree (BDT)** classifier. This model serves as a high-precision, production-grade cross-verification framework running alongside the existing Artificial Neural Network (ANN) model currently deployed in the Mu2e experiment's offline track reconstruction pipeline.

The core objective of this framework is to classify track quality using precise measurements, isolate true signal conversion electrons ($CE$), and reject all the background signals.

##  Repository Architecture & Modular Walkthrough
The codebase has been split into 4 dedicated, independent research chapters to prevent memory leaks and variable collisions:

*   **`00_TrkQualTrain_main.ipynb`**: The master training pipeline. Handles raw feature loading from ROOT files, balances sample weights, tunes tree hyperparameters, trains the final 500-tree BDT ensemble, and save the model as .joblib for offline use.
*   **`01_data_perturbation.ipynb`**: The file that is responsible for creating and perturbing all of the required data sets.
*   **`02_factive_nactive_analysis.ipynb`**: Investigates the physical properties of track hits. Evaluates feature correlations and isolates the impact of `factive`  and `nactive` on the model's background rejection capabilities.
*   **`03_momerr_robustness_analysis.ipynb`**: A targeted stress-test notebook. Subjects the model to severe $\pm10\%$ and extreme $\pm50\%$ momentum error systematic variations to evaluate algorithmic stability under tracking resolution degradation.
*   **`04_all_data_perturbation_analysis.ipynb`**: The global analysis file. Compiles all perturbed data streams into a unified evaluation framework to compare static cut strategies against adaptive dynamic domain adaptation.
