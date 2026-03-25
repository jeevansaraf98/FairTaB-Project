# Fair Synthetic Data Generation using Bayesian Networks

This repository contains an end-to-end framework for generating and evaluating **fair synthetic tabular data** using **Bayesian Networks (BNs)** and state-of-the-art fairness methods.

The project focuses on **mitigating bias in classification tasks** by modifying dependencies between **protected attributes** (e.g., race, gender) and **target variables**.

---

# Overview

We compare multiple approaches for synthetic data generation and fairness mitigation:

* **BNOmics (Raw Synthetic)** – Data generated from learned BN structure
* **BNOmics (Debiased – Proposed Method)** – Structural bias removal (direct + one-hop dependencies)
* **FLAI** – Causal fairness intervention
* **DECAF** – Deep generative fairness model
* **Original Dataset** – Ground truth baseline

All datasets are evaluated using multiple classifiers and fairness metrics.

---

# Contribution

* Developed a **structure-based debiasing method**:

  * Removes:
    * Direct dependencies
    * One-hop indirect dependencies
* Compared against **state-of-the-art fairness methods (FLAI, DECAF)**
* Built a unified evaluation pipeline across:
  * Multiple datasets
  * Multiple classifiers
  * Multiple fairness metrics

---

# Models Used

The following classifiers are evaluated:

* Decision Tree (DT)
* Naïve Bayes (NB)
* Support Vector Machine (SVM)
* Multi-Layer Perceptron (MLP)

---

# 📊 Fairness Metrics

We evaluate models using:

* Statistical Parity
* Equal Opportunity
* Equalized Odds
* Predictive Parity
* Predictive Equality
* Treatment Equality
* ABROCA (Area Between ROC Curves)

---

# Notes

* BNOmics-based synthetic data preserves **structure**, not exact data distribution.
* Debiasing is performed via **hard structural intervention** (edge removal).
* FLAI and DECAF implementations are **adapted versions** of original repositories.

---

# 📈 Research Context

This work focuses on:

* Fairness-aware machine learning
* Causal bias mitigation
* Fair synthetic data generation
* Bayesian Network modeling

---

