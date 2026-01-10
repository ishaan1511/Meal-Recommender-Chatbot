# Meal-Recommender-Chatbot

## Introduction

This project explores the integration of large language models with structured nutritional datasets to enable interpretable, constraint-based food recommendation. Natural-language user inputs are parsed into a formal nutrient-constraint schema, which is subsequently used to optimize and rank food choices across both macronutrient and micronutrient dimensions. By combining rule-based heuristics with learned representations, the system demonstrates a practical hybrid approach to personalized nutrition modeling and decision support.

## Getting Started

To run this repository, ensure you have the following environment and libraries installed. The project is designed to run on **Google Colab** or a local Python environment with GPU support.

### Environment
- **Python** ≥ 3.9  
- **GPU (recommended)**: NVIDIA T4 / L4 / A100  

### Core Libraries
- **PyTorch** ≥ 2.0  
- **Transformers** ≥ 4.38  
- **Accelerate**  
- **BitsAndBytes** (for 4-bit quantized inference)  
- **Pandas**  
- **NumPy**  
- **scikit-learn**

### Installation

```bash
pip install torch transformers accelerate bitsandbytes pandas numpy scikit-learn
