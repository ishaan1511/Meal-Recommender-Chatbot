# Meal-Recommender-Chatbot

## Introduction

This project explores the integration of large language models with structured nutritional datasets to enable interpretable, constraint-based food recommendations. Natural-language user inputs are parsed into a formal nutrient-constraint schema, which is subsequently used to optimize and rank food choices across both macronutrient and micronutrient dimensions. By combining rule-based heuristics with learned representations, the system demonstrates a practical hybrid approach to personalized nutrition modeling and decision support.

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
```

## Repository Structure

```text
.
├── rawdata/                     # Raw FDA nutrition data
├── DataProcessing.ipynb         # Data cleaning & feature engineering
├── chatbot.ipynb                # Chatbot notebook
├── mealbot.py                  
├── structured_food_dataset.csv  # Final cleaned dataset
└── README.md
```

## Running the Chatbot

In terminal:
```bash
!cp "path_to_mealboy.py" mealbot.py
```
In the notebook:
```python
from mealbot import chatbot
```

**Example Queries:**
* `around 500 calories`
* `high protein, not too salty`
* `no dairy`
* `I don’t like chicken`

**How It Works**

1. **Input Parsing:** User input is parsed using NLP to extract specific nutritional intent (e.g., calorie counts or ingredient aversions).
2. **Constraint Mapping:** These extracted preferences are applied as filters against the structured food dataset.
3. **Scoring Engine:** Foods are scored based on how closely they match the user's desired profile (e.g., proximity to target protein levels).
4. **Ranking & Delivery:** The top-ranked meal suggestions are returned to the user via the Gradio chat interface.

## Notes & Limitations

* **Ingredient-level exclusions** are heuristic-based.
* **Composite foods** (e.g., pizza, lasagna) may still appear in some exclusions.
* **The system prioritizes interpretability** over perfect semantic understanding.
  


