# ⛳ Golf Performance Under Uncertainty – Capstone Project

Predicting professional golf performance is inherently difficult due to randomness, player variability, and course-specific effects.

This project investigates whether incorporating **uncertainty and consistency metrics** improves prediction accuracy compared to traditional skill-based models.

---

## 📊 Project Overview

Using PGA Tour data (2015–2022), I built and compared multiple statistical and machine learning models to predict **relative strokes (performance vs par)**.

The project evaluates how different modeling approaches capture:
- Skill (strokes gained)
- Player consistency
- Recent form
- Course difficulty
- Uncertainty in outcomes

---

## 🧠 Models Implemented

### 1. Baseline Model
- **OLS Regression**
- Interpretable benchmark using skill + form + volatility

### 2. Hierarchical Models (KEY CONTRIBUTION)
- **Linear Mixed-Effects Models (LMM)**
  - Player random effects
  - Course random effects
- Captures **hierarchical structure of golf performance**

### 3. Count-Based Models
- **Poisson Regression**
- **Negative Binomial Regression**
- Tests whether score behaves as a count process

### 4. Machine Learning
- **Random Forest**
- Captures non-linear relationships

### 5. Uncertainty Modeling
- **Monte Carlo Simulation**
- Generates full **prediction distributions**
- Enables probability-based insights:
  - Win probability
  - Top-10 probability
  - Risk scenarios

---

## 🏆 Key Findings

- **LMM (player + course effects) is the best model**
- Hierarchical modeling significantly improves prediction accuracy
- Course effects explain the largest share of variability
- ~34% of performance remains **pure randomness**
- Traditional models fail to capture uncertainty → distributions are more informative than point estimates

---

## 📁 Repository Structure
- Preprocessing
- Feature Engineering
- OLS Model
- LMM Models
- Count-Based Models
- Random Forest Model
- Monte Carlo Simulation

## ⚙️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/golf-performance-uncertainty-capstone.git
cd golf-performance-uncertainty-capstone

2. Install dependncies
- pip install -r requirements.txt

3. Run Models

👤 Author
Pablo Di Terlizzi
IE University – BBA & Data Analytics
Capstone Project (2026)
