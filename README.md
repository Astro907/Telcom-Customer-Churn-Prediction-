# 📊 Telecom Customer Churn Prediction

---
## 🎯 Business Problem

A telecom company is experiencing a **26.5% annual customer churn rate**. With 7,043 customers and an average customer lifetime value of $2,000, this translates to approximately **$3.7M in lost revenue annually**.

**Business Questions:**
- What drives customers to leave?
- Can we predict which customers are at risk?
- What retention strategies would have the highest ROI?

---

## 📂 Dataset Overview

- **Source:** IBM sample Telecom Customer Churn Dataset
- **Size:** 7,043 customers
- **Features:** 20 variables
  
  - Demographics (gender, senior citizen, partner, dependents)
  - Account info (tenure, contract type, payment method, charges)
  - Services (phone, internet, online security, tech support, streaming)
- **Target Variable:** Churn (Yes/No)
- **Class Distribution:** 73.5% retained, 26.5% churned

---

## 🔍 Key Findings from Exploratory Data Analysis

### 1️⃣ Contract Type is the #1 Churn Driver (77.1% variance)

| Contract Type | Churn Rate |
|---------------|------------|
| Month-to-month | 42.0% |
| One year | 11.0% |
| Two year | 3.0% |

**Insight:** Customers without long-term commitment are 14x more likely to churn.

---

### 2️⃣ New Customers Have Highest Risk

| Tenure | Churn Rate |
|--------|------------|
| 0-12 months | 50%+ |
| 12-24 months | 35% |
| 24-48 months | 20% |
| 48+ months | 15% |

**Insight:** First-year retention is critical.

---

### 3️⃣ Value-Added Services Significantly Reduce Churn

| Service | With Service | Without Service | Difference |
|---------|--------------|-----------------|------------|
| Tech Support | 15% churn | 41% churn | -26% |
| Online Security | 15% churn | 42% churn | -27% |
| Device Protection | 23% churn | 39% churn | -16% |

**Insight:** Customers with premium services feel more invested in staying.

---

### 4️⃣ High-Risk Customer Profile

**Characteristics of customers with 85%+ churn probability:**
- Month-to-month contract
- Fiber optic internet service
- No tech support
- No online security
- Electronic check payment method
- Paperless billing
- Tenure ≤ 12 months

**Population:** 89 customers matching this exact profile

---

## 🤖 Machine Learning Models

### Approach: Testing 4 Different Models

The dataset had a class imbalance problem (73% non-churners, 27% churners), so I tested multiple approaches:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline Logistic Regression | 78.8% | 65% | 49% | 56% |
| Random Forest | 79.3% | 63% | 54% | 58% |
| **Balanced Logistic Regression** ✅ | **72.1%** | **47%** | **55%** | **51%** |
| Random Forest + Balanced Data | 73.5% | 48% | 54% | 51% |

---

### 🏆 Winner: Balanced Logistic Regression

**Why I chose this model:**

1. **Highest Recall (55%)** - Catches the most churners
   - In churn prediction, missing a churner is more costly than a false alarm
   - 55% recall means we identify 55% of at-risk customers before they leave

2. **Business Interpretability**
   - Stakeholders can understand coefficient weights
   - Easy to explain "why" a customer is flagged
   - Builds trust with business teams

3. **Handles Class Imbalance**
   - Used upsampling to balance training data
   - Prevented model from just predicting "no churn" for everyone

4. **Trade-offs I Accepted**
   - 7% lower accuracy than Random Forest (72% vs 79%)
   - This was acceptable because recall improved by 1%
   - Goal: Find churners, not just be accurate

**Model Performance Metrics:**
- **True Positives:** 193 (correctly identified churners)
- **False Negatives:** 159 (missed churners)
- **False Positives:** 217 (false alarms)
- **True Negatives:** 840 (correctly identified non-churners)

---

## 📊 Feature Importance Analysis

### EDA-Based Feature Impact (Churn Rate Variance)

| Rank | Feature | Variance |
|------|---------|----------|
| 1 | Contract | 77.1% |
| 2 | InternetService | 41.2% |
| 3 | TechSupport | 30.8% |
| 4 | OnlineSecurity | 29.6% |
| 5 | PaymentMethod | 25.3% |

### ML Model Coefficients (Balanced Logistic Regression)

| Rank | Feature | Coefficient | Impact |
|------|---------|-------------|--------|
| 1 | Contract | -0.819 | Longer contracts = Less churn |
| 2 | PhoneService | -0.706 | Having phone service = Less churn |
| 3 | SeniorCitizen | -0.418 | Complex relationship |
| 4 | TechSupport | -0.321 | Tech support = Less churn |
| 5 | OnlineSecurity | -0.296 | Online security = Less churn |

**Key Insight:** Both EDA and ML agree on top features (Contract, Tech Support, Online Security), which validates the analysis.

---

## 💼 Business Recommendations

### Strategy 1: Contract Incentive Program 📄

**Problem:** Month-to-month customers churn at 42% vs 3% for two-year contracts

**Solution:**
- Offer 15-20% discount for annual commitments
- Provide free service upgrades for two-year contracts
- Create "contract migration" campaigns targeting high-risk month-to-month customers

**Implementation:**
- Target: Convert 30% of month-to-month base (~750 customers)
- Incentive cost: ~$100 discount per conversion
- Total program cost: $75K

**Expected ROI:**
```
Current situation:
- 2,500 month-to-month customers
- 42% churn rate = 1,050 churners
- Lost revenue: 1,050 × $2,000 = $2.1M

After intervention:
- 750 customers convert to annual (11% churn rate)
- New churners from converted: 750 × 11% = 83
- Churners saved: 750 × (42% - 11%) = 232
- Revenue saved: 232 × $2,000 = $464K

Net benefit: $464K - $75K = $389K annually
```

---

### Strategy 2: Early Intervention Program 🎯

**Problem:** First-year customers churn at 50%+ rate

**Solution:**
- Implement 90-day onboarding program
- Free tech support for first 6 months  
- Loyalty rewards at 6-month milestone

**Implementation:**
- Target all new sign-ups (~3,000 annually)
- Program cost: ~$150K

**Expected ROI:**
Current situation:
- 3,000 new customers × 50% churn = 1,500 churners
- Lost revenue: 1,500 × $2,000 = $3M

After intervention:
- Target: 25% churn reduction
- New churn rate: 50% reduced by 25% = 37.5%  
- New churners: 3,000 × 37.5% = 1,125
- Customers saved: 375
- Revenue saved: 375 × $2,000 = $750K

Net benefit: $750K - $150K = $600K annually
```

---

### Strategy 3: Bundled Support Services 🛠️

**Problem:** Customers without tech support churn at 41% vs 15% with support

**Solution:**
- Bundle tech support with all fiber optic plans
- Offer 3-month free trial of online security for high-risk segments
- Create tiered service packages (Basic, Premium, Ultimate)

**Implementation:**
- Target fiber customers without support (~1,200 customers)
- Discount margin: ~$10/month for 3 months
- Total cost: $36K

**Expected ROI:**
```
Current situation:
- 1,200 fiber customers without support
- 41% churn rate = 492 churners
- Lost revenue: 492 × $2,000 = $984K

After intervention:
- Target 20% churn reduction
- New churners: 492 × 80% = 394
- Customers saved: 98
- Revenue saved: 98 × $2,000 = $196K

Net benefit: $196K - $36K = $160K annually
```

---

### 📈 Total Business Impact

| Strategy | Investment | Revenue Saved | Net Benefit | ROI |
|----------|------------|---------------|-------------|-----|
| Contract Incentives | $75K | $464K | $389K | 519% |
| Early Intervention | $150K | $750K | $600K | 400% |
| Bundled Services | $36K | $196K | $160K | 444% |
| **TOTAL** | **$261K** | **$1.41M** | **$1.15M** | **440%** |

---

## 🛠️ Technical Implementation

### Technologies Used
- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine learning models
- **Matplotlib & Seaborn** - Data visualization

### Key Techniques Applied

1. **Data Preprocessing**
   - Handled missing values (11 rows with missing TotalCharges)
   - Type conversion for numerical features
   - Label encoding for categorical variables

2. **Exploratory Data Analysis**
   - Univariate analysis for each feature
   - Churn rate variance calculation
   - Customer segmentation and profiling

3. **Feature Engineering**
   - Tenure binning (0-1 year, 1-2 years, etc.)
   - Risk score calculation for customer segments

4. **Class Imbalance Handling**
   - Upsampling minority class (churners)
   - Created balanced dataset (50-50 split)
   - Prevented model bias toward majority class

5. **Model Development**
   - Train-test split (80-20)
   - Multiple algorithm testing
   - Hyperparameter tuning
   - Cross-validation

6. **Model Evaluation**
   - Confusion matrix analysis
   - Precision-Recall trade-off
   - Business-focused metric selection (recall over accuracy)

---

## 📊 Project Structure

```
telecom-churn-prediction/
│
├── TelcoChurnJupyter.ipynb          # Main analysis notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── README.md                         # Project documentation
└── images/                           # Visualizations (optional)
```

---

## 🚀 How to Run This Project

### Prerequisites
```bash
Python 3.8 or higher
Jupyter Notebook
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/telecom-churn-prediction.git
cd telecom-churn-prediction
```

2. **Install required packages**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook TelcoChurnJupyter.ipynb
```

4. **Run all cells**
- Click "Cell" → "Run All"
- Or run cells sequentially with Shift+Enter

---

## 📈 Skills Demonstrated

### Data Analysis
✅ Data cleaning and preprocessing  
✅ Exploratory data analysis (EDA)  
✅ Statistical analysis and hypothesis testing  
✅ Data visualization and storytelling  

### Machine Learning
✅ Classification modeling (Logistic Regression, Random Forest)  
✅ Handling imbalanced datasets  
✅ Model evaluation and selection  
✅ Feature importance analysis  
✅ Hyperparameter tuning  

### Business Analytics
✅ Business problem framing  
✅ KPI identification and tracking  
✅ ROI calculation and impact analysis  
✅ Customer segmentation and profiling  
✅ Strategic recommendation development  
✅ Stakeholder communication  

---

## 🎓 Key Learnings

### 1. Interpretability Often Beats Accuracy
I chose a model with 72% accuracy over one with 79% accuracy because:
- It caught 1% more churners (what actually matters)
- Business stakeholders could understand the "why"
- Trust and adoption > marginal accuracy gains

### 2. Class Imbalance is a Real Challenge
With 73% non-churners, the baseline model just predicted "no churn" for everyone and still got 73% accuracy. This taught me:
- Always check class distribution
- Accuracy can be misleading
- Focus on the metric that matters (recall for churn)

### 3. Data Tells the Story, But You Must Translate It
Technical findings mean nothing if stakeholders can't act on them:
- Turned "77% variance" into "month-to-month customers churn 14x more"
- Calculated ROI in dollars, not just percentages
- Connected every insight to a specific business action

### 4. Validation is Critical
When EDA and ML feature importance agreed (Contract, Tech Support, Online Security), it confirmed the model learned real patterns, not noise.

---


---

## 👤 About This Project

This is my second business analytics project, focusing on applying machine learning to solve real-world business problems. I leveraged modern development tools and open-source libraries while ensuring deep understanding of every analytical decision.

**My approach:**
- Used AI tools (Claude, Gemini) as coding assistants
- Made all analytical and strategic decisions independently
- Ensured I could explain and defend every choice
- Focused on business value, not just technical complexity

---

## 📬 Contact & Feedback

I'm actively learning and always open to feedback!

**Connect with me:**
- 📧 Email: aapande18044@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/anshul-pande-9b33632a4/
-  GitHub:   https://github.com/Astro907



---

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).


---
