# Fraud Detection Portfolio Project

A fraud detection AI agent, which calls a custom XGBoost model deployed on Google Cloud to score transactions for fraud, and subsequently provides explanations on the influential input features as well as the recommended next steps. Below is an example of the fraud agent's current outputs:

```
Here's a full breakdown of the fraud analysis results:

---

## 🟢 Verdict: Legitimate Transaction

| Detail | Value |
|---|---|
| **Fraud Probability** | ~0.0001% (essentially zero) |
| **Decision Threshold** | 23.6% |
| **Flagged as Fraud?** | ❌ No |

This transaction is **extremely unlikely to be fraudulent**. Its fraud probability of ~0.0001% is far below the 23.6% threshold needed to trigger an alert.

---

## 🔍 Top Contributing Features (SHAP Explanation)

All five top features **pushed the risk down**, reinforcing the legitimate classification:

| Rank | Feature | Value | Impact | What It Suggests |
|---|---|---|---|---|
| 1 | **V4** | -1.66 | ⬇️ Strongly decreases risk | V4 is one of the most powerful fraud indicators — its negative value here is strongly associated with **normal, non-fraudulent behavior** |
| 2 | **V14** | -0.51 | ⬇️ Decreases risk | V14 is another highly predictive feature; this value aligns well with **legitimate transaction patterns** |
| 3 | **V27** | 0.20 | ⬇️ Decreases risk | A mild positive value in V27 is consistent with **genuine cardholder activity** |
| 4 | **V12** | 0.02 | ⬇️ Decreases risk | V12 near zero is a neutral-to-positive signal, typical of **routine transactions** |
| 5 | **V16** | 1.03 | ⬇️ Decreases risk | A moderately elevated V16 contributes to a **lower fraud likelihood** |

---

## 💡 Summary & Recommendation

> ✅ **No action needed.** Every key signal in this transaction points away from fraud. The model's base fraud expectation (before seeing any features) started at ~4.7%, but the transaction's feature profile collectively **drove that probability down to near zero**. The transaction amount of **$175.66** is also unremarkable and does not raise any red flags.

This transaction can be safely **approved and processed**. 🟢
```


# Work In Progress
* At the moment, the agent relies on the outputs of a single XGBoost model for scoring, but I have ambitions to expand the agentic loop to make it multi-stage
    1. XGBoost for individual transaction-level scoring
    2. Graph Neural Network for fraud ring identification
    3. LSTM & Isolation Forest for transaction sequence-level anomaly scoring

* Other things the agent could do, include:
    1. Agentically generating a comprehensive fraud investigation report, including the key transaction attributes which raised suspicion, as well as pulling full transaction history of the account from a SQL database for downstream manual review by analysts
    2. For generating compliance reports, a RAG-type of system would be interesting to build, to reference the relevant regulatory policy documents


# Additional Context
I decided to work on this project as I am applying to a few fraud detection data science roles, but then I realized it was also a great opportunity to learn how to build AI agents and host custom machine learning models on MCP servers. I also wanted to recap the materials from a course I took two years ago on Graph Neural Networks (XC224: Machine Learning with Graphs by Prof. Jure Leskovec). 

I worked on some fraud detection ML validation projects for Synchrony Bank early in my professional career, and I was aware of the prevalence of XGBoost for transaction-level fraud scoring. XGBoost is popular in finance, and I've also seen it being used for credit scoring (i.e. VantageScore) and for anti-money laundering transaction monitoring applications. Therefore, this project could easily be converted to an AI agent for agentic credit decisioning, or for agentic generation of SARs reports. 

Some of the notable challenges of building a good fraud detection model include the fact that undetected cases of fraud are generally not identifiable in the data, and the sparsity of true-label examples presents a class imabalance problem when training an ML model. For this project, I paid special attention to selecting the right evaluation metrics for hyperparameter tuning, and made sure that crosstemporal contamination did not arise in how the train/eval/test splits were specified. The XGBoost model's 


<br>

![Confusion Matrix](images/confusion_matrix.png)
<br>



# Description of Files

* `fraud_detection.ipynb`
    * **Info**: Demo of an XGBoost classifier method to flag fraudulent transactions. I also built an AI agent which calls the model as a tool, providing explanations for the features which leading to an approve/decline decision. 
    * **Files**: 
* `gnn.ipynb`
    * **Info**: Demo of a Graph Neural Network based fraud detction model, to identify fraud rings 
* `fraud_agent_system.ipynb`
    * **Info**: Demo of an AI agent system that makes tool calls to the productionized XGBoost and GNN model endpoints, to explain whether a transaction is likely to be fraudulent or legitimate. If the transaction is determined to be fraud, the agent automatically populates fields in a investigative report to streamline downstream manual reviews. 