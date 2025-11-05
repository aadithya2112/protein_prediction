# ABSTRACT

---

Protein classification is a fundamental problem in bioinformatics with applications in drug discovery, disease diagnosis, and understanding biological processes. This project presents a machine learning approach to predict protein characteristics using data from the UniProt database. We focus on two classification tasks: (1) binary classification to predict whether a protein is an enzyme based on the presence of EC (Enzyme Commission) numbers, and (2) multi-class classification to categorize proteins into length-based categories (Very Short, Short, Medium, Long, Very Long).

Our dataset comprises 2,000 protein samples with features including protein length, keywords, Gene Ontology (GO) terms, and EC numbers. We performed comprehensive exploratory data analysis to understand data distribution, quality, and feature relationships. Feature engineering techniques were applied to create meaningful predictors such as keyword counts, GO term counts, annotation scores, and length-based features.

We implemented and compared two machine learning algorithms: Logistic Regression and Random Forest Classifier. For enzyme classification, the Random Forest model achieved an accuracy of 89.31% and ROC-AUC score of 0.9599, significantly outperforming Logistic Regression (76.03% accuracy, 0.8495 ROC-AUC). For length classification, Random Forest achieved 99.5% accuracy after careful feature engineering to avoid data leakage.

Key findings include: (1) protein length and annotation completeness are strong predictors of enzymatic function, (2) keyword and GO term counts provide valuable functional information, and (3) ensemble methods like Random Forest demonstrate superior performance over linear models for protein classification tasks. The models provide interpretable results with feature importance analysis revealing that keyword count and GO term count are the most discriminative features.

This work demonstrates the effectiveness of machine learning in protein classification and provides a foundation for future enhancements including deep learning approaches, sequence-based analysis, and integration with protein structure databases.

**Keywords:** Protein Classification, Machine Learning, Random Forest, Enzyme Prediction, Bioinformatics, Feature Engineering
