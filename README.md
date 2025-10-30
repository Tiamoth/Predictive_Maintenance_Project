⚙️ Predictive Maintenance for Industrial Machinery

Project Objective
My goal for this project was to build a machine learning model that predicts machine failure using sensor data. This model can help a business move from "reactive maintenance" (fixing things after they break) to "predictive maintenance" (fixing things before they break), saving on costs and downtime.

Tools Used
Python: The core programming language.
Pandas: For loading, cleaning, and manipulating the data.
Matplotlib & Seaborn: For data visualization.
Scikit-learn: For building and evaluating the machine learning model.
Imbalanced-learn (SMOTE): To handle the imbalanced dataset.


📈 The Process

1. Exploratory Data Analysis (EDA)
I started by loading the data and visualizing the relationships between sensor readings and machine failure. My key findings were:
High Torque: The median torque for a failed machine was significantly higher than for a healthy one.
High Tool Wear: Similarly, tool wear was a strong indicator of an impending failure.

2. Feature Engineering
Using my mechatronics knowledge, I created new, more powerful features from the raw sensor data. I hypothesized that these combined features would be more predictive than the raw signals alone.
-`temp_diff` = `Process temperature [K]` - `Air temperature [K]`
-`power` = `Torque [Nm]` * `Rotational speed [rpm]`
- `tool_strain` = `Torque [Nm]` * `Tool wear [min]`

3. Data Pre-processing (Handling Imbalance)
The dataset was **highly imbalanced** (96.6% healthy vs. 3.4% failure). A model trained on this would just learn to "always guess healthy." To fix this, I used the SMOTE (Synthetic Minority Over-sampling Technique) to oversample the rare "failure" cases in the training data, creating a balanced dataset for the model to learn from.

4. Modeling
I trained a `RandomForestClassifier` on the balanced, scaled, and engineered data. I chose this model because it is powerful, non-linear, and can provide feature importance scores.


📊 Results & Key Insights

The final model performed extremely well on the unseen test data, achieving 98% overall accuracy.

More importantly, it achieved a Recall of 81% for the "Failure" class. This means the model successfully caught 81% of all actual failures, which is the most critical metric for this business problem.

Feature Importance
The best part of the project was seeing why the model worked. By plotting the feature importance, the model confirmed my engineering intuition.

My custom-engineered feature, `power`, was the 3rd most important predictor of failure.
 `Rotational speed` and `Torque` were the top two, proving that the physical forces on the machine are the clearest indicators of failure.

![Feature Importance Chart](feature_importance_plot.png)
![Feature Importance Chart](feature_importance_plot.png)
