import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Select features and target
X = data[['credit_lines_outstanding', 'loan_amt_outstanding',
          'total_debt_outstanding', 'income', 'years_employed', 'fico_score']]
y = data['default']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

# Decision Tree Model
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Function: Expected Loss (Logistic Regression)
def expected_loss_logistic(credit_lines_outstanding, loan_amt_outstanding,
                           total_debt_outstanding, income, years_employed, fico_score):
    input_data = pd.DataFrame([[credit_lines_outstanding, loan_amt_outstanding,
                                total_debt_outstanding, income, years_employed, fico_score]],
                              columns=['credit_lines_outstanding', 'loan_amt_outstanding',
                                       'total_debt_outstanding', 'income', 'years_employed', 'fico_score'])
    
    input_scaled = scaler.transform(input_data)
    pd_default = log_reg.predict_proba(input_scaled)[0][1]   # Probability of Default
    
    recovery_rate = 0.10
    expected_loss_value = pd_default * (1 - recovery_rate) * loan_amt_outstanding
    
    return pd_default, expected_loss_value

# Function: Expected Loss (Decision Tree)
def expected_loss_tree(credit_lines_outstanding, loan_amt_outstanding,
                       total_debt_outstanding, income, years_employed, fico_score):
    input_data = pd.DataFrame([[credit_lines_outstanding, loan_amt_outstanding,
                                total_debt_outstanding, income, years_employed, fico_score]],
                              columns=['credit_lines_outstanding', 'loan_amt_outstanding',
                                       'total_debt_outstanding', 'income', 'years_employed', 'fico_score'])
    
    pd_default = tree.predict_proba(input_data)[0][1]   # Probability of Default
    
    recovery_rate = 0.10
    expected_loss_value = pd_default * (1 - recovery_rate) * loan_amt_outstanding
    
    return pd_default, expected_loss_value

# Apply to entire dataset

# Logistic model predictions
scaled_features = scaler.transform(X)
data['PD_Logistic'], data['EL_Logistic'] = zip(*[
    expected_loss_logistic(row.credit_lines_outstanding, row.loan_amt_outstanding,
                           row.total_debt_outstanding, row.income,
                           row.years_employed, row.fico_score)
    for row in X.itertuples(index=False)
])

# Decision Tree model predictions
data['PD_Tree'], data['EL_Tree'] = zip(*[
    expected_loss_tree(row.credit_lines_outstanding, row.loan_amt_outstanding,
                       row.total_debt_outstanding, row.income,
                       row.years_employed, row.fico_score)
    for row in X.itertuples(index=False)
])

# Show results
print(data.head())

# Save output to CSV for analysis
data.to_csv("Loan_Data_with_PD_EL.csv", index=False)
print("\nResults saved to 'Loan_Data_with_PD_EL.csv'")
