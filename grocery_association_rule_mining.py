import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# Step 1: Load the dataset
def load_dataset(file_path='C:\\grocery.csv'):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("Error: File not found.")
        return None


# Step 2: Preprocess the data
def preprocess_data(data):
    transactions = data['items'].str.split(',')
    encoded_data = pd.get_dummies(transactions.apply(pd.Series).stack())
    encoded_data = encoded_data.groupby(level=0).sum()
    encoded_data = encoded_data.astype(bool)
    return encoded_data
# Step 3: Apply Apriori Algorithm
def apply_apriori(transactions, min_support, min_confidence):
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules

# Step 4: Apply FP-Growth Algorithm
def apply_fpgrowth(transactions, min_support, min_confidence):
    frequent_itemsets = fpgrowth(transactions, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules

# Step 5: Generate Association Rules
def generate_association_rules(itemsets, min_confidence):
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

# Step 6: Alter Support and Confidence Values
def experiment_with_thresholds(data, min_support_values, min_confidence_values):
    for min_support in min_support_values:
        for min_confidence in min_confidence_values:
            print(f"Min Support: {min_support}, Min Confidence: {min_confidence}")
            transactions = preprocess_data(data)
            # Apply Apriori Algorithm
            frequent_itemsets_apriori, rules_apriori = apply_apriori(transactions, min_support, min_confidence)
            print("Apriori Results:")
            print(frequent_itemsets_apriori)
            print(rules_apriori)
            # Generate association rules with lower confidence thresholds
            lower_confidence_rules = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.3)
            print("Association Rules with Lower Confidence Threshold:")
            print(lower_confidence_rules)

            # Apply FP-Growth Algorithm
            frequent_itemsets_fpgrowth, rules_fpgrowth = apply_fpgrowth(transactions, min_support, min_confidence)
            print("FP-Growth Results:")
            print(frequent_itemsets_fpgrowth)
            print(rules_fpgrowth)

# Load the dataset
data = load_dataset('grocery.csv')

if data is not None:
    # Experiment with different support and confidence values
    min_support_values = [0.001, 0.002]
    min_confidence_values = [0.5, 0.7]
    experiment_with_thresholds(data, min_support_values, min_confidence_values)
