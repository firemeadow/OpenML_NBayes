from sklearn.datasets import fetch_openml
import numpy as np
import pandas

# Lucas Invernizzi
# Naive OpenML_NBayes Classifier
# Should work with all OpenML datasets

# Gaussian Probability for a value to be in a distribution
def gaussian_probability(val, mean, std_dev):
    num = -((val - mean) ** 2)
    den = 2 * (std_dev ** 2)
    if 0 <= den <= 0.00001:
        exponent = 999
    else:
        exponent = np.exp(num / den)

    den = np.sqrt(2 * np.pi) * std_dev
    if 0 <= den <= 0.00001:
        return 999

    return exponent / den


# Loads OpenML dataset given the ID
# Fills NaN values with attribute median if numerical, String Missing if categorical
# Returns dataset and target values in dataframes with same indexing
def load_data(dataset):
    print("Loading Data...")
    out = fetch_openml(data_id=dataset, as_frame=True)
    data = out['data']
    for i, col in data.items():
        if np.sum(col.isnull()) > 0:
            if col.dtype.name == 'category':
                data[i].fillna('Missing')
            else:
                data[i].fillna(np.median(col))
    print("Data loading complete, beginning classification.")
    return data.reset_index(drop=True), out['target'].reset_index(drop=True)


# Returns accuracy, precision, recall, specificity and F1 score
# Takes number of true positives, negatives and false positives, negatives
# Only used for binary classification problems
def binary_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = (2 * recall * precision) / (recall + precision)
    return accuracy, precision, recall, specificity, f1


# Returns the probability of a value in a attribute belonging to a class
# Takes attribute column series/dataframe, a value in the column,
# and a boolean series which is true for every index
# of the column that corresponds to the class being explored
def classify(col, val, matching_class, lam, num_unique_vals):
    # If this attribute should be treated as categorical
    if col.dtype.name == 'category' or col.unique().shape[0] < num_unique_vals:
        m_val = col.unique().shape[0]  # Set m=v

        # Boolean series which is true where the column matches the value
        matching_value = col == val

        # Boolean series which is true where the column matches the value and class label
        matching_c_v = np.logical_and(matching_class, matching_value)
        num_matching_c_v = np.sum(matching_c_v)
        num_matching_class = np.sum(matching_class)

        # Returns Laplace smoothed probability
        return (num_matching_c_v + lam) / (num_matching_class + m_val + lam)

    else:  # Attribute is continuous
        # Series only containing the values in the column that are in the class given
        matching_class_examples = col[np.where(matching_class == True)[0]]
        mean = np.mean(matching_class_examples)
        std_dev = np.std(matching_class_examples)
        return gaussian_probability(val, mean, std_dev)


# Evaluates the Naive OpenML_NBayes classifier on the given dataset
# Returns performance metrics
# Takes data, same indexed labels, Laplace Smoothing lambda values,
# and num of unique values an attribute needs to have to be considered continuous
def evaluate(test_data, test_labels, lam, num_unique_vals):
    # Initialize prediction measures
    if test_labels.cat.categories.shape[0] > 2:  # Only need correct/incorrect if more than two classes
        correct = 0
        incorrect = 0
    else:  # Can gain more metrics if the problem is binary classification
        pos_label = test_labels.cat.categories[0]  # Arbitrarily set positive label
        tp = 0
        tn = 0
        fp = 0
        fn = 0

    # For every example in test data
    for ex, row in test_data.iterrows():
        # Prints percent complete every 10ish percent
        # Not very good implementation, best I could think up off the top of my head
        p_comp = (ex / test_data.shape[0])
        if (ex + 1) % int(test_data.shape[0] / 10) == 0:
            print(str(int(np.round(p_comp, 1) * 100)) + "% Complete.")

        # Initialize dict to contain probabilities for this example being a part of every possible class
        probs = {}

        # For every possible class label
        for label in labels.cat.categories:
            # Boolean series which is true for every index of the column that corresponds to the class being explored
            matching_class = labels == label
            # Prior probability for this class label
            probs[label] = np.sum(matching_class) / labels.shape[0]

            # For every column in the data
            for attr_idx, col in data.items():
                # Get probability of this example's corresponding value in the current column being in the current class
                prob = classify(col, row[attr_idx], matching_class, lam, num_unique_vals)
                # Multiply by prior
                probs[label] *= prob

        # Label with the maximum probability
        predicted_label = max(probs, key=probs.get)
        # Actual label for this example
        real_label = test_labels[ex]

        # Assesses correctness of prediction
        if test_labels.cat.categories.shape[0] > 2:
            if predicted_label == real_label:
                correct += 1
            else:
                incorrect += 1
        else:
            if predicted_label == real_label:
                if predicted_label == pos_label:
                    tp += 1
                else:
                    tn += 1
            else:
                if predicted_label == pos_label:
                    fp += 1
                else:
                    fn += 1

    # Returns performance metrics
    if test_labels.cat.categories.shape[0] > 2:
        return correct / (correct + incorrect)
    else:
        return binary_metrics(tp, tn, fp, fn)


if __name__ == "__main__":
    lam = 1  # Lambda value for Laplace Smoothing
    num_unique_vals = 10  # Number of unique values needed for an attribute to be considered continuous
    dataset_id = 1494  # ID of OpenML dataset to use
    data, labels = load_data(dataset_id)

    # Performs Naive OpenML_NBayes classification on the data
    if labels.cat.categories.shape[0] > 2:
        acc = evaluate(data, labels, lam, num_unique_vals)
        print("Accuracy: " + str(np.round(acc * 100, 2)) + "%")
    else:
        acc, prec, rec, spec, f1 = evaluate(data, labels, lam, num_unique_vals)
        print("Accuracy: " + str(np.round(acc * 100, 2)) + "%")
        print("Precision: " + str(np.round(prec * 100, 2)) + "%")
        print("Recall: " + str(np.round(rec * 100, 2)) + "%")
        print("Specificity: " + str(np.round(spec * 100, 2)) + "%")
        print("F1 Score: " + str(np.round(f1 * 100, 2)) + "%")
