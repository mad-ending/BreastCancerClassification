from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    # print(f'Accuracy: {accuracy}')
    # print(f'F1 Score: {f1}')
    # print(f'Recall: {recall}')
    # print(f'Precision: {precision}')
    # print(f'Confusion Matrix: \n{cm}')

    return accuracy, f1, recall, precision, auc, cm