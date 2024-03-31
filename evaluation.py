from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(y_test, y_pred, show=False):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    if show:
        print(f'Accuracy: {accuracy.round(4)}')
        print(f'F1 Score: {f1.round(4)}')
        print(f'Recall: {recall.round(4)}')
        print(f'Precision: {precision.round(4)}')
        print(f'AUC: {auc.round(4)}')
        print(f'Confusion Matrix: \n{cm}')

        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.show()

    return accuracy, f1, recall, precision, auc, cm