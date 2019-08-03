import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn import neighbors, datasets, preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")



def naivebayes():
    data = pd.read_csv('Data/Admission_Predict.csv')
    data['Decision'] = data['Chance of Admit '].apply(lambda x: 1 if x >= 0.8 else 0)
    # print(data.head(5))
    columns = data.columns
    X = data[columns[1:-2]]
    y = data[columns[-1]]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    X_train = X[:int(0.75 * len(X))]
    X_test = X[int(0.75 * len(X)):]
    y_train = y[:int(0.75 * len(X))]
    y_test = y[int(0.75 * len(X)):]

    # print(X_train)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X = scaler.transform(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # print(X_test)

    from sklearn.naive_bayes import GaussianNB

    # Create a Gaussian Classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(X_test)

    from sklearn import metrics

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Training Accuracy:", accuracy_score(y_train, gnb.predict(X_train)))

    print(confusion_matrix(y_test, y_pred))

    y_pred_proba = gnb.predict_proba(X_test)[:, 1]


    #Plot Comfusion Matrix for Naive Bayes
    plot.clf()
    cm = confusion_matrix(y_test, y_pred)
    plot.imshow(cm, interpolation='nearest', cmap=plot.cm.Wistia)
    classNames = ['Rejected', 'Accepted']
    plot.title('Confusion Matrix - Naive bayes')
    plot.ylabel('True label')
    plot.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plot.xticks(tick_marks, classNames, rotation=45)
    plot.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plot.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    # plot.savefig("Confusion Matrix_Naive.png")
    plot.show()


    # Accuracy graphs
    method = ['training', 'test']
    score = [accuracy_score(y_train, gnb.predict(X_train)), accuracy_score(y_test, y_pred)]
    sns.barplot(method, score)
    plot.title("Train Vs Test - Naive Bayes")
    # plot.savefig("TrainvsTest.png")
    plot.show()

    return y_pred, y_pred_proba

def svm():
    data = pd.read_csv('Data/Admission_Predict.csv')

    data['Decision'] = data['Chance of Admit '].apply(lambda x: 1 if x >= 0.8 else 0)
    # data.to_csv('SampleData.csv')
    print(data.describe().transpose())
    columns = data.columns
    X = data[columns[1:-2]]

    y = data[columns[-1]]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    X_train = X[:int(0.75 * len(X))]
    X_test = X[int(0.75 * len(X)):]
    y_train = y[:int(0.75 * len(X))]
    y_test = y[int(0.75 * len(X)):]

    scaler = preprocessing.StandardScaler().fit(X_train)

    X = scaler.transform(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    svc = SVC(kernel='linear', probability=True)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    y_pred_full = svc.predict(X)

    # print(X_test, y_pred)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_train, svc.predict(X_train)))
    print(confusion_matrix(y_test, y_pred))

    data['Pred_decision'] = y_pred_full
    # print(data)
    y_pred_proba = svc.predict_proba(X_test)[:, 1]



    # Plot Comfusion Matrix for Naive Bayes
    plot.clf()
    cm = confusion_matrix(y_test, y_pred)
    plot.imshow(cm, interpolation='nearest', cmap=plot.cm.Wistia)
    classNames = ['Rejected', 'Accepted']
    plot.title('Confusion Matrix - SVM')
    plot.ylabel('True label')
    plot.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plot.xticks(tick_marks, classNames, rotation=45)
    plot.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plot.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    # plot.savefig("Confusion Matrix_Naive.png")
    plot.show()

    # Accuracy graphs
    method = ['training', 'test']
    score = [accuracy_score(y_train, svc.predict(X_train)), accuracy_score(y_test, y_pred)]
    sns.barplot(method, score)
    plot.title("Train Vs Test - SVM")
    # plot.savefig("TrainvsTest.png")
    plot.show()

    return y_pred, y_pred_proba

def knn():
    ## Get values for naive bayes and SVM
    y_pred_naive, y_pred_naive_proba = naivebayes()
    y_pred_svm, y_pred_svm_proba = svm()

    data = pd.read_csv('Data/Admission_Predict.csv')
    data['Decision'] = data['Chance of Admit '].apply(lambda x: 1 if x >= 0.8 else 0)
    # print(data.head(5))
    columns = data.columns
    X = data[columns[1:-2]]

    y = data[columns[-1]]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    X_train = X[:int(0.75 * len(X))]
    X_test = X[int(0.75 * len(X)):]
    y_train = y[:int(0.75 * len(X))]
    y_test = y[int(0.75 * len(X)):]

    # plot.scatter(X_train[0], y_train)
    # plot.show()
    # print(X_train)

    scaler = preprocessing.StandardScaler().fit(X_train)
    # scaler = preprocessing.MinMaxScaler().fit(X_train)
    X = scaler.transform(X)
    X_train = scaler.transform(X_train)
    # print(X_train)
    X_test = scaler.transform(X_test)

    scaled_df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
    # scaled_df_min = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
    # print(scaled_df)
    fig, (ax1, ax2) = plot.subplots(ncols=2, figsize=(6, 5))
    ax1.set_title('Before Scaling')
    sns.kdeplot(data['GRE Score'], ax=ax1)
    sns.kdeplot(data['TOEFL Score'], ax=ax1)
    sns.kdeplot(data['CGPA'], ax=ax1)
    ax2.set_title('After Standard Scaler')
    sns.kdeplot(scaled_df['x1'], ax=ax2)
    sns.kdeplot(scaled_df['x2'], ax=ax2)
    sns.kdeplot(scaled_df['x3'], ax=ax2)
    plot.show()

    # print(X_test)

    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    y_pred_full = knn.predict(X)

    data['Pred_decision'] = y_pred_full

    # print(X_test, y_pred)
    print("Test Score", accuracy_score(y_test, y_pred))
    print("Train Score", accuracy_score(y_train, knn.predict(X_train)))

    method = ['training', 'test']
    score = [accuracy_score(y_train, knn.predict(X_train)), accuracy_score(y_test, y_pred)]
    sns.barplot(method, score)
    plot.title("Train Vs Test - knn")
    plot.savefig("TrainvsTest.png")
    plot.show()

    print(confusion_matrix(y_test, y_pred))

    plot.clf()
    cm = confusion_matrix(y_test, y_pred)
    plot.imshow(cm, interpolation='nearest', cmap=plot.cm.Wistia)
    classNames = ['Rejected', 'Accepted']
    plot.title('Confusion Matrix - knn')
    plot.ylabel('True label')
    plot.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plot.xticks(tick_marks, classNames, rotation=45)
    plot.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plot.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    plot.savefig("Confusion Matrix.png")
    plot.show()

    # Plot ROC
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    # Knn
    knn_ROC_AUC = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, knn.predict_proba(X_test)[:, 1])

    # Naive Bayes
    bayes_ROC_AUC = roc_auc_score(y_test, y_pred_naive)
    fpr_2, tpr_2, thres_2 = roc_curve(y_test, y_pred_naive_proba)

    # SVM
    SVM_ROC_AUC = roc_auc_score(y_test, y_pred_svm)
    fpr_3, tpr_3, thres_3 = roc_curve(y_test, y_pred_svm_proba)

    plot.figure()
    plot.plot(fpr, tpr, color='darkorange', label="KNN (area = %0.2f)" % knn_ROC_AUC)
    plot.plot(fpr_2, tpr_2, label="Naive Bayes (area = %0.2f)" % bayes_ROC_AUC)
    plot.plot(fpr_3, tpr_3, color='green', label="SVM (area = %0.2f)" % SVM_ROC_AUC)
    plot.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plot.xlim([-0.1, 1.05])
    plot.ylim([-0.1, 1.05])
    plot.xlabel("False Positive Rate")
    plot.ylabel("True Positive Rate")
    plot.title('Receiver operating characteristic')
    plot.legend(loc="lower right")
    plot.show()

    data.to_csv('Data/FullData.csv')
    print("knn: {}, Naive Bayes: {}, SVM: {}".format(mean_absolute_error(y_test, y_pred),
                                                     mean_absolute_error(y_test, y_pred_naive),
                                                     mean_absolute_error(y_test, y_pred_svm)))
    print(classification_report(y_test, y_pred))
    


knn()