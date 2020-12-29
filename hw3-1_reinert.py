# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:36:29 2020

@author: reine
"""
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# Load the Iris Dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Create the dataframe
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
target_names = [iris.target_names[i] for i in iris.target]
iris_df['species'] = target_names

# Pair plot
#sns.pairplot(iris_df) # <-- Uncomment for pairplot
#sns.pairplot(iris_df, hue = 'species') # <-- Uncomment for pairplot with multi-color

# Split into Train/Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42, stratify = Y)

# Train the SVC
# --- GAMMA AS 1.0 ---
svc = SVC(kernel='rbf', gamma=1.0)
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
conf_mtrx_01 = confusion_matrix(Y_test, Y_predict)

#Key Performance Indicators of the Model

print('Confusion Matrix and Key Performance Indicators of the SVC Model -- GAMMA = 1.0')
print(conf_mtrx_01)

total_right = conf_mtrx_01[0][0] + conf_mtrx_01[1][1] + conf_mtrx_01[2][2]
total_wrong = conf_mtrx_01[0][1] + conf_mtrx_01[0][2] + conf_mtrx_01[1][0] + conf_mtrx_01[1][2] + conf_mtrx_01[2][0] + conf_mtrx_01[2][1]
total_acc = total_right / (total_right + total_wrong)
print(f'The total accuracy of the SVC is {total_acc:.3f}')

misclass = total_wrong / (total_right + total_wrong)
print(f'The total misclassification rate (error) of the SVC is {misclass:.3f}\n')

# TP, FP, FN for Setosa
tp_setosa = conf_mtrx_01[0][0]
fp_setosa = conf_mtrx_01[1][0]+conf_mtrx_01[2][0]
fn_setosa = conf_mtrx_01[0][1]+conf_mtrx_01[0][2]
tn_setosa = conf_mtrx_01[1][1]+conf_mtrx_01[1][2]+conf_mtrx_01[2][1]+conf_mtrx_01[2][2]
precision_setosa = tp_setosa / (tp_setosa + fp_setosa)
recall_setosa = tp_setosa / (tp_setosa + fn_setosa)
specificity_setosa = tn_setosa / (tn_setosa + fp_setosa)
f1_setosa = (2 * precision_setosa * recall_setosa) / (precision_setosa + recall_setosa)

# TP, FP, FN for Versicolor
tp_versicolor = conf_mtrx_01[1][1]
fp_versicolor = conf_mtrx_01[0][1]+conf_mtrx_01[2][1]
fn_versicolor = conf_mtrx_01[1][0]+conf_mtrx_01[1][2]
tn_versicolor = conf_mtrx_01[0][0]+conf_mtrx_01[0][2]+conf_mtrx_01[2][0]+conf_mtrx_01[2][2]
precision_versicolor = tp_versicolor / (tp_versicolor + fp_versicolor)
recall_versicolor = tp_versicolor / (tp_versicolor + fn_versicolor)
specificity_versicolor = tn_versicolor / (tn_versicolor + fp_versicolor)
f1_versicolor = (2 * precision_versicolor * recall_versicolor) / (precision_versicolor + recall_versicolor)

# TP, FP, FN for Virginica
tp_virginica = conf_mtrx_01[2][2]
fp_virginica = conf_mtrx_01[0][2]+conf_mtrx_01[1][2]
fn_virginica = conf_mtrx_01[2][0]+conf_mtrx_01[2][1]
tn_virginica = conf_mtrx_01[0][0]+conf_mtrx_01[0][1]+conf_mtrx_01[1][0]+conf_mtrx_01[1][1]
precision_virginica = tp_virginica / (tp_virginica + fp_virginica)
recall_virginica = tp_virginica / (tp_virginica + fn_virginica)
specificity_virginica = tn_virginica / (tn_virginica + fp_virginica)
f1_virginica = (2 * precision_virginica * recall_virginica) / (precision_virginica + recall_virginica)

print(f'The precision for Setosa is {precision_setosa:.3f}')
print(f'The recall for Setosa is {recall_setosa:.3f}')
print(f'The specificity for Setosa is {specificity_setosa:.3f}')
print(f'The F1 Score for Setosa is {f1_setosa:.3f}\n')

print(f'The precision for Versicolor is {precision_versicolor:.3f}')
print(f'The recall for Versicolor is {recall_versicolor:.3f}')
print(f'The specificity for Versicolor is {specificity_versicolor:.3f}')
print(f'The F1 Score for Versicolor is {f1_versicolor:.3f}\n')

print(f'The precision for Virgnica is {precision_virginica:.3f}')
print(f'The recall for Virgnica is {recall_virginica:.3f}')
print(f'The specificity for Virginica is {specificity_virginica:.3f}')
print(f'The F1 Score for Virgnica is {f1_virginica:.3f}\n')

# --- GAMMA AS 1.0 ---
svc = SVC(kernel='rbf', gamma=10)
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
conf_mtrx_10 = confusion_matrix(Y_test, Y_predict)

#Key Performance Indicators of the Model

print('Confusion Matrix and Key Performance Indicators of the SVC Model -- GAMMA = 10')
print(conf_mtrx_10)

total_right = conf_mtrx_10[0][0] + conf_mtrx_10[1][1] + conf_mtrx_10[2][2]
total_wrong = conf_mtrx_10[0][1] + conf_mtrx_10[0][2] + conf_mtrx_10[1][0] + conf_mtrx_10[1][2] + conf_mtrx_10[2][0] + conf_mtrx_10[2][1]
total_acc = total_right / (total_right + total_wrong)
print(f'The total accuracy of the SVC is {total_acc:.3f}')

misclass = total_wrong / (total_right + total_wrong)
print(f'The total misclassification rate (error) of the SVC is {misclass:.3f}\n')

# TP, FP, FN for Setosa
tp_setosa = conf_mtrx_10[0][0]
fp_setosa = conf_mtrx_10[1][0]+conf_mtrx_10[2][0]
fn_setosa = conf_mtrx_10[0][1]+conf_mtrx_10[0][2]
tn_setosa = conf_mtrx_10[1][1]+conf_mtrx_10[1][2]+conf_mtrx_10[2][1]+conf_mtrx_10[2][2]
precision_setosa = tp_setosa / (tp_setosa + fp_setosa)
recall_setosa = tp_setosa / (tp_setosa + fn_setosa)
specificity_setosa = tn_setosa / (tn_setosa + fp_setosa)
f1_setosa = (2 * precision_setosa * recall_setosa) / (precision_setosa + recall_setosa)

# TP, FP, FN for Versicolor
tp_versicolor = conf_mtrx_10[1][1]
fp_versicolor = conf_mtrx_10[0][1]+conf_mtrx_10[2][1]
fn_versicolor = conf_mtrx_10[1][0]+conf_mtrx_10[1][2]
tn_versicolor = conf_mtrx_10[0][0]+conf_mtrx_10[0][2]+conf_mtrx_10[2][0]+conf_mtrx_10[2][2]
precision_versicolor = tp_versicolor / (tp_versicolor + fp_versicolor)
recall_versicolor = tp_versicolor / (tp_versicolor + fn_versicolor)
specificity_versicolor = tn_versicolor / (tn_versicolor + fp_versicolor)
f1_versicolor = (2 * precision_versicolor * recall_versicolor) / (precision_versicolor + recall_versicolor)

# TP, FP, FN for Virginica
tp_virginica = conf_mtrx_10[2][2]
fp_virginica = conf_mtrx_10[0][2]+conf_mtrx_10[1][2]
fn_virginica = conf_mtrx_10[2][0]+conf_mtrx_10[2][1]
tn_virginica = conf_mtrx_10[0][0]+conf_mtrx_10[0][1]+conf_mtrx_10[1][0]+conf_mtrx_10[1][1]
precision_virginica = tp_virginica / (tp_virginica + fp_virginica)
recall_virginica = tp_virginica / (tp_virginica + fn_virginica)
specificity_virginica = tn_virginica / (tn_virginica + fp_virginica)
f1_virginica = (2 * precision_virginica * recall_virginica) / (precision_virginica + recall_virginica)

print(f'The precision for Setosa is {precision_setosa:.3f}')
print(f'The recall for Setosa is {recall_setosa:.3f}')
print(f'The specificity for Setosa is {specificity_setosa:.3f}')
print(f'The F1 Score for Setosa is {f1_setosa:.3f}\n')

print(f'The precision for Versicolor is {precision_versicolor:.3f}')
print(f'The recall for Versicolor is {recall_versicolor:.3f}')
print(f'The specificity for Versicolor is {specificity_versicolor:.3f}')
print(f'The F1 Score for Versicolor is {f1_versicolor:.3f}\n')

print(f'The precision for Virgnica is {precision_virginica:.3f}')
print(f'The recall for Virgnica is {recall_virginica:.3f}')
print(f'The specificity for Virginica is {specificity_virginica:.3f}')
print(f'The F1 Score for Virgnica is {f1_virginica:.3f}\n')

# --- GAMMA AS 3.0 ---
svc = SVC(kernel='rbf', gamma=3.0)
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
conf_mtrx_03 = confusion_matrix(Y_test, Y_predict)

#Key Performance Indicators of the Model

print('Confusion Matrix and Key Performance Indicators of the SVC Model -- GAMMA = 3.0')
print(conf_mtrx_03)

total_right = conf_mtrx_03[0][0] + conf_mtrx_03[1][1] + conf_mtrx_03[2][2]
total_wrong = conf_mtrx_03[0][1] + conf_mtrx_03[0][2] + conf_mtrx_03[1][0] + conf_mtrx_03[1][2] + conf_mtrx_03[2][0] + conf_mtrx_03[2][1]
total_acc = total_right / (total_right + total_wrong)
print(f'The total accuracy of the SVC is {total_acc:.3f}')

misclass = total_wrong / (total_right + total_wrong)
print(f'The total misclassification rate (error) of the SVC is {misclass:.3f}\n')

# TP, FP, FN for Setosa
tp_setosa = conf_mtrx_03[0][0]
fp_setosa = conf_mtrx_03[1][0]+conf_mtrx_03[2][0]
fn_setosa = conf_mtrx_03[0][1]+conf_mtrx_03[0][2]
tn_setosa = conf_mtrx_03[1][1]+conf_mtrx_03[1][2]+conf_mtrx_03[2][1]+conf_mtrx_03[2][2]
precision_setosa = tp_setosa / (tp_setosa + fp_setosa)
recall_setosa = tp_setosa / (tp_setosa + fn_setosa)
specificity_setosa = tn_setosa / (tn_setosa + fp_setosa)
f1_setosa = (2 * precision_setosa * recall_setosa) / (precision_setosa + recall_setosa)

# TP, FP, FN for Versicolor
tp_versicolor = conf_mtrx_03[1][1]
fp_versicolor = conf_mtrx_03[0][1]+conf_mtrx_03[2][1]
fn_versicolor = conf_mtrx_03[1][0]+conf_mtrx_03[1][2]
tn_versicolor = conf_mtrx_03[0][0]+conf_mtrx_03[0][2]+conf_mtrx_03[2][0]+conf_mtrx_03[2][2]
precision_versicolor = tp_versicolor / (tp_versicolor + fp_versicolor)
recall_versicolor = tp_versicolor / (tp_versicolor + fn_versicolor)
specificity_versicolor = tn_versicolor / (tn_versicolor + fp_versicolor)
f1_versicolor = (2 * precision_versicolor * recall_versicolor) / (precision_versicolor + recall_versicolor)

# TP, FP, FN for Virginica
tp_virginica = conf_mtrx_03[2][2]
fp_virginica = conf_mtrx_03[0][2]+conf_mtrx_03[1][2]
fn_virginica = conf_mtrx_03[2][0]+conf_mtrx_03[2][1]
tn_virginica = conf_mtrx_03[0][0]+conf_mtrx_03[0][1]+conf_mtrx_03[1][0]+conf_mtrx_03[1][1]
precision_virginica = tp_virginica / (tp_virginica + fp_virginica)
recall_virginica = tp_virginica / (tp_virginica + fn_virginica)
specificity_virginica = tn_virginica / (tn_virginica + fp_virginica)
f1_virginica = (2 * precision_virginica * recall_virginica) / (precision_virginica + recall_virginica)

print(f'The precision for Setosa is {precision_setosa:.3f}')
print(f'The recall for Setosa is {recall_setosa:.3f}')
print(f'The specificity for Setosa is {specificity_setosa:.3f}')
print(f'The F1 Score for Setosa is {f1_setosa:.3f}\n')

print(f'The precision for Versicolor is {precision_versicolor:.3f}')
print(f'The recall for Versicolor is {recall_versicolor:.3f}')
print(f'The specificity for Versicolor is {specificity_versicolor:.3f}')
print(f'The F1 Score for Versicolor is {f1_versicolor:.3f}\n')

print(f'The precision for Virgnica is {precision_virginica:.3f}')
print(f'The recall for Virgnica is {recall_virginica:.3f}')
print(f'The specificity for Virginica is {specificity_virginica:.3f}')
print(f'The F1 Score for Virgnica is {f1_virginica:.3f}\n')

"""---HOMEWORK 3-1---"""
"""Kernel = Poly | C = 2 | Degree = 5 | Gamma = 3"""
kern = 'poly'
c = 2
deg = 5
gam = 3
print(f'Kernal = {kern} | C = {c} | Degree = {deg} | Gamma = {gam}')
svc = SVC(kernel=kern, C = c , degree = deg , gamma = gam)
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
conf_mtrx = confusion_matrix(Y_test, Y_predict)

#Key Performance Indicators of the Model

print('Confusion Matrix and Key Performance Indicators of the SVC Model')
print(conf_mtrx)

total_right = conf_mtrx[0][0] + conf_mtrx[1][1] + conf_mtrx[2][2]
total_wrong = conf_mtrx[0][1] + conf_mtrx[0][2] + conf_mtrx[1][0] + conf_mtrx[1][2] + conf_mtrx[2][0] + conf_mtrx[2][1]
total_acc = total_right / (total_right + total_wrong)
print(f'The total accuracy of the SVC is {total_acc:.3f}')

misclass = total_wrong / (total_right + total_wrong)
print(f'The total misclassification rate (error) of the SVC is {misclass:.3f}\n')

# TP, FP, FN for Setosa
tp_setosa = conf_mtrx[0][0]
fp_setosa = conf_mtrx[1][0]+conf_mtrx[2][0]
fn_setosa = conf_mtrx[0][1]+conf_mtrx[0][2]
tn_setosa = conf_mtrx[1][1]+conf_mtrx[1][2]+conf_mtrx[2][1]+conf_mtrx[2][2]
precision_setosa = tp_setosa / (tp_setosa + fp_setosa)
recall_setosa = tp_setosa / (tp_setosa + fn_setosa)
specificity_setosa = tn_setosa / (tn_setosa + fp_setosa)
f1_setosa = (2 * precision_setosa * recall_setosa) / (precision_setosa + recall_setosa)

# TP, FP, FN for Versicolor
tp_versicolor = conf_mtrx[1][1]
fp_versicolor = conf_mtrx[0][1]+conf_mtrx[2][1]
fn_versicolor = conf_mtrx[1][0]+conf_mtrx[1][2]
tn_versicolor = conf_mtrx[0][0]+conf_mtrx[0][2]+conf_mtrx[2][0]+conf_mtrx[2][2]
precision_versicolor = tp_versicolor / (tp_versicolor + fp_versicolor)
recall_versicolor = tp_versicolor / (tp_versicolor + fn_versicolor)
specificity_versicolor = tn_versicolor / (tn_versicolor + fp_versicolor)
f1_versicolor = (2 * precision_versicolor * recall_versicolor) / (precision_versicolor + recall_versicolor)

# TP, FP, FN for Virginica
tp_virginica = conf_mtrx[2][2]
fp_virginica = conf_mtrx[0][2]+conf_mtrx[1][2]
fn_virginica = conf_mtrx[2][0]+conf_mtrx[2][1]
tn_virginica = conf_mtrx[0][0]+conf_mtrx[0][1]+conf_mtrx[1][0]+conf_mtrx[1][1]
precision_virginica = tp_virginica / (tp_virginica + fp_virginica)
recall_virginica = tp_virginica / (tp_virginica + fn_virginica)
specificity_virginica = tn_virginica / (tn_virginica + fp_virginica)
f1_virginica = (2 * precision_virginica * recall_virginica) / (precision_virginica + recall_virginica)

print(f'The precision for Setosa is {precision_setosa:.3f}')
print(f'The recall for Setosa is {recall_setosa:.3f}')
print(f'The specificity for Setosa is {specificity_setosa:.3f}')
print(f'The F1 Score for Setosa is {f1_setosa:.3f}\n')

print(f'The precision for Versicolor is {precision_versicolor:.3f}')
print(f'The recall for Versicolor is {recall_versicolor:.3f}')
print(f'The specificity for Versicolor is {specificity_versicolor:.3f}')
print(f'The F1 Score for Versicolor is {f1_versicolor:.3f}\n')

print(f'The precision for Virgnica is {precision_virginica:.3f}')
print(f'The recall for Virgnica is {recall_virginica:.3f}')
print(f'The specificity for Virginica is {specificity_virginica:.3f}')
print(f'The F1 Score for Virgnica is {f1_virginica:.3f}\n')

"""Kernel = Poly | C = 5 | Degree = 2 | Gamma = 10"""
kern = 'poly'
c = 5
deg = 2
gam = 10
print(f'Kernal = {kern} | C = {c} | Degree = {deg} | Gamma = {gam}')
svc = SVC(kernel=kern, C = c , degree = deg , gamma = gam)
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
conf_mtrx = confusion_matrix(Y_test, Y_predict)

#Key Performance Indicators of the Model

print('Confusion Matrix and Key Performance Indicators of the SVC Model')
print(conf_mtrx)

total_right = conf_mtrx[0][0] + conf_mtrx[1][1] + conf_mtrx[2][2]
total_wrong = conf_mtrx[0][1] + conf_mtrx[0][2] + conf_mtrx[1][0] + conf_mtrx[1][2] + conf_mtrx[2][0] + conf_mtrx[2][1]
total_acc = total_right / (total_right + total_wrong)
print(f'The total accuracy of the SVC is {total_acc:.3f}')

misclass = total_wrong / (total_right + total_wrong)
print(f'The total misclassification rate (error) of the SVC is {misclass:.3f}\n')

# TP, FP, FN for Setosa
tp_setosa = conf_mtrx[0][0]
fp_setosa = conf_mtrx[1][0]+conf_mtrx[2][0]
fn_setosa = conf_mtrx[0][1]+conf_mtrx[0][2]
tn_setosa = conf_mtrx[1][1]+conf_mtrx[1][2]+conf_mtrx[2][1]+conf_mtrx[2][2]
precision_setosa = tp_setosa / (tp_setosa + fp_setosa)
recall_setosa = tp_setosa / (tp_setosa + fn_setosa)
specificity_setosa = tn_setosa / (tn_setosa + fp_setosa)
f1_setosa = (2 * precision_setosa * recall_setosa) / (precision_setosa + recall_setosa)

# TP, FP, FN for Versicolor
tp_versicolor = conf_mtrx[1][1]
fp_versicolor = conf_mtrx[0][1]+conf_mtrx[2][1]
fn_versicolor = conf_mtrx[1][0]+conf_mtrx[1][2]
tn_versicolor = conf_mtrx[0][0]+conf_mtrx[0][2]+conf_mtrx[2][0]+conf_mtrx[2][2]
precision_versicolor = tp_versicolor / (tp_versicolor + fp_versicolor)
recall_versicolor = tp_versicolor / (tp_versicolor + fn_versicolor)
specificity_versicolor = tn_versicolor / (tn_versicolor + fp_versicolor)
f1_versicolor = (2 * precision_versicolor * recall_versicolor) / (precision_versicolor + recall_versicolor)

# TP, FP, FN for Virginica
tp_virginica = conf_mtrx[2][2]
fp_virginica = conf_mtrx[0][2]+conf_mtrx[1][2]
fn_virginica = conf_mtrx[2][0]+conf_mtrx[2][1]
tn_virginica = conf_mtrx[0][0]+conf_mtrx[0][1]+conf_mtrx[1][0]+conf_mtrx[1][1]
precision_virginica = tp_virginica / (tp_virginica + fp_virginica)
recall_virginica = tp_virginica / (tp_virginica + fn_virginica)
specificity_virginica = tn_virginica / (tn_virginica + fp_virginica)
f1_virginica = (2 * precision_virginica * recall_virginica) / (precision_virginica + recall_virginica)

print(f'The precision for Setosa is {precision_setosa:.3f}')
print(f'The recall for Setosa is {recall_setosa:.3f}')
print(f'The specificity for Setosa is {specificity_setosa:.3f}')
print(f'The F1 Score for Setosa is {f1_setosa:.3f}\n')

print(f'The precision for Versicolor is {precision_versicolor:.3f}')
print(f'The recall for Versicolor is {recall_versicolor:.3f}')
print(f'The specificity for Versicolor is {specificity_versicolor:.3f}')
print(f'The F1 Score for Versicolor is {f1_versicolor:.3f}\n')

print(f'The precision for Virgnica is {precision_virginica:.3f}')
print(f'The recall for Virgnica is {recall_virginica:.3f}')
print(f'The specificity for Virginica is {specificity_virginica:.3f}')
print(f'The F1 Score for Virgnica is {f1_virginica:.3f}\n')

"""Kernel = Linear | C = .1 | Degree = 2 | Gamma = 1"""
kern = 'linear'
c = .1
deg = 2
gam = 1
print(f'Kernal = {kern} | C = {c} | Degree = {deg} | Gamma = {gam}')
svc = SVC(kernel=kern, C = c , degree = deg , gamma = gam)
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
conf_mtrx = confusion_matrix(Y_test, Y_predict)

#Key Performance Indicators of the Model

print('Confusion Matrix and Key Performance Indicators of the SVC Model')
print(conf_mtrx)

total_right = conf_mtrx[0][0] + conf_mtrx[1][1] + conf_mtrx[2][2]
total_wrong = conf_mtrx[0][1] + conf_mtrx[0][2] + conf_mtrx[1][0] + conf_mtrx[1][2] + conf_mtrx[2][0] + conf_mtrx[2][1]
total_acc = total_right / (total_right + total_wrong)
print(f'The total accuracy of the SVC is {total_acc:.3f}')

misclass = total_wrong / (total_right + total_wrong)
print(f'The total misclassification rate (error) of the SVC is {misclass:.3f}\n')

# TP, FP, FN for Setosa
tp_setosa = conf_mtrx[0][0]
fp_setosa = conf_mtrx[1][0]+conf_mtrx[2][0]
fn_setosa = conf_mtrx[0][1]+conf_mtrx[0][2]
tn_setosa = conf_mtrx[1][1]+conf_mtrx[1][2]+conf_mtrx[2][1]+conf_mtrx[2][2]
precision_setosa = tp_setosa / (tp_setosa + fp_setosa)
recall_setosa = tp_setosa / (tp_setosa + fn_setosa)
specificity_setosa = tn_setosa / (tn_setosa + fp_setosa)
f1_setosa = (2 * precision_setosa * recall_setosa) / (precision_setosa + recall_setosa)

# TP, FP, FN for Versicolor
tp_versicolor = conf_mtrx[1][1]
fp_versicolor = conf_mtrx[0][1]+conf_mtrx[2][1]
fn_versicolor = conf_mtrx[1][0]+conf_mtrx[1][2]
tn_versicolor = conf_mtrx[0][0]+conf_mtrx[0][2]+conf_mtrx[2][0]+conf_mtrx[2][2]
precision_versicolor = tp_versicolor / (tp_versicolor + fp_versicolor)
recall_versicolor = tp_versicolor / (tp_versicolor + fn_versicolor)
specificity_versicolor = tn_versicolor / (tn_versicolor + fp_versicolor)
f1_versicolor = (2 * precision_versicolor * recall_versicolor) / (precision_versicolor + recall_versicolor)

# TP, FP, FN for Virginica
tp_virginica = conf_mtrx[2][2]
fp_virginica = conf_mtrx[0][2]+conf_mtrx[1][2]
fn_virginica = conf_mtrx[2][0]+conf_mtrx[2][1]
tn_virginica = conf_mtrx[0][0]+conf_mtrx[0][1]+conf_mtrx[1][0]+conf_mtrx[1][1]
precision_virginica = tp_virginica / (tp_virginica + fp_virginica)
recall_virginica = tp_virginica / (tp_virginica + fn_virginica)
specificity_virginica = tn_virginica / (tn_virginica + fp_virginica)
f1_virginica = (2 * precision_virginica * recall_virginica) / (precision_virginica + recall_virginica)

print(f'The precision for Setosa is {precision_setosa:.3f}')
print(f'The recall for Setosa is {recall_setosa:.3f}')
print(f'The specificity for Setosa is {specificity_setosa:.3f}')
print(f'The F1 Score for Setosa is {f1_setosa:.3f}\n')

print(f'The precision for Versicolor is {precision_versicolor:.3f}')
print(f'The recall for Versicolor is {recall_versicolor:.3f}')
print(f'The specificity for Versicolor is {specificity_versicolor:.3f}')
print(f'The F1 Score for Versicolor is {f1_versicolor:.3f}\n')

print(f'The precision for Virgnica is {precision_virginica:.3f}')
print(f'The recall for Virgnica is {recall_virginica:.3f}')
print(f'The specificity for Virginica is {specificity_virginica:.3f}')
print(f'The F1 Score for Virgnica is {f1_virginica:.3f}\n')

"""Kernel = Linear | C = 2 | Degree = 1 | Gamma = 20"""
kern = 'linear'
c = 2
deg = 1
gam = 20
print(f'Kernal = {kern} | C = {c} | Degree = {deg} | Gamma = {gam}')
svc = SVC(kernel=kern, C = c , degree = deg , gamma = gam)
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
conf_mtrx = confusion_matrix(Y_test, Y_predict)

#Key Performance Indicators of the Model

print('Confusion Matrix and Key Performance Indicators of the SVC Model')
print(conf_mtrx)

total_right = conf_mtrx[0][0] + conf_mtrx[1][1] + conf_mtrx[2][2]
total_wrong = conf_mtrx[0][1] + conf_mtrx[0][2] + conf_mtrx[1][0] + conf_mtrx[1][2] + conf_mtrx[2][0] + conf_mtrx[2][1]
total_acc = total_right / (total_right + total_wrong)
print(f'The total accuracy of the SVC is {total_acc:.3f}')

misclass = total_wrong / (total_right + total_wrong)
print(f'The total misclassification rate (error) of the SVC is {misclass:.3f}\n')

# TP, FP, FN for Setosa
tp_setosa = conf_mtrx[0][0]
fp_setosa = conf_mtrx[1][0]+conf_mtrx[2][0]
fn_setosa = conf_mtrx[0][1]+conf_mtrx[0][2]
tn_setosa = conf_mtrx[1][1]+conf_mtrx[1][2]+conf_mtrx[2][1]+conf_mtrx[2][2]
precision_setosa = tp_setosa / (tp_setosa + fp_setosa)
recall_setosa = tp_setosa / (tp_setosa + fn_setosa)
specificity_setosa = tn_setosa / (tn_setosa + fp_setosa)
f1_setosa = (2 * precision_setosa * recall_setosa) / (precision_setosa + recall_setosa)

# TP, FP, FN for Versicolor
tp_versicolor = conf_mtrx[1][1]
fp_versicolor = conf_mtrx[0][1]+conf_mtrx[2][1]
fn_versicolor = conf_mtrx[1][0]+conf_mtrx[1][2]
tn_versicolor = conf_mtrx[0][0]+conf_mtrx[0][2]+conf_mtrx[2][0]+conf_mtrx[2][2]
precision_versicolor = tp_versicolor / (tp_versicolor + fp_versicolor)
recall_versicolor = tp_versicolor / (tp_versicolor + fn_versicolor)
specificity_versicolor = tn_versicolor / (tn_versicolor + fp_versicolor)
f1_versicolor = (2 * precision_versicolor * recall_versicolor) / (precision_versicolor + recall_versicolor)

# TP, FP, FN for Virginica
tp_virginica = conf_mtrx[2][2]
fp_virginica = conf_mtrx[0][2]+conf_mtrx[1][2]
fn_virginica = conf_mtrx[2][0]+conf_mtrx[2][1]
tn_virginica = conf_mtrx[0][0]+conf_mtrx[0][1]+conf_mtrx[1][0]+conf_mtrx[1][1]
precision_virginica = tp_virginica / (tp_virginica + fp_virginica)
recall_virginica = tp_virginica / (tp_virginica + fn_virginica)
specificity_virginica = tn_virginica / (tn_virginica + fp_virginica)
f1_virginica = (2 * precision_virginica * recall_virginica) / (precision_virginica + recall_virginica)

print(f'The precision for Setosa is {precision_setosa:.3f}')
print(f'The recall for Setosa is {recall_setosa:.3f}')
print(f'The specificity for Setosa is {specificity_setosa:.3f}')
print(f'The F1 Score for Setosa is {f1_setosa:.3f}\n')

print(f'The precision for Versicolor is {precision_versicolor:.3f}')
print(f'The recall for Versicolor is {recall_versicolor:.3f}')
print(f'The specificity for Versicolor is {specificity_versicolor:.3f}')
print(f'The F1 Score for Versicolor is {f1_versicolor:.3f}\n')

print(f'The precision for Virgnica is {precision_virginica:.3f}')
print(f'The recall for Virgnica is {recall_virginica:.3f}')
print(f'The specificity for Virginica is {specificity_virginica:.3f}')
print(f'The F1 Score for Virgnica is {f1_virginica:.3f}\n')

"""Kernel = Sigmoid | C = 5 | Degree = 3 | Gamma = 1"""
kern = 'sigmoid'
c = .1
deg = 3
gam = .5
print(f'Kernal = {kern} | C = {c} | Degree = {deg} | Gamma = {gam}')
svc = SVC(kernel=kern, C = c , degree = deg , gamma = gam)
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
conf_mtrx = confusion_matrix(Y_test, Y_predict)

#Key Performance Indicators of the Model

print('Confusion Matrix and Key Performance Indicators of the SVC Model')
print(conf_mtrx)

total_right = conf_mtrx[0][0] + conf_mtrx[1][1] + conf_mtrx[2][2]
total_wrong = conf_mtrx[0][1] + conf_mtrx[0][2] + conf_mtrx[1][0] + conf_mtrx[1][2] + conf_mtrx[2][0] + conf_mtrx[2][1]
total_acc = total_right / (total_right + total_wrong)
print(f'The total accuracy of the SVC is {total_acc:.3f}')

misclass = total_wrong / (total_right + total_wrong)
print(f'The total misclassification rate (error) of the SVC is {misclass:.3f}\n')

# TP, FP, FN for Setosa
tp_setosa = conf_mtrx[0][0]
fp_setosa = conf_mtrx[1][0]+conf_mtrx[2][0]
fn_setosa = conf_mtrx[0][1]+conf_mtrx[0][2]
tn_setosa = conf_mtrx[1][1]+conf_mtrx[1][2]+conf_mtrx[2][1]+conf_mtrx[2][2]
precision_setosa = tp_setosa / (tp_setosa + fp_setosa)
recall_setosa = tp_setosa / (tp_setosa + fn_setosa)
specificity_setosa = tn_setosa / (tn_setosa + fp_setosa)
f1_setosa = (2 * precision_setosa * recall_setosa) / (precision_setosa + recall_setosa)

# TP, FP, FN for Versicolor
tp_versicolor = conf_mtrx[1][1]
fp_versicolor = conf_mtrx[0][1]+conf_mtrx[2][1]
fn_versicolor = conf_mtrx[1][0]+conf_mtrx[1][2]
tn_versicolor = conf_mtrx[0][0]+conf_mtrx[0][2]+conf_mtrx[2][0]+conf_mtrx[2][2]
precision_versicolor = tp_versicolor / (tp_versicolor + fp_versicolor)
recall_versicolor = tp_versicolor / (tp_versicolor + fn_versicolor)
specificity_versicolor = tn_versicolor / (tn_versicolor + fp_versicolor)
f1_versicolor = (2 * precision_versicolor * recall_versicolor) / (precision_versicolor + recall_versicolor)

# TP, FP, FN for Virginica
tp_virginica = conf_mtrx[2][2]
fp_virginica = conf_mtrx[0][2]+conf_mtrx[1][2]
fn_virginica = conf_mtrx[2][0]+conf_mtrx[2][1]
tn_virginica = conf_mtrx[0][0]+conf_mtrx[0][1]+conf_mtrx[1][0]+conf_mtrx[1][1]
precision_virginica = tp_virginica / (tp_virginica + fp_virginica)
recall_virginica = tp_virginica / (tp_virginica + fn_virginica)
specificity_virginica = tn_virginica / (tn_virginica + fp_virginica)
f1_virginica = (2 * precision_virginica * recall_virginica) / (precision_virginica + recall_virginica)

print(f'The precision for Setosa is {precision_setosa:.3f}')
print(f'The recall for Setosa is {recall_setosa:.3f}')
print(f'The specificity for Setosa is {specificity_setosa:.3f}')
print(f'The F1 Score for Setosa is {f1_setosa:.3f}\n')

print(f'The precision for Versicolor is {precision_versicolor:.3f}')
print(f'The recall for Versicolor is {recall_versicolor:.3f}')
print(f'The specificity for Versicolor is {specificity_versicolor:.3f}')
print(f'The F1 Score for Versicolor is {f1_versicolor:.3f}\n')

print(f'The precision for Virgnica is {precision_virginica:.3f}')
print(f'The recall for Virgnica is {recall_virginica:.3f}')
print(f'The specificity for Virginica is {specificity_virginica:.3f}')
print(f'The F1 Score for Virgnica is {f1_virginica:.3f}\n')

"""Kernel = Sigmoid | C = 5 | Degree = 3 | Gamma = 1"""
kern = 'sigmoid'
c = 1
deg = 5
gam = 3
print(f'Kernal = {kern} | C = {c} | Degree = {deg} | Gamma = {gam}')
svc = SVC(kernel=kern, C = c , degree = deg , gamma = gam)
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
conf_mtrx = confusion_matrix(Y_test, Y_predict)

#Key Performance Indicators of the Model

print('Confusion Matrix and Key Performance Indicators of the SVC Model')
print(conf_mtrx)

total_right = conf_mtrx[0][0] + conf_mtrx[1][1] + conf_mtrx[2][2]
total_wrong = conf_mtrx[0][1] + conf_mtrx[0][2] + conf_mtrx[1][0] + conf_mtrx[1][2] + conf_mtrx[2][0] + conf_mtrx[2][1]
total_acc = total_right / (total_right + total_wrong)
print(f'The total accuracy of the SVC is {total_acc:.3f}')

misclass = total_wrong / (total_right + total_wrong)
print(f'The total misclassification rate (error) of the SVC is {misclass:.3f}\n')

# TP, FP, FN for Setosa
tp_setosa = conf_mtrx[0][0]
fp_setosa = conf_mtrx[1][0]+conf_mtrx[2][0]
fn_setosa = conf_mtrx[0][1]+conf_mtrx[0][2]
tn_setosa = conf_mtrx[1][1]+conf_mtrx[1][2]+conf_mtrx[2][1]+conf_mtrx[2][2]
precision_setosa = tp_setosa / (tp_setosa + fp_setosa)
recall_setosa = tp_setosa / (tp_setosa + fn_setosa)
specificity_setosa = tn_setosa / (tn_setosa + fp_setosa)
f1_setosa = (2 * precision_setosa * recall_setosa) / (precision_setosa + recall_setosa)

# TP, FP, FN for Versicolor
tp_versicolor = conf_mtrx[1][1]
fp_versicolor = conf_mtrx[0][1]+conf_mtrx[2][1]
fn_versicolor = conf_mtrx[1][0]+conf_mtrx[1][2]
tn_versicolor = conf_mtrx[0][0]+conf_mtrx[0][2]+conf_mtrx[2][0]+conf_mtrx[2][2]
precision_versicolor = tp_versicolor / (tp_versicolor + fp_versicolor)
recall_versicolor = tp_versicolor / (tp_versicolor + fn_versicolor)
specificity_versicolor = tn_versicolor / (tn_versicolor + fp_versicolor)
f1_versicolor = (2 * precision_versicolor * recall_versicolor) / (precision_versicolor + recall_versicolor)

# TP, FP, FN for Virginica
tp_virginica = conf_mtrx[2][2]
fp_virginica = conf_mtrx[0][2]+conf_mtrx[1][2]
fn_virginica = conf_mtrx[2][0]+conf_mtrx[2][1]
tn_virginica = conf_mtrx[0][0]+conf_mtrx[0][1]+conf_mtrx[1][0]+conf_mtrx[1][1]
precision_virginica = tp_virginica / (tp_virginica + fp_virginica)
recall_virginica = tp_virginica / (tp_virginica + fn_virginica)
specificity_virginica = tn_virginica / (tn_virginica + fp_virginica)
f1_virginica = (2 * precision_virginica * recall_virginica) / (precision_virginica + recall_virginica)

print(f'The precision for Setosa is {precision_setosa:.3f}')
print(f'The recall for Setosa is {recall_setosa:.3f}')
print(f'The specificity for Setosa is {specificity_setosa:.3f}')
print(f'The F1 Score for Setosa is {f1_setosa:.3f}\n')

print(f'The precision for Versicolor is {precision_versicolor:.3f}')
print(f'The recall for Versicolor is {recall_versicolor:.3f}')
print(f'The specificity for Versicolor is {specificity_versicolor:.3f}')
print(f'The F1 Score for Versicolor is {f1_versicolor:.3f}\n')

print(f'The precision for Virgnica is {precision_virginica:.3f}')
print(f'The recall for Virgnica is {recall_virginica:.3f}')
print(f'The specificity for Virginica is {specificity_virginica:.3f}')
print(f'The F1 Score for Virgnica is {f1_virginica:.3f}\n')


