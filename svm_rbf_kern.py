import arff
import numpy as np
import pandas as pd
import itertools as it
from sklearn.svm import SVC

data = arff.load(open('chronic_kidney_disease_full.arff', 'r'))
data = data['data']
data = np.array(data)
data1 = pd.DataFrame(data)
data1.columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
                 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']

data1['age'].fillna(data1['age'].mean(), inplace=True)
data1['bp'].fillna(data1['bp'].mean(), inplace=True)
data1.sg = data1.sg.map({'1.005': 1.005, '1.010': 1.010, '1.015': 1.015, '1.020': 1.020, '1.025': 1.025})
data1['sg'].fillna(1.015, inplace=True)
data1.al = data1.al.map({'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5})
data1['al'].fillna(2, inplace=True)
data1.su = data1.su.map({'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5})
data1['su'].fillna(2, inplace=True)
data1.rbc = data1.rbc.map({'normal': 0, 'abnormal': 1})
data1['rbc'].fillna(0, inplace=True)
data1.pc = data1.pc.map({'normal': 0, 'abnormal': 1})
data1['pc'].fillna(0, inplace=True)
data1.pcc = data1.pcc.map({'present': 0, 'notpresent': 1})
data1['pcc'].fillna(1, inplace=True)
data1.ba = data1.ba.map({'present': 0, 'notpresent': 1})
data1['ba'].fillna(1, inplace=True)
data1['bgr'].fillna(data1['bgr'].mean(), inplace=True)
data1['bu'].fillna(data1['bu'].mean(), inplace=True)
data1['sc'].fillna(data1['sc'].mean(), inplace=True)
data1['sod'].fillna(data1['sod'].mean(), inplace=True)
data1['pot'].fillna(data1['pot'].mean(), inplace=True)
data1['hemo'].fillna(data1['hemo'].mean(), inplace=True)
data1['pcv'].fillna(data1['pcv'].mean(), inplace=True)
data1['wbcc'].fillna(data1['wbcc'].mean(), inplace=True)
data1['rbcc'].fillna(data1['rbcc'].mean(), inplace=True)
data1.htn = data1.htn.map({'yes': 0, 'no': 1})
data1['htn'].fillna(1, inplace=True)
data1.dm = data1.dm.map({'yes': 0, 'no': 1})
data1['dm'].fillna(1, inplace=True)
data1.cad = data1.cad.map({'yes': 0, 'no': 1})
data1['cad'].fillna(1, inplace=True)
data1.appet = data1.appet.map({'good': 0, 'poor': 1})
data1['appet'].fillna(0, inplace=True)
data1.pe = data1.pe.map({'yes': 0, 'no': 1})
data1['pe'].fillna(1, inplace=True)
data1.ane = data1.ane.map({'yes': 0, 'no': 1})
data1['ane'].fillna(1, inplace=True)
data1['class'] = data1['class'].map({'ckd': 0, 'notckd': 1})
data1['class'].fillna(1, inplace=True)

cols_to_norm = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
data1[cols_to_norm] = data1[cols_to_norm].apply(lambda x: (x - x.mean()) / x.std())

train = data1.sample(frac=0.8, random_state=1)
test = data1.drop(train.index)

X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values


clf_train = SVC(gamma='auto', kernel='linear')
clf_train.fit(X_train, y_train)

p_train = clf_train.predict(X_train)

TP_train, TN_train, FP_train, FN_train = 0.0, 0.0, 0.0, 0.0

for pin, yin in it.izip(p_train, y_train):
    if pin == 0 and yin == 0:
        TP_train = TP_train + 1
    elif pin == 1 and yin == 1:
        TN_train = TN_train + 1
    elif pin == 0 and yin == 1:
        FP_train = FP_train + 1
    elif pin == 1 and yin == 0:
        FN_train = FN_train + 1

pre_train = TP_train / (TP_train + FP_train)
rec_train = TP_train / (TP_train + FN_train)

f_mes_train = (2 * pre_train * rec_train) / (pre_train + rec_train)

print f_mes_train

clf_test = SVC(gamma='auto', kernel='rbf')
clf_test.fit(X_test, y_test)

p_test = clf_test.predict(X_test)

TP_test, TN_test, FP_test, FN_test = 0.0, 0.0, 0.0, 0.0

for pin, yin in it.izip(p_test, y_test):
    if pin == 0 and yin == 0:
        TP_test = TP_test + 1
    elif pin == 1 and yin == 1:
        TN_test = TN_test + 1
    elif pin == 0 and yin == 1:
        FP_test = FP_test + 1
    elif pin == 1 and yin == 0:
        FN_test = FN_test + 1

pre_test = TP_test / (TP_test + FP_test)
rec_test = TP_test / (TP_test + FN_test)

f_mes_test = (2 * pre_test * rec_test) / (pre_test + rec_test)

print f_mes_test