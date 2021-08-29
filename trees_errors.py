from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import tree

data = pd.read_csv(r"C:\Users\Agata\Desktop\MAGISTERKA\Bruegel\full_data.csv")
test_data = pd.read_csv(r"C:\Users\Agata\Desktop\MAGISTERKA\Bruegel\test_data.csv")

train_data = pd.read_csv("train_set.csv", index_col = 0)
validation_data = pd.read_csv("validation_set.csv", index_col = 0 )
# usunięcie NA
data = data[data['dist_sd'].notna()]
test_data = test_data[test_data['dist_sd'].notna()]
data = data[data['dist_mean'].notna()]
test_data = test_data[test_data['dist_mean'].notna()]
data = data.reset_index()
test_data = test_data.reset_index()
print(data.shape[0])
print(test_data.shape[0])

# model 0: brak modelu
# false positives
fp = sum((test_data['norm_fro'].notna()) & (test_data['class']=='false'))
fp_ratio = fp/sum(test_data['class']=='false')

# false negatives
fn = sum((test_data['norm_fro'].isna()) & (test_data['classification']==1))
fn_ratio = fn/sum(test_data['classification']==1)

# false negatives: identity
fn_id = sum((test_data['norm_fro'].isna()) & (test_data['class']=='identity'))
fn_id_ratio = fn/sum(test_data['class']=='identity')

# false negatives: reproductions
fn_rep = sum((test_data['norm_fro'].isna()) & (test_data['class']=='reproductions'))
fn_rep_ratio = fn/sum(test_data['class']=='reproductions')

# ogółem
err = (fp+fn)/test_data.shape[0]

print(f"Model 0: FP: {fp}[{fp_ratio*100}%], FN: {fn}[{fn_ratio*100}%], ogółem {fp+fn}[{err*100}%]")
print(f"Model 0: false: {fp}[{fp_ratio*100}%], identity: {fn_id}[{fn_id_ratio*100}%], reproductions:{fn_rep}[{fn_rep_ratio*100}%], ogółem {fp+fn}[{err*100}%]")

# Model 1: drzewo decyzyjne na bazie samego sd
X = data.loc[:,['dist_sd']].to_numpy()
Y = data.loc[:,'classification'].to_numpy()
X_test = test_data.loc[:,['dist_sd']].to_numpy()
Y_test = test_data.loc[:,'classification'].to_numpy()
X = StandardScaler().fit_transform(X)
X_test = StandardScaler().fit_transform(X_test)
clf = tree.DecisionTreeClassifier(max_depth = 2)
clf = clf.fit(X, Y)
y_pred = clf.predict(X_test)

# false positives
false_indices = test_data[test_data['class']=='false'].index
fp = (Y_test[false_indices] != y_pred[false_indices]).sum()
fp_ratio = fp/(X_test[false_indices,:].shape[0])

# false negatives
pos_indices = test_data[test_data['classification']==1].index
fn =  (Y_test[pos_indices] != y_pred[pos_indices]).sum()
fn_ratio = fn/(X_test[pos_indices,:].shape[0])

# false negatives: identity
id_indices = test_data[test_data['class']=='identity'].index
fn_id =  (Y_test[id_indices] != y_pred[id_indices]).sum()
fn_id_ratio = fn_id/(X_test[id_indices,:].shape[0])

# false negatives: reproductions
rep_indices = test_data[test_data['class']=='reproductions'].index
fn_rep = (Y_test[rep_indices] != y_pred[rep_indices]).sum()
fn_rep_ratio = fn_rep/(X_test[rep_indices,:].shape[0])

# ogółem
err = (fp+fn)/X_test.shape[0]

print(f"Model 1: FP: {fp}[{fp_ratio*100}%], FN: {fn}[{fn_ratio*100}%], ogółem {fp+fn}[{err*100}%]")
print(f"Model 1: false: {fp}[{fp_ratio*100}%], identity: {fn_id}[{fn_id_ratio*100}%], reproductions:{fn_rep}[{fn_rep_ratio*100}%], ogółem {fp+fn}[{err*100}%]")

# Model 2: odchylenie + średnia
X = data.loc[:,['dist_sd', 'dist_mean']].to_numpy()
Y = data.loc[:,'classification'].to_numpy()
X_test = test_data.loc[:,['dist_sd','dist_mean']].to_numpy()
Y_test = test_data.loc[:,'classification'].to_numpy()
X = StandardScaler().fit_transform(X)
X_test = StandardScaler().fit_transform(X_test)
clf = tree.DecisionTreeClassifier(max_depth = 5)
clf = clf.fit(X, Y)
y_pred = clf.predict(X_test)

# false positives
false_indices = test_data[test_data['class']=='false'].index
fp = (Y_test[false_indices] != y_pred[false_indices]).sum()
fp_ratio = fp/(X_test[false_indices,:].shape[0])

# false negatives
pos_indices = test_data[test_data['classification']==1].index
fn =  (Y_test[pos_indices] != y_pred[pos_indices]).sum()
fn_ratio = fn/(X_test[pos_indices,:].shape[0])

# false negatives: identity
id_indices = test_data[test_data['class']=='identity'].index
fn_id =  (Y_test[id_indices] != y_pred[id_indices]).sum()
fn_id_ratio = fn_id/(X_test[id_indices,:].shape[0])

# false negatives: reproductions
rep_indices = test_data[test_data['class']=='reproductions'].index
fn_rep = (Y_test[rep_indices] != y_pred[rep_indices]).sum()
fn_rep_ratio = fn_rep/(X_test[rep_indices,:].shape[0])

# ogółem
err = (fp+fn)/X_test.shape[0]

print(f"Model 2: FP: {fp}[{fp_ratio*100}%], FN: {fn}[{fn_ratio*100}%], ogółem {fp+fn}[{err*100}%]")
print(f"Model 2: false: {fp}[{fp_ratio*100}%], identity: {fn_id}[{fn_id_ratio*100}%], reproductions:{fn_rep}[{fn_rep_ratio*100}%], ogółem {fp+fn}[{err*100}%]")

# Model 3: sd + liczba dopasowań
X = data.loc[:,['dist_sd', 'no_of_matches']].to_numpy()
Y = data.loc[:,'classification'].to_numpy()
X_test = test_data.loc[:,['dist_sd','no_of_matches']].to_numpy()
Y_test = test_data.loc[:,'classification'].to_numpy()
X = StandardScaler().fit_transform(X)
X_test = StandardScaler().fit_transform(X_test)
clf = tree.DecisionTreeClassifier(max_depth = 5)
clf = clf.fit(X, Y)
y_pred = clf.predict(X_test)

# false positives
false_indices = test_data[test_data['class']=='false'].index
fp = (Y_test[false_indices] != y_pred[false_indices]).sum()
fp_ratio = fp/(X_test[false_indices,:].shape[0])

# false negatives
pos_indices = test_data[test_data['classification']==1].index
fn =  (Y_test[pos_indices] != y_pred[pos_indices]).sum()
fn_ratio = fn/(X_test[pos_indices,:].shape[0])

# false negatives: identity
id_indices = test_data[test_data['class']=='identity'].index
fn_id =  (Y_test[id_indices] != y_pred[id_indices]).sum()
fn_id_ratio = fn_id/(X_test[id_indices,:].shape[0])

# false negatives: reproductions
rep_indices = test_data[test_data['class']=='reproductions'].index
fn_rep = (Y_test[rep_indices] != y_pred[rep_indices]).sum()
fn_rep_ratio = fn_rep/(X_test[rep_indices,:].shape[0])

# ogółem
err = (fp+fn)/X_test.shape[0]

print(f"Model 3: FP: {fp}[{fp_ratio*100}%], FN: {fn}[{fn_ratio*100}%], ogółem {fp+fn}[{err*100}%]")
print(f"Model 3: false: {fp}[{fp_ratio*100}%], identity: {fn_id}[{fn_id_ratio*100}%], reproductions:{fn_rep}[{fn_rep_ratio*100}%], ogółem {fp+fn}[{err*100}%]")

# Model 4: sd, średnia, liczba dopasowan,
X = data.loc[:,['dist_sd','dist_mean', 'no_of_matches']].to_numpy()
Y = data.loc[:,'classification'].to_numpy()
X_test = test_data.loc[:,['dist_sd','dist_mean','no_of_matches']].to_numpy()
Y_test = test_data.loc[:,'classification'].to_numpy()
X = StandardScaler().fit_transform(X)
X_test = StandardScaler().fit_transform(X_test)
clf = tree.DecisionTreeClassifier(max_depth = 5)
clf = clf.fit(X, Y)
y_pred = clf.predict(X_test)

# false positives
false_indices = test_data[test_data['class']=='false'].index
fp = (Y_test[false_indices] != y_pred[false_indices]).sum()
fp_ratio = fp/(X_test[false_indices,:].shape[0])

# false negatives
pos_indices = test_data[test_data['classification']==1].index
fn =  (Y_test[pos_indices] != y_pred[pos_indices]).sum()
fn_ratio = fn/(X_test[pos_indices,:].shape[0])

# false negatives: identity
id_indices = test_data[test_data['class']=='identity'].index
fn_id =  (Y_test[id_indices] != y_pred[id_indices]).sum()
fn_id_ratio = fn_id/(X_test[id_indices,:].shape[0])

# false negatives: reproductions
rep_indices = test_data[test_data['class']=='reproductions'].index
fn_rep = (Y_test[rep_indices] != y_pred[rep_indices]).sum()
fn_rep_ratio = fn_rep/(X_test[rep_indices,:].shape[0])

# ogółem
err = (fp+fn)/X_test.shape[0]

print(f"Model 4: FP: {fp}[{fp_ratio*100}%], FN: {fn}[{fn_ratio*100}%], ogółem {fp+fn}[{err*100}%]")
print(f"Model 4: false: {fp}[{fp_ratio*100}%], identity: {fn_id}[{fn_id_ratio*100}%], reproductions:{fn_rep}[{fn_rep_ratio*100}%], ogółem {fp+fn}[{err*100}%]")


# Model 5: sd, średnia, liczba dopasowan, (random forest)
X = data.loc[:,['dist_sd','dist_mean', 'no_of_matches']].to_numpy()
Y = data.loc[:,'classification'].to_numpy()
X_test = test_data.loc[:,['dist_sd','dist_mean','no_of_matches']].to_numpy()
Y_test = test_data.loc[:,'classification'].to_numpy()
X = StandardScaler().fit_transform(X)
X_test = StandardScaler().fit_transform(X_test)
clf = RandomForestClassifier(n_estimators=200,max_depth=5)
clf = clf.fit(X, Y)
y_pred = clf.predict(X_test)

# false positives
false_indices = test_data[test_data['class']=='false'].index
fp = (Y_test[false_indices] != y_pred[false_indices]).sum()
fp_ratio = fp/(X_test[false_indices,:].shape[0])

# false negatives
pos_indices = test_data[test_data['classification']==1].index
fn =  (Y_test[pos_indices] != y_pred[pos_indices]).sum()
fn_ratio = fn/(X_test[pos_indices,:].shape[0])

# false negatives: identity
id_indices = test_data[test_data['class']=='identity'].index
fn_id =  (Y_test[id_indices] != y_pred[id_indices]).sum()
fn_id_ratio = fn_id/(X_test[id_indices,:].shape[0])

# false negatives: reproductions
rep_indices = test_data[test_data['class']=='reproductions'].index
fn_rep = (Y_test[rep_indices] != y_pred[rep_indices]).sum()
fn_rep_ratio = fn_rep/(X_test[rep_indices,:].shape[0])

# ogółem
err = (fp+fn)/X_test.shape[0]

print(f"Model 5: FP: {fp}[{fp_ratio*100}%], FN: {fn}[{fn_ratio*100}%], ogółem {fp+fn}[{err*100}%]")
print(f"Model 5: false: {fp}[{fp_ratio*100}%], identity: {fn_id}[{fn_id_ratio*100}%], reproductions:{fn_rep}[{fn_rep_ratio*100}%], ogółem {fp+fn}[{err*100}%]")


