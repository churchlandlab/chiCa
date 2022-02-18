import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn import metrics
#%% PREPROCESSING DATA! using one single cell trace ft 1 event state

X_neuron_trace = zScoreC[0:50].transpose()
Y_event_vector = np.array(binarizeTaskEvents())
plt.plot(Y_event_vector)

#%%
for Model in [Ridge, Lasso]:
    model = Model()
    print('%s: %s' % (Model.__name__, cross_val_score(model, X_neuron_trace, Y_event_vector,cv=10).mean()))
#%% using Lasso Regression:
    
lasso_model = Lasso()
y_pred_lasso_acc=[]
b_coeff_lasso=[]
kf = KFold(n_splits=10)

for train, test in kf.split(X_neuron_trace):
    X_train, X_test, y_train, y_test = X_neuron_trace[train], X_neuron_trace[test], Y_event_vector[train], Y_event_vector[test]
    
    lasso_model.fit(X_train, y_train)
    #print(lasso_model.predict(X_test))
    #print(y_test)
    y_pred_lasso_acc.append(lasso_model.score(X_test,y_test))
    b_coeff_lasso.append(lasso_model.coef_)

print(np.mean(b_coeff_lasso,0))
print(y_pred_lasso_acc)
print(np.mean(y_pred_lasso_acc))