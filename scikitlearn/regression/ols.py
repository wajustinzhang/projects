from sklearn import linear_model, datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:, np.newaxis, 2]

diabetes_x_train = diabetes_X[:-20]
diabetes_x_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = Pipeline([('poly', PolynomialFeatures(degree=2)),
     ('linear', linear_model.LinearRegression(fit_intercept=False))])
# regr = linear_model.LinearRegression()
regr.fit(diabetes_x_train, diabetes_y_train)

#print('Coefficient is {}'.format(regr.coef_))
print('Coefficient is {}'.format(regr.named_steps))

print('MSE: %2f'%np.mean(regr.predict(diabetes_x_test) - diabetes_y_test) **2)
print('Variance score: %.2f'%regr.score(diabetes_x_test, diabetes_y_test))

plt.scatter(diabetes_x_test, diabetes_y_test,  color='black')
plt.plot(diabetes_x_test, regr.predict(diabetes_x_test), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()