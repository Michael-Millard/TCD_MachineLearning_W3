# -------------------------------------------- #

# CS7CS4 Machine Learning
# Week 3 Assignment
#
# Name:         Michael Millard
# Student ID:   24364218
# Due date:     12/10/2024
# Dataset ID:   # id:15-15--15

# -------------------------------------------- #
# Imports
# -------------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score

# -------------------------------------------- #
# Read in data and set labels
# -------------------------------------------- #

# NB: dataset csv file must be in same directory as this solution
labels = ["X1", "X2", "y"]
df = pd.read_csv("millardm_W3_dataset.csv", names=labels)
print("Dataframe head:")
print(df.head())

# Split data frame up into X and y 
X1 = df["X1"].to_numpy()
X2 = df["X2"].to_numpy()
X = np.column_stack((X1, X2))
y = df["y"].to_numpy()

# -------------------------------------------- #
# Question (i)(a)
# -------------------------------------------- #

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, y)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.legend(['Training data'])

# Saving done manually
plt.show()

# -------------------------------------------- #
# Question (i)(b)
# -------------------------------------------- #

print("\nLasso:")

# Adding extra polynomial features (combos of all powers up to 5)
X_poly = PolynomialFeatures(5).fit_transform(X)

# Lasso model param arrays
lasso_models = []

# Range of C values to sweep
theta0s = []
thetas1to21 = []
C_range = [1, 5, 10, 100, 1000]
for C in C_range:
    # Create Lasso model and fit training data
    lasso_model = Lasso(alpha=1/(2*C)).fit(X_poly, y)
    lasso_models.append(lasso_model)
    
    # Extract params and print them out
    theta0 = lasso_model.intercept_.item()
    thetas = lasso_model.coef_.T
    theta0s.append(theta0)
    
    if (len(thetas1to21) == 0):
        thetas1to21 = thetas
    else:
        thetas1to21 = np.column_stack((thetas1to21, thetas))
    result_str = "C, theta0"
    for i in range(len(thetas)):
        result_str += ", theta{}".format(i + 1)
    result_str += " = {c} & {theta:.3f}".format(c=C, theta=theta0)
    for i in range(len(thetas)):
        result_str += " & {theta:.3f}".format(theta=thetas[i].item())
    print(result_str)
    
print(np.shape(theta0s))
print(np.shape(thetas1to21))

# These print statements are used to make Latex table creation easy
print("")
result_str = "C = "
for i in range(len(C_range)):
    result_str += " & {C}".format(C=C_range[i])
print(result_str)

result_str = "theta0 = "
for i in range(len(C_range)):
    result_str += " & {theta:.4f}".format(theta=theta0s[i])
print(result_str)

for i in range(np.shape(thetas1to21)[0]):
    result_str = "theta{} = ".format(i + 1)
    for j in range(np.shape(thetas1to21)[1]):
        result_str += " & {theta:.4f}".format(theta=thetas1to21[i][j])
    print(result_str)
    
# -------------------------------------------- #
# Question (i)(c)
# -------------------------------------------- #

# Creating grid of features extending beyond current dataset, need to do this for each feature

# Find min and max values for X1 and X2
min_X1, max_X1 = np.min(X1), np.max(X1)
min_X2, max_X2 = np.min(X2), np.max(X2)

# Expand their ranges by 1 unit beyond either extreme and create a new column of values for each
num_pts = 100
X1_test = np.linspace(min_X1 - 1, max_X1 + 1, num_pts).reshape(-1, 1) 
X2_test = np.linspace(min_X2 - 1, max_X2 + 1, num_pts).reshape(-1, 1)

# Make grid from test values
X_test = []
for i in X1_test:
    for j in X2_test:
        X_test.append([i, j])
X_test = np.array(X_test).reshape(-1, 2)
X1_test, X2_test = X_test[:, 0], X_test[:, 1]

# Create polynomial features for X_test
X_test_poly = PolynomialFeatures(5).fit_transform(X_test)

# Try first model
for i in range(len(lasso_models)):
    y_pred = lasso_models[i].predict(X_test_poly)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1_test.reshape(num_pts, num_pts), X2_test.reshape(num_pts, num_pts), y_pred.reshape(num_pts, num_pts), alpha=0.6, cmap='coolwarm', linewidth=0, antialiased=False)
    ax.scatter(X1, X2, y)
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.legend(["y_pred", "y_train"])
    
    # Saving done manually
    plt.show()

# -------------------------------------------- #
# Question (i)(d)
# -------------------------------------------- #

# Written answer

# -------------------------------------------- #
# Question (i)(e)
# -------------------------------------------- #

print("\nRidge:")

# Ridge model param arrays
ridge_models = []

# Sweep through same C range as Lasso
theta0s = []
thetas1to21 = []
for C in C_range:
    # Create Ridge model and fit training data
    ridge_model = Ridge(alpha=1/(2*C)).fit(X_poly, y)
    ridge_models.append(ridge_model)
    
    # Extract params and print them out
    theta0 = ridge_model.intercept_.item()
    thetas = ridge_model.coef_.T
    theta0s.append(theta0)
    
    if (len(thetas1to21) == 0):
        thetas1to21 = thetas
    else:
        thetas1to21 = np.column_stack((thetas1to21, thetas))
    result_str = "C, theta0"
    for i in range(len(thetas)):
        result_str += ", theta{}".format(i + 1)
    result_str += " = {c} & {theta:.3f}".format(c=C, theta=theta0)
    for i in range(len(thetas)):
        result_str += " & {theta:.3f}".format(theta=thetas[i].item())
    print(result_str)
    
print(np.shape(theta0s))
print(np.shape(thetas1to21))

# For easy Latex table populating
print("")
result_str = "C = "
for i in range(len(C_range)):
    result_str += " & {C}".format(C=C_range[i])
print(result_str)

result_str = "theta0 = "
for i in range(len(C_range)):
    result_str += " & {theta:.4f}".format(theta=theta0s[i])
print(result_str)

for i in range(np.shape(thetas1to21)[0]):
    result_str = "theta{} = ".format(i + 1)
    for j in range(np.shape(thetas1to21)[1]):
        result_str += " & {theta:.4f}".format(theta=thetas1to21[i][j])
    print(result_str)

# Use same grids defined above (independent of model type)

# Try first model
for i in range(len(ridge_models)):
    y_pred = ridge_models[i].predict(X_test_poly)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1_test.reshape(num_pts, num_pts), X2_test.reshape(num_pts, num_pts), y_pred.reshape(num_pts, num_pts), alpha=0.6, cmap='coolwarm', linewidth=0, antialiased=False)
    ax.scatter(X1, X2, y)
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.legend(["y_pred", "y_train"])
    
    # Saving done manually
    plt.show()

# -------------------------------------------- #
# Question (ii)(a)
# -------------------------------------------- #

# Test arrays 
mean_error_test=[]
std_error_test=[]

# Train arrays
mean_error_train=[]
std_error_train=[]

# Smaller range for C taken here
C_range = [1, 2, 4, 6, 8, 10, 16]
for C in C_range:
    # Create Lasso model
    model = Lasso(alpha=1/(2*C))
    
    # Temp arrays for storing MSE in each fold below
    temp_test=[]
    temp_train=[]
    
    # K-fold cross validation with 5 splits
    k_fold = KFold(n_splits=5)
    for train, test in k_fold.split(X):
        # Fit model to training data for this fold 
        model.fit(X[train], y[train])
        
        # Validate on test data for this fold
        y_pred_test = model.predict(X[test])
        temp_test.append(mean_squared_error(y[test], y_pred_test))
        
        # Performance on training data for this fold
        y_pred_train = model.predict(X[train])
        temp_train.append(mean_squared_error(y[train], y_pred_train))
        
    # Append errors to their respective arrays
    mean_error_test.append(np.array(temp_test).mean())
    std_error_test.append(np.array(temp_test).std())
    mean_error_train.append(np.array(temp_train).mean())
    std_error_train.append(np.array(temp_train).std())
    
# Generate error bar plot
plt.errorbar(C_range, mean_error_test, yerr=std_error_test)
plt.errorbar(C_range, mean_error_train, yerr=std_error_train, linewidth=3)
plt.xlabel('C') 
plt.ylabel('Mean square error')
plt.legend(['Test data', 'Training data'])
plt.savefig("error_bar_lasso_ii_a.png")
plt.show()

# These print statements are used to make Latex table creation easy
print("")
result_str = "C = "
for i in range(len(C_range)):
    result_str += " & {C}".format(C=C_range[i])
print(result_str)

result_str = "mean_error = "
for i in range(len(C_range)):
    result_str += " & {err:.4f}".format(err=mean_error_test[i])
print(result_str)

result_str = "std_error = "
for i in range(len(C_range)):
    result_str += " & {std:.4f}".format(std=std_error_test[i])
print(result_str)

# -------------------------------------------- #
# Question (ii)(b)
# -------------------------------------------- #

# Written answer

# -------------------------------------------- #
# Question (ii)(c)
# -------------------------------------------- #

# Test arrays 
mean_error_test=[]
std_error_test=[]

# Train arrays
mean_error_train=[]
std_error_train=[]

# Need to change range of C values again for Ridge
C_range = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 1]
for C in C_range:
    # Create Ridge model
    model = Ridge(alpha=1/(2*C))
    
    # Temp arrays for storing MSE in each fold below
    temp_test=[]
    temp_train=[]
    
    # K-fold cross validation with 5 splits
    k_fold = KFold(n_splits=5)
    for train, test in k_fold.split(X):
        # Fit model to training data for this fold 
        model.fit(X[train], y[train])
        
        # Validate on test data for this fold
        y_pred_test = model.predict(X[test])
        temp_test.append(mean_squared_error(y[test], y_pred_test))
        
        # Performance on training data for this fold
        y_pred_train = model.predict(X[train])
        temp_train.append(mean_squared_error(y[train], y_pred_train))
        
    # Append errors to their respective arrays
    mean_error_test.append(np.array(temp_test).mean())
    std_error_test.append(np.array(temp_test).std())
    mean_error_train.append(np.array(temp_train).mean())
    std_error_train.append(np.array(temp_train).std())
    
# Generate error bar plot
plt.errorbar(C_range, mean_error_test, yerr=std_error_test)
plt.errorbar(C_range, mean_error_train, yerr=std_error_train, linewidth=3)
plt.xlabel('C') 
plt.ylabel('Mean square error')
plt.legend(['Test data', 'Training data'])
plt.savefig("error_bar_ridge_ii_c.png")
plt.show()

# These print statements are used to make Latex table creation easy
print("")
result_str = "C = "
for i in range(len(C_range)):
    result_str += " & {C}".format(C=C_range[i])
print(result_str)

result_str = "mean_error = "
for i in range(len(C_range)):
    result_str += " & {err:.4f}".format(err=mean_error_test[i])
print(result_str)

result_str = "std_error = "
for i in range(len(C_range)):
    result_str += " & {std:.4f}".format(std=std_error_test[i])
print(result_str)

# -------------------------------------------- #
# END OF ASSIGNMENT
# -------------------------------------------- #