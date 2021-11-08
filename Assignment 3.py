# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:57:03 2021

@author: 19564
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn import random_projection
import seaborn as sns
from sklearn import tree
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture
from xgboost.sklearn import XGBClassifier
from scipy.stats import norm, kurtosis
from sklearn.preprocessing import StandardScaler
import time as time
from sklearn.metrics import homogeneity_completeness_v_measure as hcv
import statistics

#Establish a seed value
seed = 0
nums=[1,2,3,4,5,6,7,8,9,10]
#Create two random sets for classification
full_1_X, full_1_y = make_classification(n_samples=10000, n_features=10, n_informative=8, n_redundant=2, random_state=seed, class_sep=.1, shuffle=True)
full_2_X, full_2_y = make_classification(n_samples=8000, n_features=10, n_informative=2, n_redundant=1, n_clusters_per_class=2, random_state=seed, flip_y=.3, class_sep=.5, shuffle=True, weights=[0.7,0.3])

#Create a training dataset and a holdout set. The training set will be 
x_1_train, x_1_test, y_1_train, y_1_test = train_test_split(full_1_X, full_1_y, test_size=0.3, random_state=seed)
x_2_train, x_2_test, y_2_train, y_2_test = train_test_split(full_2_X, full_2_y, test_size=0.3, random_state=seed)

x_1_train_copy=x_1_train.copy()
x_1_test_copy=x_1_test.copy()
y_1_train_copy=y_1_train.copy()
y_1_test_copy=y_1_test.copy()

x_2_train_copy=x_2_train.copy()
x_2_test_copy=x_2_test.copy()
y_2_train_copy=y_2_train.copy()
y_2_test_copy=y_2_test.copy()

#Scale the data for easier processing by algorithm
standard_scaler =  StandardScaler()

fit_scaler = standard_scaler.fit(x_1_train)
x_1_train=fit_scaler.transform(x_1_train)
x_1_test=fit_scaler.transform(x_1_test)

fit_scaler = standard_scaler.fit(x_2_train)
x_2_train=fit_scaler.transform(x_2_train)
x_2_test=fit_scaler.transform(x_2_test)

#Show interesting features of dataset 1
full_1=pd.DataFrame(full_1_X)
full_1['y']=full_1_y
check_data1=full_1.describe().transpose()
full_1['y'].describe()
plt.show()

#Show interesting features of dataset 2
full_2=pd.DataFrame(full_2_X)
full_2['y']=full_2_y
#sn.heatmap(full_2.corr(), annot=True)
check_data2=full_2.describe().transpose()
full_2['y'].describe()
plt.show()


##########################################START EXPERIMENTS###################################################################################

def runKmeans(training_X, training_Y, test_X, test_Y, dataset_num):


    kmeans_classifier=KMeans(random_state=seed)
    
    #Code for testing multiple values of k for Kmeans and creating an elbow plot was borrowed from
    #Author:Najee Smith
    #Source: Retrieved from https://najeesmith.github.io/KMeans/#
    wCSS_auto = []
    for i in range(1,10):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 1000, n_init = 100, algorithm="auto", random_state=seed)
        kmeans.fit(training_X)
        wCSS_auto.append(kmeans.inertia_) #Collects all of the within cluster sum of squares
    
    plt.plot(range(1,10), wCSS_auto, color="black")
    plt.title("Elbow Method-Training Set {}".format(dataset_num))
    plt.xlabel('Clusters')
    plt.ylabel('wCSS')
    plt.show()
    
    kmeans_classifier = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 1000, n_init = 100, algorithm="auto", random_state=seed)
    kmeans_classifier.fit(training_X)
    
    y_prediction=kmeans_classifier.predict(training_X)
    y_prediction_test=kmeans_classifier.predict(test_X)
    
    number_rows, number_cols = training_X.shape
    
    #Plot the clusterings by pairs of attributes
    for j in range(0,number_cols):
        fig, ax = plt.subplots(1, 10, figsize=(24, 3))
        plt.tight_layout(rect=[0, 0.05, 1, 0.94])
        for i in range(0,number_cols):
    
            fig.suptitle("K-means Clustering - Pairwise View - Dataset {}".format(dataset_num))
            ax[i].scatter(training_X[:, j], training_X[:, i],c=y_prediction, s=5, cmap='viridis')
            ax[i].set_ylabel("Variable {}".format(i))
            ax[i].set_xlabel("Variable {}".format(j))
            centers = kmeans_classifier.cluster_centers_
            if number_cols>1:
                ax[i].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plt.show()
    
    y_1_pred=pd.DataFrame(kmeans_classifier.predict(training_X), columns=["label"]) 
    y_1_pred['label']=y_1_pred["label"].astype('category')
    
    
    x_1_full=pd.DataFrame(training_X)
    x_1_full["label"]=y_1_pred
    
    sns.pairplot(x_1_full, hue="label")
    plt.show()
    
    #Find the error on the full training dataset
    #Code to plot classification report
    #Source: Adapted from Stack Overflow example by user akilat90
    #Article "How to plot scikit learn classification report?"
    #Code modified from 
    #https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
    
    ax = plt.axes()
    clf_report = classification_report(training_Y,
                                       y_1_pred,
                                       output_dict=True)
    print( classification_report(training_Y, y_1_pred))
    sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
    ax.set_title("Classification Report - K-Means- Training Dataset {}".format(dataset_num))
    plt.show()
    #print(clf_report)
    
    
    y_1_pred=pd.DataFrame(kmeans_classifier.predict(test_X), columns=["label"]) 
    y_1_pred['label']=y_1_pred["label"].astype('category')
    
    ax = plt.axes()
    clf_report = classification_report(test_Y,
                                       y_1_pred,
                                       output_dict=True)
    print( classification_report(test_Y, y_1_pred))
    sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
    ax.set_title("Classification Report - K-Means- Test Dataset {}".format(dataset_num))
    plt.show()
    #print(clf_report)
    
    hcv_train=hcv(training_Y, y_prediction)
    hcv_test=hcv(test_Y, y_prediction_test)
    print("The values for homogeneity, completeness, and V-measure for Training Dataset {} : {}".format(dataset_num, hcv_train))
    print("The values for homogeneity, completeness, and V-measure for Test Dataset {} : {}".format(dataset_num, hcv_test))
    
    print(" ")
    print(" ")
    print(" ")
    return y_prediction,y_prediction_test


def EM(training_X, training_Y, test_X, test_Y, dataset_num):

    #Code to test out EM Algorithm using a varying number of components and graph the results
    #Source: Scikit Learn Example for using the Gaussian Mixtures model
    #Gaussian Mixture Model Selection"
    #Code modified from 
    #https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
    
    number_rows, number_cols = training_X.shape
    
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 10)
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type=cv_type, random_state=seed
            )
            gmm.fit(training_X)
            bic.append(gmm.bic(training_X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    
    bic = np.array(bic)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    clf = best_gmm
    bars = []
    
    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 0.2 * (i - 2)
        bars.append(
            plt.bar(
                xpos,
                bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                width=0.2,
                color=color,
            )
        )
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
    plt.title("BIC score per model")
    xpos = (
        np.mod(bic.argmin(), len(n_components_range))
        + 0.65
        + 0.2 * np.floor(bic.argmin() / len(n_components_range))
    )
    plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
    spl.set_xlabel("Number of components")
    spl.legend([b[0] for b in bars], cv_types)
    
    # Plot the winner
    if number_cols>1:
        splot = plt.subplot(2, 1, 2)
        Y_ = clf.predict(training_X)
        for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
            v, w = linalg.eigh(cov)
            if not np.any(Y_ == i):
                continue
            plt.scatter(training_X[Y_ == i, 0], training_X[Y_ == i, 1], 0.8, color=color)
        
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)
        
        plt.xticks(())
        plt.yticks(())
        plt.title(
            f"Selected GMM: {best_gmm.covariance_type} model, "
            f"{best_gmm.n_components} components"
        )
        plt.subplots_adjust(hspace=0.35, bottom=0.02)
        plt.show()

    
    gmm = mixture.GaussianMixture(
         n_components=2, covariance_type='full'
    )
    gmm.fit(training_X)

    y_prediction=gmm.predict(training_X)
    y_prediction_test=gmm.predict(test_X)
    
    y_1_pred=pd.DataFrame(gmm.predict(training_X), columns=["label"]) 
    y_1_pred['label']=y_1_pred["label"].astype('category')
    
    
    x_1_full=pd.DataFrame(training_X)
    x_1_full["label"]=y_1_pred
    
    sns.pairplot(x_1_full, hue="label")
    plt.show()
    
    #Find the error on the full training dataset
    #Code to plot classification report
    #Source: Adapted from Stack Overflow example by user akilat90
    #Article "How to plot scikit learn classification report?"
    #Code modified from 
    #https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
    
    ax = plt.axes()
    clf_report = classification_report(training_Y,
                                       y_1_pred,
                                       output_dict=True)
    print("Training Reults")
    print(classification_report(training_Y, y_1_pred))
    sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
    ax.set_title("Classification Report - E-M- Training Dataset {}".format(dataset_num))
    plt.show()
    #print(clf_report)
    
    
    y_1_pred=pd.DataFrame(gmm.predict(test_X), columns=["label"]) 
    y_1_pred['label']=y_1_pred["label"].astype('category')
    
    ax = plt.axes()
    clf_report = classification_report(test_Y,
                                       y_1_pred,
                                       output_dict=True)
    print(classification_report(test_Y, y_1_pred))
    sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
    ax.set_title("Classification Report - E-M - Test Dataset {}".format(dataset_num))
    plt.show()
    #print(clf_report)

    hcv_train=hcv(training_Y, y_prediction)
    hcv_test=hcv(test_Y, y_prediction_test)
    print("The values for homogeneity, completeness, and V-measure for Training Dataset {} : {}".format(dataset_num, hcv_train))
    print("The values for homogeneity, completeness, and V-measure for Test Dataset {} : {}".format(dataset_num, hcv_test))
    
    print(" ")
    print(" ")
    print(" ")
    
    return y_prediction, y_prediction_test

######################################################
#Function to plot validation curves
#Source: Scikit Learn webpage.
#Article "Plotting Validation Curves"
#Code modified from 
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
def plot_validation_curve(classifier, train_x, train_y, par, hyper, metric, cv_choice, classifier_type, hyper_name):
    
    train_scores, test_scores = validation_curve(classifier, train_x, train_y, param_name=par, param_range=hyper, scoring=metric, n_jobs=-1, cv=cv_choice)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
#    print(train_scores)
#    print(train_scores_mean)
    
    ind=np.argmax(np.mean(test_scores, axis=1), axis=0)
    
#    print(hyper[ind])
    
    plt.title("Validation Curve with {0}".format(classifier_type))
    plt.xlabel("{name}".format(name=hyper_name))
    plt.ylabel("Score - {}".format(metric))
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(hyper, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(hyper, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(hyper, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(hyper, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")

    plt.annotate("Best Value: {0}".format(hyper[ind]),
            xy=(hyper[ind],test_scores_mean[ind]), xycoords='data',
            xytext=(20, -30), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
    plt.show()

    
#Dataset 1 K-Means
runKmeans(x_1_train, y_1_train, x_1_test, y_1_test, dataset_num=1)
EM(x_1_train, y_1_train, x_1_test, y_1_test, dataset_num=1)

#Dataset 2 K-Means
runKmeans(x_2_train, y_2_train, x_2_test, y_2_test, dataset_num=2)
EM(x_2_train, y_2_train, x_2_test, y_2_test, dataset_num=2)

#Function to run the neural networks
def run_NeuralNetwork(x_1_train, x_1_test, y_1_train, y_1_test, chosen_cv, rm_type):
    #Dataset 1
    
    NeuralNets_classifier=MLPClassifier(solver='adam', max_iter=2000, learning_rate_init=.001)
    
    hyperparameter_1=[{'activation': ['identity', 'logistic', 'tanh', 'relu']}]
    hyperparameter_2=[{'hidden_layer_sizes': [(5), (5,5), (5,5,5)]}]
    
    #Validation Curve for KNN - # of Neighbors Hyperparameter
    hyper_1=['identity', 'logistic', 'tanh', 'relu']
    plot_validation_curve(NeuralNets_classifier, x_1_train, y_1_train, "activation",
                          hyper_1, "accuracy", chosen_cv, "Neural Networks", "Activation Function")
    
    #Validation Curve for Decision Tree - Max Features Hyperparameter
    hyper_2=[(5), (5,5), (5,5,5)]
    
    train_scores, test_scores = validation_curve(NeuralNets_classifier, x_1_train, y_1_train, param_name="hidden_layer_sizes",
                                                 param_range=hyper_2, scoring="accuracy", n_jobs=-1, cv=chosen_cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
        
    ind=np.argmax(np.mean(test_scores, axis=1), axis=0)
    
    hyper=["1","2","3"]
    plt.title("Validation Curve with Neural Networks")
    plt.xlabel("# of Hidden Layers")
    plt.ylabel("Score - Accuracy")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(hyper, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(hyper, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(hyper, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(hyper, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    
    plt.annotate("Best Value: {0}".format(hyper[ind]),
            xy=(hyper[ind],test_scores_mean[ind]), xycoords='data',
            xytext=(20, -30), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
    plt.show()
    
    #After testing the two hyperparameters I will run a GridSearch to find the best combination of both
    hyper_3=[{'activation': ['identity', 'logistic', 'tanh', 'relu'],
             'hidden_layer_sizes': [(5), (5,5), (5,5,5)]}]
    
    
    classifier=GridSearchCV(NeuralNets_classifier, hyper_3, scoring='accuracy', return_train_score=True, verbose=3, cv=chosen_cv, n_jobs=-1)
    classifier.fit(x_1_train, y_1_train)
    print('Highest Score: %s' % classifier.best_score_)
    print('Corresponding Hyperparameters: %s' % classifier.best_params_)
    
    
    best_activation=classifier.best_params_['activation']
    best_num_hidden_layers=classifier.best_params_['hidden_layer_sizes']
    
    NeuralNets_final_1=MLPClassifier(activation=best_activation, hidden_layer_sizes=best_num_hidden_layers, max_iter=2000, solver='adam', learning_rate_init=.001)
    
    NN=NeuralNets_final_1.fit(x_1_train, y_1_train)
    pred_train=NeuralNets_final_1.predict(x_1_train)
    pred_test=NeuralNets_final_1.predict(x_1_test)
    
    #Find the error on the full training dataset
    #Code to plot classification report
    #Source: Adapted from Stack Overflow example by user akilat90
    #Article "How to plot scikit learn classification report?"
    #Code modified from 
    #https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
    
    ax = plt.axes()
    clf_report = classification_report(y_1_train,
                                       pred_train,
                                       output_dict=True)
    sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
    ax.set_title("Classification Report - Neural Networks - {} - Training Dataset 1".format(rm_type))
    
    plot_conf = plot_confusion_matrix(NeuralNets_final_1, x_1_train, y_1_train,
                                     cmap=plt.cm.Blues, normalize='true')
    plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - {} - Training Dataset 1".format(rm_type))
    
    plt.show()
    
    #Find the error on the test dataset
    ax = plt.axes()
    clf_report = classification_report(y_1_test,
                                       pred_test,
                                       output_dict=True)
    sn.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True ,cmap='Blues', ax=ax)
    ax.set_title("Classification Report - Neural Network - {} - Test Dataset 1".format(rm_type))
    
    #Create a plot of the confusion matrix for the test dataset
    plot_conf = plot_confusion_matrix(NeuralNets_final_1, x_1_test, y_1_test,
                                     cmap=plt.cm.Blues, normalize='true')
    plot_conf.ax_.set_title("Confusion Matrix - Neural Networks - {} - Test Dataset 1".format(rm_type))
    
    plt.show()

#####################################Feature Selection#########################################
#PCA

#Dataset 1

#Calculate the Principal Components for Data and plot the cumulative variance explained
#Code to calculate PCA and plot variance 
#Source: Book "Python Data Science Handbook" by Jake VanderPlas
#Chapter "In Depth: Princpal Component Analysis"
#Code modified from 
#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
pca = PCA().fit(x_1_train)

fig, ax = plt.subplots(1, 2, figsize=(8,4))
plt.tight_layout(w_pad=2, rect=[0, 0.02, 1, .98])
            
ax[0].plot(nums, np.cumsum(pca.explained_variance_ratio_))
ax[0].set_title("Cum Var Explained by # of Comps - Dataset 1")
ax[0].set_xlabel('Number of components')
ax[0].set_ylabel('Cumulative Explained Variance');
print(pca.components_)
print(pca.explained_variance_)

#Calculate the Reconstruction Loss
#Code to calculate Reconstruction Loss
#Source: Stackoverflow user Eickenberg
#PCA projection and reconstruction in scikit-learn
#Code modified from 
#https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn
recon_loss=[]
for i in range(1,11):
    pca = PCA(n_components=i).fit(x_1_train)
    pca_x_1_train=pca.fit_transform(x_1_train)
    X_projected = pca.inverse_transform(pca_x_1_train)
    loss = np.sum((x_1_train - X_projected) ** 2, axis=1).mean()
    recon_loss.append(loss)
    
ax[1].plot(nums, recon_loss)
ax[1].set_title("Reconstruction Loss - Dataset 1")
ax[1].set_xlabel('Number of components')
ax[1].set_ylabel('Reconstruction Loss');
plt.show()


pca = PCA(n_components=8).fit(x_1_train)
pca_x_1_train=pca.transform(x_1_train)
pca_x_1_test=pca.transform(x_1_test)

X_projected = pca.inverse_transform(pca_x_1_train)
loss = np.sum((x_1_train - X_projected) ** 2, axis=1).mean()
print(loss)


#Plot Data prokected onto 2 PC
#Source: Scikit Learn Documentation
#Comparison of LDA and PCA 2D projection of Iris dataset
#Code modified from 
#https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py
plt.figure()
colors = ["navy", "turquoise"]
lw = 2

for color, i, target_name in zip(colors, [0, 1], ["0","1"]):
    plt.scatter(
        pca_x_1_train[y_1_train == i, 0], pca_x_1_train[y_1_train == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of dataset 1 ")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

pca_x_1_train_kmeans_clusters, pca_x_1_test_kmeans_clusters =runKmeans(pca_x_1_train, y_1_train, pca_x_1_test, y_1_test, dataset_num=1)
pca_x_1_train_EM_clusters, pca_x_1_test_EM_clusters=EM(pca_x_1_train, y_1_train, pca_x_1_test, y_1_test, dataset_num=1)


#Dataset 2
pca = PCA().fit(x_2_train)

fig, ax = plt.subplots(1, 2, figsize=(8,4))
plt.tight_layout(w_pad=2, rect=[0, 0.02, 1, .98])
            
ax[0].plot(nums, np.cumsum(pca.explained_variance_ratio_))
ax[0].set_title("Cum Var Explained by # of Comps - Dataset 2")
ax[0].set_xlabel('Number of components')
ax[0].set_ylabel('Cumulative Explained Variance');
print(pca.components_)
print(pca.explained_variance_)

#Calculate the Reconstruction Loss
#Code to calculate Reconstruction Loss
#Source: Stackoverflow user Eickenberg
#PCA projection and reconstruction in scikit-learn
#Code modified from 
#https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn
recon_loss=[]
for i in range(1,11):
    pca = PCA(n_components=i).fit(x_2_train)
    pca_x_2_train=pca.fit_transform(x_2_train)
    X_projected = pca.inverse_transform(pca_x_2_train)
    loss = np.sum((x_2_train - X_projected) ** 2, axis=1).mean()
    recon_loss.append(loss)
    
ax[1].plot(nums, recon_loss)
ax[1].set_title("Reconstruction Loss - Dataset 2")
ax[1].set_xlabel('Number of components')
ax[1].set_ylabel('Reconstruction Loss');
plt.show()

X_projected = pca.inverse_transform(pca_x_2_train)
loss = np.sum((x_2_train - X_projected) ** 2, axis=1).mean()
print(loss)

pca = PCA(n_components=9).fit(x_2_train)
pca_x_2_train=pca.transform(x_2_train)
pca_x_2_test=pca.transform(x_2_test)


plt.figure()
colors = ["navy", "turquoise"]
lw = 2

for color, i, target_name in zip(colors, [0, 1], ["0","1"]):
    plt.scatter(
        pca_x_2_train[y_2_train == i, 0], pca_x_2_train[y_2_train == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of dataset 2 ")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

runKmeans(pca_x_2_train, y_2_train, pca_x_2_test, y_2_test, dataset_num=2)
EM(pca_x_2_train, y_2_train, pca_x_2_test, y_2_test, dataset_num=2)

#ICA

#Dataset 1
kurt=[]
for i in range(1,11):
    ICA = FastICA(n_components=i,max_iter=10000, tol=1, random_state=seed).fit(x_1_train)
    comps=ICA.components_
    sources_est = ICA.fit_transform(x_1_train) 
    kurt_est=np.mean(kurtosis(sources_est))
    kurt.append(kurt_est)

plt.plot(nums,kurt)
plt.title("Kurtosis vs. Components - Dataset 1")
plt.xlabel('Number of Components')
plt.ylabel('Kurtosis');
plt.show()

ICA = FastICA(n_components=4,max_iter=10000, tol=1, random_state=seed).fit(x_1_train)
ICA_x_1_train=ICA.transform(x_1_train)
ICA_x_1_test=ICA.transform(x_1_test)

plt.plot(ICA_x_1_train)
plt.title("ICA recovered signals")

X_projected = ICA.inverse_transform(ICA_x_1_train)
loss = np.sum((x_1_train - X_projected) ** 2, axis=1).mean()
print(loss)

ica_x_1_train_kmeans_clusters, ica_x_1_test_kmeans_clusters =runKmeans(ICA_x_1_train, y_1_train, ICA_x_1_test, y_1_test, dataset_num=1)
ica_x_1_train_EM_clusters, ica_x_1_test_EM_clusters=EM(ICA_x_1_train, y_1_train, ICA_x_1_test, y_1_test, dataset_num=1)

#Dataset 2
nums=[1,2,3,4,5,6,7,8,9,10]
kurt=[]
for i in range(1,11):
    ICA = FastICA(n_components=i,max_iter=10000, tol=1, random_state=seed).fit(x_2_train)
    comps=ICA.components_
    sources_est = ICA.fit_transform(x_2_train) 
    kurt_est=np.mean(kurtosis(sources_est))
    kurt.append(kurt_est)

plt.plot(nums,kurt)
plt.title("Kurtosis vs. Components - Dataset 2")
plt.xlabel('Number of Components')
plt.ylabel('Kurtosis');
plt.show()

recon_loss=[]
for i in range(1,11):
    ICA = FastICA(n_components=i,max_iter=10000, tol=1, random_state=seed).fit(x_2_train)
    ICA_x_2_train=ICA.fit_transform(x_2_train)
    X_projected = ICA.inverse_transform(ICA_x_2_train)
    loss = np.sum((x_2_train - X_projected) ** 2, axis=1).mean()
    recon_loss.append(loss)
    
plt.plot(nums, recon_loss)
plt.title("Reconstruction Loss - ICA - Dataset 2")
plt.xlabel("# of Components")
plt.ylabel("Loss")
plt.show()

ICA = FastICA(n_components=10,max_iter=10000, tol=1, random_state=seed).fit(x_2_train)
ICA_x_2_train=ICA.transform(x_2_train)
ICA_x_2_test=ICA.transform(x_2_test)

plt.plot(ICA_x_2_train)
plt.title("ICA recovered signals")

X_projected = ICA.inverse_transform(ICA_x_2_train)
loss = np.sum((x_2_train - X_projected) ** 2, axis=1).mean()
print(loss)

runKmeans(ICA_x_2_train, y_2_train, ICA_x_2_test, y_2_test, dataset_num=2)
EM(ICA_x_2_train, y_2_train, ICA_x_2_test, y_2_test, dataset_num=2)

#Random Projections

#Dataset 1
smallest_loss=[]
for i in range(1,1000):
    recon_loss=[]
    for i in range(1,11):
        RP = random_projection.SparseRandomProjection(n_components=i)
        
        RP.fit(x_1_train)
        components =  RP.components_.toarray()
        p_inverse = np.linalg.pinv(components.T)
    
        reduced_data = RP.transform(x_1_train) 
        X_projected= reduced_data.dot(p_inverse) 
       # assert  reduced_data.shape ==  X_projected.shape
        loss = np.sum((x_1_train - X_projected) ** 2, axis=1).mean()
        recon_loss.append(loss)
    min_loss_value = min(recon_loss)
    min_loss_index = recon_loss.index(min_loss_value)
    smallest_loss.append(min_loss_index)
    
mode=statistics.mode(smallest_loss)
print(mode)


RP = random_projection.SparseRandomProjection(n_components=mode, random_state=seed)
RP.fit(x_1_train)

RP_x_1_train=RP.transform(x_1_train)
RP_x_1_test=RP.transform(x_1_test)

rp_x_1_train_kmeans_clusters, rp_x_1_test_kmeans_clusters =runKmeans(RP_x_1_train, y_1_train, RP_x_1_test, y_1_test, dataset_num=1)
rp_x_1_train_EM_clusters, rp_x_1_test_EM_clusters=EM(RP_x_1_train, y_1_train, RP_x_1_test, y_1_test, dataset_num=1)


#Dataset 2
smallest_loss=[]
for i in range(1,1000):
    recon_loss=[]
    for i in range(1,11):
        RP = random_projection.SparseRandomProjection(n_components=i)
        
        RP.fit(x_2_train)
        components =  RP.components_.toarray()
        p_inverse = np.linalg.pinv(components.T)
    
        reduced_data = RP.transform(x_2_train) 
        X_projected= reduced_data.dot(p_inverse) 
       # assert  reduced_data.shape ==  X_projected.shape
        loss = np.sum((x_2_train - X_projected) ** 2, axis=1).mean()
        recon_loss.append(loss)
    min_loss_value = min(recon_loss)
    min_loss_index = recon_loss.index(min_loss_value)
    smallest_loss.append(min_loss_index)
    
mode=statistics.mode(smallest_loss)
print(mode)

RP = random_projection.SparseRandomProjection(n_components=mode, random_state=seed)
RP.fit(x_2_train)

RP_x_2_train=RP.transform(x_2_train)
RP_x_2_test=RP.transform(x_2_test)

runKmeans(RP_x_2_train, y_2_train, RP_x_2_test, y_2_test, dataset_num=2)
EM(RP_x_2_train, y_2_train, RP_x_2_test, y_2_test, dataset_num=2)

#Xgboost
import xgboost
import shap

#Builds Xgboost model and Calculates and Visualizes the Shapley Values for Feature Importance
#Source: SHAP Documentation - Scott Lundberg
#Census income classification with XGBoost
#Code modified from 
#https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20XGBoost.html

#Dataset 1
d_train = xgboost.DMatrix(x_1_train, label=y_1_train)
d_test = xgboost.DMatrix(x_1_test, label=y_1_test)

params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_1_train),
    "eval_metric": "logloss"
}
model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_1_train)
shap.summary_plot(shap_values, x_1_train, plot_type="bar")
shap.summary_plot(shap_values, x_1_train)

SHAP_x_1_train=x_1_train[:, [0,1,3,5,6,7,8,9]]
SHAP_x_1_test=x_1_test[:, [0,1,3,5,6,7,8,9]]

shap_x_1_train_kmeans_clusters, shap_x_1_test_kmeans_clusters =runKmeans(SHAP_x_1_train, y_1_train, SHAP_x_1_test, y_1_test, dataset_num=1)
shap_x_1_train_EM_clusters, shap_x_1_test_EM_clusters=EM(SHAP_x_1_train, y_1_train, SHAP_x_1_test, y_1_test, dataset_num=1)

#Dataset 2

d_train = xgboost.DMatrix(x_2_train, label=y_2_train)
d_test = xgboost.DMatrix(x_2_test, label=y_2_test)

params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_2_train),
    "eval_metric": "logloss"
}
model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_2_train)
shap.summary_plot(shap_values, x_2_train, plot_type="bar")
shap.summary_plot(shap_values, x_2_train)

SHAP_x_2_train=x_2_train[:, [0,3,7]]
SHAP_x_2_test=x_2_test[:, [0,3,7]]

runKmeans(SHAP_x_2_train, y_2_train, SHAP_x_2_test, y_2_test, dataset_num=2)
EM(SHAP_x_2_train, y_2_train, SHAP_x_2_test, y_2_test, dataset_num=2)

###################################### Neural Networks  Retrain ####################################

chosen_cv=KFold(n_splits=5, shuffle=True, random_state=seed)

#Original Training of Neural Network

x_1_train=x_1_train_copy
x_1_test=x_1_test_copy


start=time.time()
run_NeuralNetwork(x_1_train, x_1_test, y_1_train, y_1_test, chosen_cv, "Original")
end=time.time()
final_time=end-start
print("Time to train Original Neural Network: ", final_time)



#PCA Neural Network Re-Training

x_1_train=pca_x_1_train
x_1_test=pca_x_1_test

x_1_train_pca_km=np.c_[x_1_train, pca_x_1_train_kmeans_clusters]
x_1_test_pca_km=np.c_[x_1_test, pca_x_1_test_kmeans_clusters]

x_1_train_pca_em=np.c_[x_1_train, pca_x_1_train_EM_clusters]
x_1_test_pca_em=np.c_[x_1_test, pca_x_1_test_EM_clusters]


start=time.time()
run_NeuralNetwork(x_1_train, x_1_test, y_1_train, y_1_test, chosen_cv, "PCA")
end=time.time()
final_time=end-start
print("Time to train Neural Network using PCA Reduced Data: ", final_time)


start=time.time()
run_NeuralNetwork(x_1_train_pca_km, x_1_test_pca_km, y_1_train, y_1_test, chosen_cv, "PCA + K-Means ")
end=time.time()
final_time=end-start
print("Time to train Neural Network using PCA + K-means Reduced Data: ", final_time)


start=time.time()
run_NeuralNetwork(x_1_train_pca_em, x_1_test_pca_em, y_1_train, y_1_test, chosen_cv, "PCA + E-M")
end=time.time()
final_time=end-start
print("Time to train Neural Network using PCA + E-M Reduced Data: ", final_time)



#ICA Neural Network Re-Training
x_1_train=ICA_x_1_train
x_1_test=ICA_x_1_test


x_1_train_ica_km=np.c_[x_1_train, ica_x_1_train_kmeans_clusters]
x_1_test_ica_km=np.c_[x_1_test, ica_x_1_test_kmeans_clusters]

x_1_train_ica_em=np.c_[x_1_train, ica_x_1_train_EM_clusters]
x_1_test_ica_em=np.c_[x_1_test, ica_x_1_test_EM_clusters]


start=time.time()
run_NeuralNetwork(x_1_train, x_1_test, y_1_train, y_1_test, chosen_cv, "ICA")
end=time.time()
final_time=end-start
print("Time to train Neural Network using ICA Reduced Data: ", final_time)

start=time.time()
run_NeuralNetwork(x_1_train_ica_km, x_1_test_ica_km, y_1_train, y_1_test, chosen_cv, "ICA + K-Means ")
end=time.time()
final_time=end-start
print("Time to train Neural Network using ICA + K-means Reduced Data: ", final_time)


start=time.time()
run_NeuralNetwork(x_1_train_ica_em, x_1_test_ica_em, y_1_train, y_1_test, chosen_cv, "ICA + E-M")
end=time.time()
final_time=end-start
print("Time to train Neural Network using ICA + E-M Reduced Data: ", final_time)



#Random Projections Neural Network Re-Training
x_1_train=RP_x_1_train
x_1_test=RP_x_1_test

x_1_train_rp_km=np.c_[x_1_train, rp_x_1_train_kmeans_clusters]
x_1_test_rp_km=np.c_[x_1_test, rp_x_1_test_kmeans_clusters]

x_1_train_rp_em=np.c_[x_1_train, rp_x_1_train_EM_clusters]
x_1_test_rp_em=np.c_[x_1_test, rp_x_1_test_EM_clusters]

start=time.time()
run_NeuralNetwork(x_1_train, x_1_test, y_1_train, y_1_test, chosen_cv, "Random Projections")
end=time.time()
final_time=end-start
print("Time to train Neural Network using Random Projections Reduced Data: ", final_time)

start=time.time()
run_NeuralNetwork(x_1_train_rp_km, x_1_test_rp_km, y_1_train, y_1_test, chosen_cv, "RP + K-Means ")
end=time.time()
final_time=end-start
print("Time to train Neural Network using Random Projections + K-means Reduced Data: ", final_time)


start=time.time()
run_NeuralNetwork(x_1_train_rp_em, x_1_test_rp_em, y_1_train, y_1_test, chosen_cv, "RP + E-M")
end=time.time()
final_time=end-start
print("Time to train Neural Network using Random Projections + E-M Reduced Data: ", final_time)


#Feature Selection using Shapley Values and xgboost Neural Network Re-Training
x_1_train=SHAP_x_1_train
x_1_test=SHAP_x_1_test

x_1_train_shap_km=np.c_[x_1_train, shap_x_1_train_kmeans_clusters]
x_1_test_shap_km=np.c_[x_1_test, shap_x_1_test_kmeans_clusters]

x_1_train_shap_em=np.c_[x_1_train, shap_x_1_train_EM_clusters]
x_1_test_shap_em=np.c_[x_1_test, shap_x_1_test_EM_clusters]

start=time.time()
run_NeuralNetwork(x_1_train, x_1_test, y_1_train, y_1_test, chosen_cv, "SHAP Values")
end=time.time()
final_time=end-start
print("Time to train Neural Network using Shapley Value Feature Importance: ", final_time)

start=time.time()
run_NeuralNetwork(x_1_train_shap_km, x_1_test_shap_km, y_1_train, y_1_test, chosen_cv, "SHAP + K-Means ")
end=time.time()
final_time=end-start
print("Time to train Neural Network using Feature Selection with SHAP + K-means Reduced Data: ", final_time)


start=time.time()
run_NeuralNetwork(x_1_train_shap_em, x_1_test_shap_em, y_1_train, y_1_test, chosen_cv, "SHAP + E-M")
end=time.time()
final_time=end-start
print("Time to train Neural Network using Feature Selection with SHAP + E-M Reduced Data: ", final_time)