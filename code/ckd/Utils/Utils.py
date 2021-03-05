import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict
from colorama import Fore, Back, Style
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, plot_roc_curve
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, recall_score
from xgboost import XGBClassifier, plot_importance
from sklearn.naive_bayes import BernoulliNB

class ColoredFormatter(logging.Formatter):
    """ 
    Colored log formatter class, implements logging.Formatter.

    Parameters: 
        None

    Attributes: 
        None

    """
    def __init__(self, *args, colors: Optional[Dict[str, str]]=None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL

        return super().format(record)

    
def get_formatter(log=True):
    """ 
    Helper function to set log format

    Parameters: 
        log (Boolean):  True if color formatter is for stdout
                        False if formatter is for log file

    Returns: 
        None

    """
    fmt = ColoredFormatter(
        '{asctime} |{levelname:8}| {message}',
        style='{', datefmt='%Y-%m-%d %H:%M:%S',
        colors={
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
        }
    )
    
    if log:
        fmt = ColoredFormatter(
            '{asctime} |{color}{levelname:8}{reset}| {message}',
            style='{', datefmt='%Y-%m-%d %H:%M:%S',
            colors={
                'DEBUG': Fore.CYAN,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
            }
        )
    
    return fmt
    
    
def model_metrics(tag, pipe, param, X_tr, X_ts, y_tr, y_ts, models, preds_df, RS=21, CV=5, SCORE='roc_auc'):
    """ 
    Function to fit and collect model metrics
  
    Parameters: 
        tag (str): name for this pipeline
        pipe (pipeline): pipeline to be executed
		param (dict): dictionary of parameters to be used
        X_tr (Series): Series holding the train feature
        X_ts (Series): Series holding the test feature
        y_tr (Series): Series holding the train target
        y_ts (Series): Series holding the test target
  
    Returns: 
		None:
  
    """
    gs = GridSearchCV(pipe, param_grid=param, cv=CV, scoring=SCORE)
    gs.fit(X_tr, y_tr)
    bm = gs.best_estimator_
    models[tag] = gs
    y_pred = bm.predict(X_ts)
    preds_df[tag]=y_pred
    print(f'{tag}->Best params: {gs.best_params_}')
    
    metric = {}
    metric['01 Train score'] = f'{bm.score(X_tr, y_tr):,.4f}'
    metric['02 Test score'] = f'{bm.score(X_ts, y_ts):,.4f}'
    metric['03 Score diff'] = float(metric['01 Train score'])-float(metric['02 Test score'])
    metric['04 Train recall'] = f'{recall_score(y_tr, bm.predict(X_tr)):,.4f}'
    metric['05 Test recall'] = f'{recall_score(y_ts, y_pred):,.4f}'

    # calculate Specificity and Sensitivity
    tn, fp, fn, tp = confusion_matrix(y_ts, y_pred).ravel()
    metric['06 Precision'] = f'{(tp /(tp + fp)):.4f}'
    metric['07 Specificity'] = f'{(tn / (tn + fp)):.4f}'
    metric['08 Sensitivity'] = f'{(tp / (tp + fn)):.4f}'
    metric['09 True Negatives'] = tn
    metric['10 False Positives'] = fp
    metric['11 False Negatives'] = fn
    metric['12 True Positives'] = tp
    metric['13 Train ROC Score'] = f'{roc_auc_score(y_tr, bm.predict_proba(X_tr)[:,1]):,.4f}'
    metric['14 Test ROC Score'] = f'{roc_auc_score(y_ts, bm.predict_proba(X_ts)[:,1]):,.4f}'
    metric['15 Train CV Score'] = f'{cross_val_score(bm, X_tr, y_tr, cv=CV).mean():,.4f}'
    metric['16 Test CV Score'] = f'{cross_val_score(bm, X_ts, y_ts, cv=CV).mean():,.4f}'

    # plot roc and confusion matrix
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    plot_roc_curve(gs, X_ts, y_ts, name=tag, ax=ax[0]);
    # Plot baseline. (Perfect overlap between the two populations.)
    ax[0].plot(np.linspace(0, 1, 200), np.linspace(0, 1, 200),
             label='baseline', linestyle='--')
    plot_confusion_matrix(gs, X_ts, y_ts, ax=ax[1], cmap='RdPu');
    ax[0].set_title(f'ROC for {tag}');
    ax[1].set_title(f'Confusion Matrix for {tag}');
    
    return metric

def model_testdata(tag, model, X_tr, X_ts, y_tr, y_ts, preds_df, RS=21, CV=5, SCORE='roc_auc'):
    """ 
    Function to fit and collect model metrics
  
    Parameters: 
        X_tr (Series): Series holding the train feature
        X_ts (Series): Series holding the test feature
        y_tr (Series): Series holding the train target
        y_ts (Series): Series holding the test target
  
    Returns: 
		None:
  
    """
    bm = model
    y_pred = bm.predict(X_ts)
    preds_df[tag]=y_pred
    
    
    metric = {}
    metric['01 Train score'] = f'{bm.score(X_tr, y_tr):,.4f}'
    metric['02 Test score'] = f'{bm.score(X_ts, y_ts):,.4f}'
    metric['03 Score diff'] = float(metric['01 Train score'])-float(metric['02 Test score'])
    metric['04 Train recall'] = f'{recall_score(y_tr, bm.predict(X_tr)):,.4f}'
    metric['05 Test recall'] = f'{recall_score(y_ts, y_pred):,.4f}'

    # calculate Specificity and Sensitivity
    tn, fp, fn, tp = confusion_matrix(y_ts, y_pred).ravel()
    metric['06 Precision'] = f'{(tp /(tp + fp)):.4f}'
    metric['07 Specificity'] = f'{(tn / (tn + fp)):.4f}'
    metric['08 Sensitivity'] = f'{(tp / (tp + fn)):.4f}'
    metric['09 True Negatives'] = tn
    metric['10 False Positives'] = fp
    metric['11 False Negatives'] = fn
    metric['12 True Positives'] = tp
    metric['13 Train ROC Score'] = f'{roc_auc_score(y_tr, bm.predict_proba(X_tr)[:,1]):,.4f}'
    metric['14 Test ROC Score'] = f'{roc_auc_score(y_ts, bm.predict_proba(X_ts)[:,1]):,.4f}'
    metric['15 Train CV Score'] = f'{cross_val_score(bm, X_tr, y_tr, cv=CV).mean():,.4f}'
    metric['16 Test CV Score'] = f'{cross_val_score(bm, X_ts, y_ts, cv=CV).mean():,.4f}'

    # plot roc and confusion matrix
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    plot_roc_curve(bm, X_ts, y_ts, name=tag, ax=ax[0]);
    # Plot baseline. (Perfect overlap between the two populations.)
    ax[0].plot(np.linspace(0, 1, 200), np.linspace(0, 1, 200),
             label='baseline', linestyle='--')
    plot_confusion_matrix(bm, X_ts, y_ts, ax=ax[1], cmap='RdPu');
    ax[0].set_title(f'ROC for {tag}');
    ax[1].set_title(f'Confusion Matrix for {tag}');
    
    return metric



def calc_gfr(scr, age, gender, ethnicity):
    """
    Source: https://github.com/chapmanbe/shilpy_6018_2017_term_project
    Calculates the estimated Glomerular Filteration Rate(eGFR)
    
    Based on MDRD equation.
    GFR=175×(Scr)^-1.154×(Age)^-0.203×(0.742 if female)×(1.212 if AA)
    
    Arguments: 
        scr: Patient's serum creatinine level in mg/dL as float
        age: Patient's age in years as integer
        gender: Patient's gender "male" or "female" as string
        ethnicity: Patient's ethicity   
        
    Returns: 
        gfr: Patient's eGFR in mL/min rounded to decimal places.
        
    """
    if scr == 0.0 or age == 0.0:
        return 0
  
    if gender == 'Female' and ethnicity == 'Black':
        egfr = round(175*(scr)**(-1.154) * age**(-0.203) * (0.742) * 1.212, 2)
    elif gender == 'Female' and ethnicity != 'Black':
        egfr = round(175*(scr)**(-1.154) * age**(-0.203) * (0.742), 2)
    elif gender == 'Male' and ethnicity == 'Black':
        egfr = round(175*(scr)**(-1.154) * age**(-0.203) * 1.212, 2)
    else:
        egfr = round(175*(scr)**(-1.154) * age**(-0.203), 2)    
    
    if egfr>=90:
        stage = 1
    elif egfr<90 and egfr>=60:
        stage = 2
    elif egfr<60 and egfr>=30:
        stage = 3
    elif egfr<30 and egfr>=15:
        stage = 4
    else:
        stage = 5
    
    return stage