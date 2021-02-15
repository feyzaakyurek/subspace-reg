 # %% Setup Packages
import os
fpath = '/raid/lingo/akyurek/git/rfs-incremental'
if os.getcwd() != fpath: os.chdir(fpath)
%matplotlib inline
import pandas as pd
from IPython.display import HTML
from IPython.display import Latex
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
print(os.getcwd())
# %% Load Data
df = pd.read_csv("inc_results.csv")
df.columns
(df['class'] == df['predicted']).sum() / len(df)



# %%Displays Data Frame
HTML(df[-500:].to_html(escape=False))


classes = df['class'].unique()

confusion = confusion_matrix(df['class'], df['predicted'], labels=classes)

confusion = confusion / confusion.sum(axis=1, keepdims=True)
confusion
sns.set(rc={'figure.figsize':(30,30)})
ax = sns.heatmap(confusion, annot=False, xticklabels=classes, yticklabels=classes).invert_yaxis()


# %% Latex Test
Latex('''
$$E=mc^2$$''')



# <markdown>
# # Heading
# Text
# **bold text**
