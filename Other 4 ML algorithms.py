# https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
# https://learnopencv.com/histogram-of-oriented-gradients/ 
# need arrays for data and graphs for visualisation
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pprint
import seaborn as sns
pp = pprint.PrettyPrinter(indent=4)

# Preparing the data set

import joblib
from skimage.io import imread
from skimage.transform import resize
 
def resize_all(src, pklname, include, width=28, height=None): # width=150
         
    height = height if height is not None else width
     
    data = dict()
    data['description'] = 'resized ({0}x{1}) images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
 
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)
 
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    #im = imread(os.path.join(current_path, file))
                    im = imread(os.path.join(current_path, file),as_gray=True)
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir) 
                    # data['label'].append(subdir[:-4])
                    data['filename'].append(file)
                    data['data'].append(im)
 
        joblib.dump(data, pklname)

data_path = fr'F:\Kyrgyz letters\All'
os.listdir(data_path)

base_name = 'kyrgyz_letters'
width = 28
 
#include = {'А', 'Б', 'Ү', 'Ө'}
include = {'Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я', 'Ң', 'Ү', 'Ө'}
 
resize_all(src=data_path, pklname=base_name, width=width, include=include)

# Let’s load the data from disk and print a summary.

from collections import Counter
 
data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
 
print('number of samples: ', len(data['data']))
print('keys: ', list(data.keys()))
print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))
 
Counter(data['label'])

# The images below show an example of each letter included.
# use np.unique to get all unique values in the list of labels
labels = np.unique(data['label'])
 
fig, axes = plt.subplots(1, len(labels))
fig.set_size_inches(15,4)
fig.tight_layout()
 
for ax, label in zip(axes, labels):
    idx = data['label'].index(label)
     
    ax.imshow(data['data'][idx])
    ax.axis('off')
    ax.set_title(label)
plt.show()

# By convention, we name the input data X and result (labels) y.
X = np.array(data['data'])
y = np.array(data['label'])
# to split our data into a test set and a training set. 
from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True,
    random_state=42,
)


from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
 
letter = imread(os.path.join('F:\Kyrgyz letters\All\А', '58bbc494c9805.png'),as_gray=True)

# scale down the image to one third
letter = rescale(letter, 1/3, mode='reflect')
# calculate the hog and return a visual representation.
letter_hog, letter_hog_img = hog(
    letter, pixels_per_cell=(14,14), 
    cells_per_block=(2, 2), 
    orientations=9, 
    visualize=True, 
    block_norm='L2-Hys')
 
fig, ax = plt.subplots(1,2)
fig.set_size_inches(8,6)
# remove ticks and their labels
[a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False) 
    for a in ax]
 
ax[0].imshow(letter, cmap='gray')
ax[0].set_title('letter')
ax[1].imshow(letter_hog_img, cmap='gray')
ax[1].set_title('hog')
plt.show()


# Transformers
# Define custom transformer
from sklearn.base import BaseEstimator, TransformerMixin
 
class HogTransformer(BaseEstimator, TransformerMixin):
     
    def __init__(self, y=None, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
import skimage
seed = 9
 
# define an instance of each transformer
hogify = HogTransformer(
    pixels_per_cell=(8, 8), 
    cells_per_block=(3,3), 
    orientations=9, 
    block_norm='L2-Hys'
)
scalify = StandardScaler()
 
X_train_hog = hogify.fit_transform(X_train)
X_train_prepared = scalify.fit_transform(X_train_hog)
 
print(X_train_prepared.shape)

# Training
sgd_clf = SVC(kernel='rbf',random_state=seed)
sgd_clf.fit(X_train_prepared, y_train)

# Testing
X_test_hog = hogify.transform(X_test)
X_test_prepared = scalify.transform(X_test_hog)


y_pred = sgd_clf.predict(X_test_prepared)
print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test)) # 91.75

from sklearn.metrics import confusion_matrix
cmx = confusion_matrix(y_test, y_pred)
cmx

from tabulate import tabulate
clm_titles= ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я', 'Ң', 'Ү', 'Ө']
df_cm=pd.DataFrame(cmx, clm_titles, clm_titles)
print(tabulate(df_cm,  tablefmt = 'psql', headers=clm_titles))


# Optimisation
# In the next step, we’ll set up a pipeline that preprocesses the data, 
# trains the model and allows us to adjust parameters more easily.
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


 
HOG_pipeline = Pipeline([
    ('hogify', HogTransformer(
        pixels_per_cell=(14, 14), 
        cells_per_block=(2, 2), 
        orientations=9, 
        block_norm='L2-Hys')
    ),
    ('scalify', StandardScaler()),
    ('classify', SGDClassifier(random_state=seed, max_iter=1000, tol=1e-3))
])
 
clf = HOG_pipeline.fit(X_train, y_train)
print('Percentage correct: ', 100*np.sum(clf.predict(X_test) == y_test)/len(y_test))

# Grid search
from sklearn.model_selection import GridSearchCV

 
param_grid = [
    {
        'hogify__orientations': [9],
        'hogify__cells_per_block': [(3, 3)],
        'hogify__pixels_per_cell': [(8, 8)],
    
        'classify': [
             #SGDClassifier(random_state=seed, max_iter=1000, tol=1e-3),
             SVC(kernel='rbf',random_state=seed),
             LogisticRegression(random_state=seed),
             #DecisionTreeClassifier(random_state=seed),
             #KNeighborsClassifier(),
             #GaussianNB(),
             #LinearDiscriminantAnalysis(),
             RandomForestClassifier(n_estimators=100, random_state=seed),
             MLPClassifier(activation='relu', hidden_layer_sizes=(100, 36), alpha = 0.3, random_state=seed),
             
          
         ]
    }
]

grid_search = GridSearchCV(HOG_pipeline, 
                           param_grid, 
                           cv=3,
                           n_jobs=-1,
                           scoring='accuracy',
                           verbose=1,
                           return_train_score=True)
 
grid_res = grid_search.fit(X_train, y_train)

# save the model to disk
joblib.dump(grid_res, 'hog_sgd_model.pkl');

# description of the best performing object, a pipeline in our case.
grid_res.best_estimator_

grid_res.best_params_
pp.pprint(grid_res.best_params_)

# the highscore during the search
grid_res.best_score_



best_pred = grid_res.predict(X_test)
print('Percentage correct: ', 100*np.sum(best_pred == y_test)/len(y_test))

cmx_svm = confusion_matrix(y_test, best_pred)
plot_confusion_matrix(cmx, vmax1=225, vmax2=100, vmax3=12)
plot_confusion_matrix(cmx_svm, vmax1=225, vmax2=100, vmax3=12)


import pandas as pd

results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values(by=['rank_test_score'])
results_df = (
    results_df
    .set_index(results_df["params"].apply(
        lambda x: "_".join(str(val) for val in x.values()))
    )
    .rename_axis('kernel')
)
results_df[
    #['params', 'rank_test_score', 'mean_test_score', 'std_test_score']
    [ 'rank_test_score', 'mean_test_score', 'std_test_score']
]

results_df[
    #['params', 'rank_test_score', 'mean_test_score', 'std_test_score']
    [ 'rank_test_score', 'mean_test_score']
]

grid_search.cv_results_.keys()
from IPython.display import display
from tabulate import tabulate
display((pd.DataFrame(grid_search.cv_results_)[['params', 'rank_test_score', 'mean_test_score', 'std_test_score']]))
df=pd.DataFrame(grid_search.cv_results_)[['params', 'rank_test_score', 'mean_test_score', 'std_test_score']]
df.style
# displaying the DataFrame
print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

# Plot the models results
plt.figure().set_size_inches(7, 5)
fg3 = sns.barplot(x=df['rank_test_score'], y=df['mean_test_score'],data=df, palette="hls")
fg3.set_xticklabels(rotation=12, labels=['SVM','ANN','Random Forest','Logistic Regression'])
fg3.set(xlabel='Machine Learning Classifier Models', ylabel='Mean of Test scores')
# Iterrating over the bars one-by-one
for bar in fg3.patches:
    
  # Using Matplotlib's annotate function and
  # passing the coordinates where the annotation shall be done
  # x-coordinate: bar.get_x() + bar.get_width() / 2
  # y-coordinate: bar.get_height()
  # free space to be left to make graph pleasing: (0, 8)
  # ha and va stand for the horizontal and vertical alignment
    fg3.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')

plt.show()









