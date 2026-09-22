import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_iris, load_diabetes
from sklearn import datasets
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import pprint
import operator

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, cohen_kappa_score, balanced_accuracy_score, hamming_loss
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import DistanceMetric


tf.keras.backend.set_floatx('float64')
#from mlxtend.plotting import plot_decision_regions

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer

from scipy.spatial import distance
import numpy as np
from scipy.spatial.distance import mahalanobis
from collections import defaultdict

#dados
dados=pd.read_excel("centroides2003a2013.xlsx", engine='openpyxl')

#pré-tratamento
TipoDic={'Áreas Antropizadas':1, 'FAP - Aluvial + Vs':2, 'FAP - Aluvial':3, 'FAP + FD':4, 'FD + FAP':4, 'FAB - Aluvial':5, 'FAP':6, 'FAB + FD':7, 'FD + FAB':7, 'FAB + FAP':8, 'FAP + FAB':8, 'FAP - Aluvial + Pab':9, 'Campinaranas':10, '':0, 'FAB + FAP + FD':11, 'FAP + FAB + FD':11, 'FAP + FD + FAB':11, 'FABD':12, 'FD':13, 'FD - Submontana':14, 'FAP + Pab':15, np.nan:16}

dadosveg=dados
dadosveg["VEG_TIP"]=dadosveg.VEG_TIP.map(TipoDic)
#display(dadosveg)

dadosvegSF=dadosveg.loc[(dadosveg['RASTERVALU']==-9999)|(dadosveg['VEG_TIP']==0)|(dadosveg['VEG_TIP']==16)]
dadosvegalt=dadosveg.drop(dadosvegSF.index)
#display(dadosvegalt)

#coord=dadosvegalt.drop(columns=['VEG_TIP','Distance','Distance_1','RASTERVALU','Count__1','Count_2','Count2010'],axis=1)
coord=dadosvegalt[['Y','X','x_g','y_g2']]
X=dadosvegalt[['VEG_TIP','Distance','Distance_1','RASTERVALU']]
X['VEG_TIP']=X['VEG_TIP'].astype('float64', copy=False)
X['RASTERVALU']=X['RASTERVALU'].astype('float64', copy=False)
y=dadosvegalt[['Count2010']]

#Separa os dados em dois conjuntos Leste e Oeste
coordE=coord.loc[(coord['x_g']<337100)]
coordD=coord.loc[(coord['x_g']>=337100)]
Xe=X.loc[coordE.index]
ye=y.loc[coordE.index]
Xd=X.loc[coordD.index]
yd=y.loc[coordD.index]

yd['Count2010']=yd['Count2010'].apply(lambda x:1 if x>0 else 0)
ye['Count2010']=ye['Count2010'].apply(lambda x:1 if x>0 else 0)

#reduzir os conjuntos de dados para balaceá-los

#mapear as duas classes
ye0=ye.loc[ye['Count2010']==0]
yd0=yd.loc[yd['Count2010']==0]
ye1=ye.loc[ye['Count2010']==1]
yd1=yd.loc[yd['Count2010']==1]
#reduzir amostras do maior conjunto
ye0s=ye0.sample(ye1.shape[0])
yd0s=yd0.sample(yd1.shape[0])
#juntar as amostras
yen=pd.concat([ye1, ye0s])
ydn=pd.concat([yd1, yd0s])
#aleatorizar o resultado
yeA=yen.sample(yen.shape[0])
ydA=ydn.sample(ydn.shape[0])
#buscar dados correspondentes no dados de entrada X
XeA=Xe.loc[yeA.index]
XdA=Xd.loc[ydA.index]
#transformar em vetor numerico
ye=np.array(yeA).flatten()
yd=np.array(ydA).flatten()

#classificador knn-fuzzy
class FuzzyKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k, q=2):
        self.k = k
        self.q = q

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)

    def predict(self, X_test):
        predictions = []
        pertinencias = []
        for x in X_test:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            neighbors_idx = np.argsort(distances)[:self.k]
            memberships = {cls: 0 for cls in self.classes}

            for idx in neighbors_idx:
                dist = distances[idx]
                weight = 1 / (dist ** (2 / (self.q - 1)) + 1e-9)
                memberships[self.y_train[idx]] += weight

            predictions.append(max(memberships, key=memberships.get))
            pertinencias.append(memberships)
        return [np.array(predictions), np.array(pertinencias)]
#função para realizar o k-fold cross validation
def EvalClassifiers(Name,Classifiers, X, y, n_splits=10, score = roc_auc_score):
    df = pd.DataFrame()
    #Name = "Incendio"
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    for train_index, test_index in skf.split(X, y):
        # display(train_index)
        df_sim = pd.DataFrame()
        Xtr, Xte = X[train_index], X[test_index]
        ytr, yte = y[train_index], y[test_index]
        # Process the data
        scaler = StandardScaler() #mean zero and variance one
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        for name, clf in Classifiers:
            try:
                clone_clf = clone(clf)
                clone_clf.fit(Xtr,ytr)
                [y_pred, pertine] = clone_clf.predict(Xte)
                df_sim[name] = [score(yte,y_pred)]
                print(df_sim)
            except:
                print("Classifier %s failed to process dataset %s" % (name,Name))
        df = pd.concat([df,df_sim])
    df.to_csv("CSVsF1Score/%s.csv" % Name)
    return df

#Realizar teste da procura do parâmetro ótimo k
Classifiers = [
#    ("Nearest_Neighbors",KNeighborsClassifier(55))
#    ("Linear_SVM", SVC(kernel="linear", C=0.025)),
#    ("RBF_SVM", SVC(gamma='scale', C=1)),
#    ("Gaussian_Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
#    ("Decision_Tree", DecisionTreeClassifier(max_depth=None)),
#    ("Random_Forest", RandomForestClassifier(max_depth=None, n_estimators=100, max_features="auto")),
#    ("Neural_Net", MLPClassifier(alpha=1, max_iter=1000)),
#    ("AdaBoost", AdaBoostClassifier()),
#    ("Naive_Bayes", GaussianNB()),
#    ("QDA", QuadraticDiscriminantAnalysis()),
#    ("TS_Classifier", TSClassifier(Nrules=3, epochs = 300, verbose = False)),
    ("FuzzyKNN3",FuzzyKNN(3,2)),
    ("FuzzyKNN5",FuzzyKNN(5,2)),
    ("FuzzyKNN7",FuzzyKNN(7,2)),
    ("FuzzyKNN9",FuzzyKNN(9,2)),
    ("FuzzyKNN11",FuzzyKNN(11,2)),
    ("FuzzyKNN13",FuzzyKNN(13,2)),
    ("FuzzyKNN15",FuzzyKNN(15,2)),
    ("FuzzyKNN17",FuzzyKNN(17,2)),
    ("FuzzyKNN19",FuzzyKNN(19,2)),
    ("FuzzyKNN21",FuzzyKNN(21,2)),
    ("FuzzyKNN23",FuzzyKNN(23,2)),
    ("FuzzyKNN25",FuzzyKNN(25,2)),
    ("FuzzyKNN27",FuzzyKNN(27,2)),
    ("FuzzyKNN29",FuzzyKNN(29,2)),
    ("FuzzyKNN31",FuzzyKNN(31,2)),
    ("FuzzyKNN33",FuzzyKNN(33,2)),
    ("FuzzyKNN35",FuzzyKNN(35,2)),
    ("FuzzyKNN37",FuzzyKNN(37,2)),
    ("FuzzyKNN39",FuzzyKNN(39,2)),
    ("FuzzyKNN41",FuzzyKNN(41,2)),
    ("FuzzyKNN43",FuzzyKNN(43,2)),
    ("FuzzyKNN45",FuzzyKNN(45,2)),
    ("FuzzyKNN47",FuzzyKNN(47,2)),
    ("FuzzyKNN49",FuzzyKNN(49,2)),
    ("FuzzyKNN51",FuzzyKNN(51,2)),
    ("FuzzyKNN53",FuzzyKNN(53,2)),
    ("FuzzyKNN55",FuzzyKNN(55,2))
    ]

#Conjunto de dados do lado Esquerdo
if not os.path.exists("CSVsF1Score"):
    os.mkdir("CSVsF1Score")
    print("The directory CSVsF1Score has been created.")

data = pd.DataFrame()
#for name, dataset, version in AllDataSets:
name="Incendios"
start_time = time.time()
#    print("\nProcessing dataset: ",name)
#    X, y = datasets.fetch_openml(dataset,version=version,return_X_y = True)
#display(X)
#    y=np.array(y).flatten()
    # Imput missing data
#    X = SimpleImputer().fit_transform(X) #preenche
XeB = SimpleImputer().fit_transform(XeA)
yeB = np.array(yeA).flatten()
    #classifier = FuzzyKNN2(k,2)
df = EvalClassifiers(name,Classifiers, XeB, yeB) #realizar o k-fold cross validation
data = pd.concat([data,df])
data.to_csv("CSVsF1Score/DataSetsEsquedo.csv")
print("\nTime to process the dataset: %2.2f seconds." % (time.time() - start_time))

#Conjunto de Dados do lado Direito
if not os.path.exists("CSVsF1Score"):
    os.mkdir("CSVsF1Score")
    print("The directory CSVsF1Score has been created.")

data = pd.DataFrame()
#for name, dataset, version in AllDataSets:
name="Incendios"
start_time = time.time()
#    print("\nProcessing dataset: ",name)
#    X, y = datasets.fetch_openml(dataset,version=version,return_X_y = True)
#display(X)
#    y=np.array(y).flatten()
    # Imput missing data
#    X = SimpleImputer().fit_transform(X) #preenche
XdB = SimpleImputer().fit_transform(XdA)
ydB = np.array(ydA).flatten()
    #classifier = FuzzyKNN2(k,2)
df = EvalClassifiers(name,Classifiers, XdB, ydB)
data = pd.concat([data,df])
data.to_csv("CSVsF1Score/DataSetsDireito.csv")
print("\nTime to process the dataset: %2.2f seconds." % (time.time() - start_time))

#guardar os resultados
resultDireito=pd.read_csv("CSVsF1Score/DataSetsDireito.csv")
resultEsquerdo=pd.read_csv("CSVsF1Score/DataSetsEsquedo.csv")

#encontrar k ótimo dados do lado direito
MediaDir = resultDireito.mean(0)
KDir = MediaDir.idxmax()


#encontrar k ótimo dados do lado esquerdo
MediaDir = resultDireito.mean(0)
MediaEsq = resultEsquerdo.mean(0)
KEsq = MediaEsq.idxmax()


#Separa os dados em dois conjuntos Leste e Oeste
coordE=coord.loc[(coord['x_g']<337100)]
coordD=coord.loc[(coord['x_g']>=337100)]
Xe=X.loc[coordE.index]
ye=y.loc[coordE.index]
Xd=X.loc[coordD.index]
yd=y.loc[coordD.index]
yd['Count2010']=yd['Count2010'].apply(lambda x:1 if x>0 else 0)
ye['Count2010']=ye['Count2010'].apply(lambda x:1 if x>0 else 0)
#reduzir os conjuntos de dados para balaceá-los
#mapear as duas classes
ye0=ye.loc[ye['Count2010']==0]
yd0=yd.loc[yd['Count2010']==0]
ye1=ye.loc[ye['Count2010']==1]
yd1=yd.loc[yd['Count2010']==1]
#reduzir amostras do maior conjunto
ye0s=ye0.sample(ye1.shape[0])
yd0s=yd0.sample(yd1.shape[0])
#juntar as amostras
yen=pd.concat([ye1, ye0s])
ydn=pd.concat([yd1, yd0s])
#aleatorizar o resultado
yeA=yen.sample(yen.shape[0])
ydA=ydn.sample(ydn.shape[0])
#buscar dados correspondentes no dados de entrada X
XeA=Xe.loc[yeA.index]
XdA=Xd.loc[ydA.index]

#esquerdo
XeB = SimpleImputer().fit_transform(XeA)
yeB = np.array(yeA).flatten()
XeC = SimpleImputer().fit_transform(Xe)
yeC = np.array(ye).flatten()
scaler3 = StandardScaler() #mean zero and variance one
XeB = scaler3.fit_transform(XeB)
XeC = scaler3.transform(XeC)
clf=FuzzyKNN(29,2)
clone_clf = clone(clf)
clone_clf.fit(XeB,yeB)
[y_pred_e, risk_e] = clone_clf.predict(XeC)
risco_normalizado_E = [{0:el[0]/(el[0]+el[1]), 1:el[1]/(el[0]+el[1])} for el in risk_e]
riscoE = [elemento[1] for elemento in risco_normalizado_E]

#direito
XdB = SimpleImputer().fit_transform(XdA)
ydB = np.array(ydA).flatten()
XdC = SimpleImputer().fit_transform(Xd)
ydC = np.array(yd).flatten()
scaler4 = StandardScaler() #mean zero and variance one
XdB = scaler4.fit_transform(XdB)
XdC = scaler4.transform(XdC)
clf2=FuzzyKNN(51,2)
clone_clf2 = clone(clf2)
clone_clf2.fit(XdB,ydB)
[y_pred_d, risk_d] = clone_clf2.predict(XdC)
risco_normalizado_D = [{0:el[0]/(el[0]+el[1]), 1:el[1]/(el[0]+el[1])} for el in risk_d]
riscoD = [elemento[1] for elemento in risco_normalizado_D]

#juntar os resultados dos lados esquerdos e direito em um mapa só
MapaRiscoE=coordE.copy()
MapaRiscoD=coordD.copy()
MapaRiscoE['risco']=riscoE
MapaRiscoD['risco']=riscoD
MapaRisco= pd.concat([MapaRiscoD, MapaRiscoE])
#exibir o gráfico
grafico = plt.scatter(
    x=MapaRisco['x_g'], 
    y=MapaRisco['y_g2'], 
    c=MapaRisco['risco'],      # A 3ª coluna define os valores da cor
    cmap='jet',       # A nossa paleta do azul para o vermelho
    marker='s',            # 's' transforma os pontos em quadrados (pixels)
    s=1,                 # TAMANHO DO PIXEL: Ajuste esse número para mais ou para menos
    edgecolors='none',     # Remove a bordinha branca ou preta ao redor do pixel
    alpha=1.0              # Garante que a cor seja sólida (100% opaca)
)

# Adiciona a legenda do degradê ao lado do gráfico
plt.colorbar(grafico, label='Risco de Incêndio Florestal')

plt.title("Mapa de Risco de Incêndio Florestal")
plt.savefig("mapa_risco_incendio.png", dpi=150, bbox_inches="tight")
plt.show()
print('Mapa salvo como mapa_risco_incendio.png')
