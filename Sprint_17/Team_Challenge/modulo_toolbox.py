import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import math
import warnings
warnings.filterwarnings('ignore')


def paramns_check(df:pd.DataFrame, target_col:str, columns:list, pvalue:float) -> bool:
    '''
    Esta es una funcion de comprobacion para los parametros.

    Comprobamos que:

    .- el parametro df es un dataframe de pandas
    .- el target seleccionado es categorico, definido por un str que referencia clases, en caso de ser numerico corresponderia mapearlo a str
    .- que las columnas proporcionadas son numericas 
    .- que el pvalue es numerico y esta entre 0 y 1

    La funcion devuelve un booleano que certifica si los parametros introducidos son adecuados.
    '''
    
    try:
        if type(df) != pd.core.frame.DataFrame:
            return False
        if df[target_col].dtype != 'object':
            return False
        for col in columns:
            pd.to_numeric(df[col])
        if (float(pvalue) > 1) or (float(pvalue) < 0):
            return False
    except:
        return False
    
    return True

def plot_features_num_classification(df:pd.DataFrame, target_col:str= '', columns:list= [], pvalue:float= 0.05) -> list:
    # version con generador de indices

    '''
    Parametros:
    .- df: un dataframe de pandas
    .- target_col: el nombre de la variable target (debe ser categorica objeto/str, si contiene numeros, procede mapearla)
    .- columns: el nombre de las variables numericas del df, adjuntas en una lista (vacia por defecto)
    .- pvalue: el valor con que queremos comprobar la significancia estadistica, 0.05 por defecto

    Esta funcion cumple tras objetivos: a saber:

    1.- retorna una lista con los nombres de las features numericas que superan un test anova de significancia estadistica superior al establecido en pvalue
    2.- printa una relacion de graficas comparativas de correlacion target-variables numericas para su estudio y miniEDA
    3.- printa una relacion de graficas comparativas de colinealidad entre las distinta variables numericas para su estudio y miniEDA

    Explicamos la funcion mas en detalle a continuacion.
    '''

    paramns_ok = paramns_check(df, target_col, columns, pvalue) # comprobamos que los parametros son adecuados, si no lo son retornamos None y printamos que no lo son
    if not paramns_ok:
        print('Los parametros introduciodos son incorrectos.')
        return None

    if not columns: # si no adjuntamos lista de var numericas, cogemos todas las numericas del df
        columns = df.describe().columns.tolist()

    col_anova = [] # creamos lista vacia donde almacenaremos los nombres de var numericas que cumplen el test anova

    # a continuacion realizamos el test anova
    grps = df[target_col].unique().tolist() # almacenamo los diferentes valores posibles del target en una lista
    for feature in columns: # iteramos las var numricas
        prov_list = [] # lista provisional donde almacenaremos las series de realcion de cada var numrica con los diferentes valores del target
        
        for grp in grps:
            prov_list.append(df[df[target_col] == grp][feature]) # agregamos a la lista las series que comentabamos antes
        
        f_st, p_va = stats.f_oneway(*prov_list) # realizamos el test anova sobre la var numerica de turno (en iteracion actual) en relacion con cada valor del target y comprobamos su pvalue en funcion de su varianza
        if p_va <= pvalue: # si hay significancia estadistica recahazamos H0(medias similares) y adjuntamos el nombre de la feature a col_anova 
            col_anova.append(feature) 
    
    # empezamos con las graficas
    col_anova.insert(0, target_col) # adjuntamos el target a col_anova porque lo necesitaremos para comparar y graficar

    # creamos una primera serie de graficas relacion target(categorica) con las features numericas
    # utilizaremos subplots para reflejar cada grafica individualmente. Estos subplots son referenciados mediante arrays, importante

    q_lineas = math.ceil((len(col_anova)-1)/5) # calculamos la cantidad de lineas que compondra en la figura grafica / array (cada linea comprendera 5 subplots / columnas)

    # vamos a jugar con generadores, uno simple en realidad, no lo hemos visto en temario pero para este caso resulta de mucha utilidad
    # para movernos por los subplots de la figura grafica de turno deberemos iterar las columnas segun grafiquemos diferentes relaciones target-features
    # este generador genera los indices para el subplot
    def gen_indice():
        
        while True:
            for linea in range(100):
                for columna in range(5):
                    yield linea, columna

    contador_indice = gen_indice() # instanciamos el generador


    fig, axs = plt.subplots(q_lineas, 5, figsize=(20, 4*q_lineas)) # generamos la figura grafica con la cantidad de lineas y 5 columnas, tamño acorde a la q de lineas
    fig.suptitle('Correlación target categorico VS features numéricas con significancia estadistica > 1-pvalue')
    plt.subplots_adjust(top=0.9)

    columna = 0 # comenzamos en la linea 0, primera
    indice = next(contador_indice) # primer indice [0, 0]
    # comenzamos a iterar las features que tenemos que graficas
    for feature_index in range(1, len(col_anova)): # rango 1 hata final porque la primera es el target y no queremos graficar target-target
    
        try: # presumimos que la grafica dispondra de mas de 1 linea y graficaremos en consecuencia... 
            for i in df[col_anova[0]].unique():     
                sns.histplot(df[df[col_anova[0]] == i][col_anova[feature_index]], kde= True, ax= axs[indice], label= i)
            axs[indice].legend()
            indice = next(contador_indice) # siguiente indice
        except IndexError: # ...si la figura solo dispone de 1 linea la graficacion dara error y graficamos en consecuencia
            for i in df[col_anova[0]].unique():     
                sns.histplot(df[df[col_anova[0]] == i][col_anova[feature_index]], kde= True, ax= axs[columna], label= i)
            axs[columna].legend()
            columna += 1 # siguiente columna
    plt.show() # mostramos la figura grafica

    # graficamos la colinealidad
    sns.pairplot(df[col_anova], hue= target_col) # pairplot para todas las features numericas que han superado la significancia estadistica
    plt.suptitle('Colinealidad features numéricas con significancia estadistica > 1-pvalue')
    plt.subplots_adjust(top=0.9) 
    plt.show() # mostramos grafica
    col_anova.remove(target_col) # quitamos el target de la lista de features que han superado el test (ya ha sido util para graficar)
    
    return col_anova # devolvemos los nombres de las features que han superado la significancia estadistica
    