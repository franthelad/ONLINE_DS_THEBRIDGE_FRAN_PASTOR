import pandas as pd
from scipy.stats import pearsonr

def get_features_num_regresion(dataframe, target_col, umbral_corr, pvalue= None):

    '''
    La funcion get_features_num_regresion devuelve las features para la creacion de un modelo de machine learning.

    Estas features deben ser variables numericas y disponer de una correlacón y significacion estadistica significativa con el target, definidos previamente por el usuario.
    La significacion estadistica es nula por defecto.

    Argumentos:

    .-dataframe(pandas.core.frame.DataFrame) -> un dataframe pandas sobre el que realizar el estudio
    .-target_col(str) -> la columna seleccionada como target para nuestro modelo
    .-umbral_corr(float) -> la correlacion minima exigida a una variable con el target para ser designado como feature. Debe estar comprendido entre 0 y 1
    .-pvalue(float) -> la significacion estadistica Pearson maxima exigida a una variable para ser designada como feature (generalmente 0.005). None por defecto

    Retorna:

    Lista con las columnas designadas como features para el modelo.
    Tipo lista compuesto por cadenas de texto.
    '''

    cardinalidad = dataframe[target_col].nunique() / len(dataframe[target_col]) # calculamos la cardinalidad del target

    if (type(umbral_corr) != float) or (umbral_corr < 0) or (umbral_corr > 1): # comprobamos un umbral_corr aceptable

        print('Variable umbral_corr incorrecto.')
        return None

    elif dataframe[target_col].dtype not in ['int8', 'int16', 'int32','int64', 'float16', 'float32', 'float64']: # comprobamos un tipo de target aceptable. Ojo que esto creo debe combinarse con el trabajo de los compañeros

        print('La columna seleccionada como target debe ser numerica.')
        return None
    
    elif cardinalidad < 0: # este no se si ponerlo. Comprobamos cardinalidad aceptable de target. Acorde al enunciado debe ser numerica continua, creo que este punto debe coordinarse con el trabajo de los compañeros
        # esta a cero temporalmente
        print('Tu variable target tiene una cardinalidad muy baja para ser target.')
        return None
    
    lista_numericas = [] # creo lista de variables / columnas nuericas
    for column in dataframe.columns: 
        
        if dataframe[column].dtypes in ['int8', 'int16', 'int32','int64', 'float16', 'float32', 'float64']:
            lista_numericas.append(column) 

    lista_numericas.remove(target_col) # elimino el target para eliminar redundancia y no incluirlo en las posibles features 
    lista_features = [] # creo lista de features 
    for columna in lista_numericas:

        no_nulos = dataframe.dropna(subset= [target_col, columna]) # elimino nulos para estudiar pearson y correlacion entre variables
        corr, pearson = pearsonr(no_nulos[target_col], no_nulos[columna]) # pearson y corr entre target y variables numericas

        if pvalue != None: # si hay pvalue de entrada
            if (abs(corr) >= umbral_corr) and (pearson <= pvalue): # si pvalue y corr pasan filtro del ausuario para adentro
                lista_features.append(columna)
        else: # si no hay pvalue de entrada
            if abs(corr) >= umbral_corr: # si corr pasa filtro de entrada para adentro
                lista_features.append(columna)
    
    return lista_features # devuelvo features aceptables para el usuario