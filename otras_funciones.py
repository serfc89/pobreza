#librerías y una función de cuadernos pasados
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.datasets.samples_generator import make_blobs
import random
# import pygal
# from pygal.style import Style
from IPython.display import SVG

def modificar_datos_iniciales(df):
    # La siguiente función modifica los datos iniciales para que sea 
    # más sencilla la interpretación de los datos de Costa Rica y para consistencia C-CR
    df["pobreza"] = df.np.apply(lambda x : 1 if x < 2 else 0)
    df["menores12"] = df.R4T1
    df["hacinamiento"] = df.TamViv / df.V8.apply(lambda x : x if x > 0 else 1)
    df["escolaridad"] = np.floor(df.Escolari_mean)
    df.drop("np")

def recode_family(data_frame):
    # OJO: estamos juntando Ninyos y adultos mayores con biparental. 
    # Por motivos de consistencia entre bases C-CR.
    dictio = {"Otros":"bi-parental", "Solo adultos y mayores":"adultos_y_mayores", 
            "Solo mayores":"solo_mayores", "Monoparental":"monoparental",
            "Ninyos y adultos mayores":"bi-parental"}
    data_frame.map(dictio)

def drop_many_columns(data_frame):
    # OJO: se tumbaron interacciones, entropías y modas que estaban en la lista de variables.
    # Se quedaron: medias y desviaciones estándar.
    df.drop(list(df.filter(regex = '(_entropy)|(_mode)|(_r_)')), axis = 1, inplace = True)

def datos_generados():
    X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
    return X, y

def grafica_1():
    X, y = datos_generados()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plt.show()

def grafica_2():
    X, y = datos_generados()
    xfit = np.linspace(-1, 3.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

    for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
        plt.plot(xfit, m * xfit + b, '-k')

    plt.xlim(-1, 3.5)
    plt.show()

def grafica_3():
    X, y = datos_generados()
    xfit = np.linspace(-1, 3.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

    for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
        yfit = m * xfit + b
        plt.plot(xfit, m * xfit + b, '-k')
        plt.plot(xfit, yfit, '-k')
        plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)

    plt.xlim(-1, 3.5)
    plt.show()

def grafica_4(svc):
    X, y = datos_generados()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

    for vec in svc.support_vectors_:
        plt.plot(vec[0], vec[1],'x', color='blue', markeredgewidth=2, markersize=8)
    w = svc.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (svc.intercept_[0]) / w[1]
    margin = 1 / np.sqrt(np.sum(svc.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    #plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.xlim(-1, 3.5)
    plt.show()

def inclusion_exclusion_curva(pobreza, probabilidades):
    precision, recall, thresholds = precision_recall_curve(pobreza,probabilidades)
    exclusion_errors = []
    inclusion_errors = []

    for elemento in precision:
        inclusion_error = 1-elemento
        inclusion_errors.append(inclusion_error)

    for elemento in recall:
        exclusion_error = 1-elemento
        exclusion_errors.append(exclusion_error)
    return inclusion_errors, exclusion_errors, thresholds

def mapa_calor(nl, train_or_test):
    plt.figure(figsize=(10,8))
    cmap= "YlGnBu_r"
    if(train_or_test == "train"):
        dfxhogxregion_aux = pd.DataFrame(nl).pivot(index="Número de árboles", columns="Profundidad", values="Performance_train")
        ax = sns.heatmap(dfxhogxregion_aux, annot=True, fmt='.2f', cmap=cmap)
        plt.show()
    else:
        dfxhogxregion_aux = pd.DataFrame(nl).pivot(index="Número de árboles", columns="Profundidad", values="Performance_test")
        ax = sns.heatmap(dfxhogxregion_aux, annot=True, fmt='.2f', cmap=cmap)
        plt.show()

def mixer(c1, c2, ron):
    return (c1[0]*ron+c2[0]*(1-ron), c1[1]*ron+c2[1]*(1-ron), c1[2]*ron+c2[2]*(1-ron))

def clasificar_deseabilidad(df, approved_vars):
    nl=[]
    nombres_or=sorted(list(set(df.Variable)))
    for var in approved_vars:
        nd=dict()
        flag=0
        nombre_original=""
        for nor in nombres_or:
            if nor in var:
                flag=1
                nombre_original=nor
        if var=="urban":
            nombre_original='ZONA'
            flag=1
        if var == 'Geovar':
            nombre_original = 'ZONA'
            flag = 1
        if var == "serv_domestico_total":
            nombre_original = 'A3'
            flag = 1
        if var == "pensionistas_total":
            nombre_original = 'A3'
            flag = 1
        if flag==0:
            print(var)
            print(0/0)
        nd["Variable"]=var
        nd["Deseabilidad"]=int(df.loc[df.Variable==nombre_original, "Deseabilidad"])
        nd["Categoría"]=str(list(df.loc[df.Variable==nombre_original, "Categoría"])[0])
        nd["Nombre_colapsado"]=str(list(df.loc[df.Variable==nombre_original, "Descripción"])[0])
        nl.append(nd)
    df=pd.DataFrame(nl)
    return df


def findclosest(lista, closer):
    kindex = 0
    mindist = abs(lista[0]-closer)
    index = 0
    for k in lista:
        if abs(k-closer)==mindist:
            if random.random()<0.2:
                mindist = abs(k-closer)
                kindex = index
            
        if abs(k-closer)<mindist:
            mindist = abs(k-closer)
            kindex = index
        index =index+ 1
    return kindex


def inclusion_exclusion_curva(pobreza, probabilidades):
    precision, recall, thresholds = precision_recall_curve(pobreza,probabilidades)
    exclusion_errors = []
    inclusion_errors = []
  
    for elemento in precision:
        inclusion_error = 1-elemento
        inclusion_errors.append(inclusion_error)
    
    for elemento in recall:
        exclusion_error = 1-elemento
        exclusion_errors.append(exclusion_error)
    return inclusion_errors, exclusion_errors, thresholds



def treemap(approved_vars, importancia_variables):
    df=pd.read_csv("../datos/AllowedVars.csv")
    df2=clasificar_deseabilidad(df, approved_vars)
    
    df3=importancia_variables.set_index("Variable").join(df2.set_index("Variable")).reset_index()
    df4=df3.groupby("Nombre_colapsado").sum().drop("Deseabilidad", axis=1).join(df2.set_index("Nombre_colapsado")).reset_index().groupby("Nombre_colapsado").first().reset_index()
    labels=df4.Nombre_colapsado
    sizes=df4.Importancia
    labelcategories=df4["Categoría"]
    plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
    c2=(0.045272830626102545, 0.5275126949074208, 0.6837146759611158)
    c1=(0.8401918643172234, 0.941694704442424, 0.7451437718106987)
    maxi=np.max(sizes)
    mini=np.min(sizes)
    coloris=[mixer(c1, c2, (i-mini)/(maxi-mini)) for i in sizes]
    # If you have 2 lists
    treemap = pygal.Treemap()
    treemap.title = 'Feature Importance'
    names=[]
    nnames=[]
    finalcats=list(set(labelcategories))
    for cat in finalcats:
        nd=[]
        for l, s, c in zip(labels, sizes, labelcategories):
            lab=l
            #if l in names: lab=nnames[names.index(l)]
            #lab=lab.decode("utf-8", errors="ignore")
            if c==cat: nd.append({"value": round(s,4), "label":lab})      # rounds to 4 decimals
        treemap.add(cat, nd)
    a = treemap.render()
    styl = Style(
        background = 'transparent',
        plot_background = '#fff',
        foreground = '#333333',
        foreground_strong = '#666',
        foreground_subtle = '#222222',
        opacity = '.85',
        opacity_hover = '.9',
        transition = '250ms ease-in',    
        allow_interruptions=True,
        colors = ('#08b69f', '#1f9395', '#36708b', '#4e4d81', '#652a77', '#7c086d')
    #colors = colorines
    )
    sc=1.8
    display(SVG(treemap.render(show_legend=True, print_labels=False, 
                               legend_at_bottom=False, legend_at_bottom_columns=5,
                               legend_box_size=10, style=styl,allow_interruptions=True,dynamic_print_values=True,
                               width=int(700*sc), height=int(400*sc))))
    treemap.render_to_file('treemap.svg', show_legend=True, print_labels=False, legend_at_bottom=False, legend_at_bottom_columns=5 ,legend_box_size=40, style=styl,allow_interruptions=True,dynamic_print_values=False,width=int(700), height=int(400))
    
    
    
    
def comparar_curvas(y, score1, score2, label1, label2):
    # Aquí obtendremos los errores de exclusión e inclusión para ambos predictores
    inclusion_errors1, exclusion_errors1, thresholds1 = inclusion_exclusion_curva(y,score1)
    inclusion_errors2, exclusion_errors2, thresholds2 = inclusion_exclusion_curva(y,score2)
  
    #Graficamos la inclusión y exclusión de ambos predictores
    plt.figure(figsize = (9,9))
    plt.plot(inclusion_errors1, exclusion_errors1, label = label1)
    plt.plot(inclusion_errors2, exclusion_errors2, label = label2)
    
    # Obtenemos un punto de exclusión = 50% en el predictor sin regularización y lo graficamos
    s = findclosest(exclusion_errors2, 0.5)
    
    # Obtenemos dos índices en los errores del predictor con regularización. Uno donde el eraror de 
    # exclusión es igual al punto graficado, y otro análogo para el error de inclusión
    s11 = findclosest(exclusion_errors1, exclusion_errors2[s])
    s12 = findclosest(inclusion_errors1, inclusion_errors2[s])
  
    #Hacemos bootstrap para cada punto que encontramos, y graficamos su intervalo de confianza. 
    graficar_intervalos(thresholds1[s11], score1, y)
    graficar_intervalos(thresholds1[s12], score1, y)
    
    graficar_intervalos(thresholds2[s], score2, y)
  
    plt.ylim([0.0, 1.00])
    plt.xlim([0.0, 1.0])
    plt.legend()


def graficar_intervalos(th, score, y):
    #Obtenemos valores originales de exclusión e inclusión
    exclusion = error_excl(th, score, y)
    inclusion = error_incl(th, score, y)
  
    #Obtenemos percentiles de intervalos de confianza bootstrap
    e25, e975, i25, i975 = exclusion_bootstrap(score, y, th, print_res = False)
  
    #Graficamos los intervalos de confianza como líneas negras
    plt.plot( [inclusion, inclusion], [e25, e975], color = "black", zorder = 20)
    plt.plot( [i25, i975], [exclusion, exclusion], color = "black", zorder = 20)
  
    #Graficamos el punto original
    plt.scatter(inclusion, exclusion, s = 25, color = "red", zorder = 10)
   
    #Arreglamos ejes
    plt.ylabel('Error de exclusión')
    plt.xlabel('Error de inclusión')
    plt.ylim([0.0, 1.00])
    plt.xlim([0.0, 1.0])
    
    
def error_excl(umbral, scores, pobreza):
    #Obtenemos el total de pobres y total de pobres no incluídos con np.sum
    total_pobres = np.sum(pobreza == 1)
    total_pobres_no_incluidos = np.sum((pobreza == 1) & (scores<umbral))
  
    #Definimos error de exclusión usando las variables que acabamos de definir
    error_excl = total_pobres_no_incluidos/total_pobres
    return error_excl

def error_incl(umbral, scores, pobreza):
    #Obtenemos el total de hogares incluídos y total de no pobres incluídos con np.sum
    total_incluidos = np.sum(scores>=umbral)
    total_nopobres_incluidos = np.sum((pobreza == 0) & (scores>=umbral))
  
    #Definimos error de inclusión usando las variables que acabamos de definir
    error_incl = total_nopobres_incluidos/total_incluidos
    return error_incl



def exclusion_bootstrap(scores, y, umbral, print_res = True):

  #Para hacer nuestras muestras con reemplazo, iniciaremos enlistando todos los índices de observaciones que tenemos.
  lista = y.index

  #En exclusions guardaremos los errores de exclusión de cada simulación bootstrap
  exclusions = []
  inclusions = []
  #Haremos 100 simulaciones. La variable u no importa y no se usa, pero irá del 0 al 999.
  for u in range(100):
    print(u, end = "\r")

    #lista_aux trae una serie de índices de hogares obtenidos en muestreo con reemplazo
    list_aux = random.choices(lista, k = len(lista))

    #Guardaremos las probabilidades y la pobreza de los hogares seleccionados en arreglos auxiliares que se 
      #redefinirán cada iteración.
    new_scores = []
    new_y = []
    for s in list_aux:
      new_scores.append(scores[s])
      new_y.append(y[s])
      
    #Para usar la función que tenemos de error_excl, debemos de convertir la lista a función
    new_scores = pd.Series(new_scores)
    new_y = pd.Series(new_y)
    new_exclusion = error_excl(umbral, new_scores, new_y)
    new_inclusion = error_incl(umbral, new_scores, new_y)
    exclusions.append(new_exclusion)
    inclusions.append(new_inclusion)
  #Graficamos la densidad de las exclusiones
  if print_res: 
    ax = sns.distplot(exclusions, kde = True, rug = False, hist = True)
    plt.show()
  
  #Con base en los percentiles de las simulaciones, obtenemos un intervalo de confianza del 95%
  e_p25 = np.percentile(exclusions, 2.5)
  e_p975 = np.percentile(exclusions, 97.5)
  i_p25 = np.percentile(inclusions, 2.5)
  i_p975 = np.percentile(inclusions, 97.5)
  #Lo imprimimos
  if print_res: print("Intervalo de exclusión al 95% de confianza: ("+str(round(100*e_p25, 2))+"% ,"+str(round(100*e_p975, 2))+"%)")
  
  return e_p25, e_p975, i_p25, i_p975