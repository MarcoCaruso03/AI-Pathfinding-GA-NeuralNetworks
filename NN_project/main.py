import warnings
import time
from typing import Tuple, List, Dict, Callable
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, KFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.exceptions import ConvergenceWarning


####################################
#### CARICAMENTO DATASET ###########
####################################

def load_dataset(filepath : str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    - Carica il dataset
    - Gestisce missing values
    """
    df = pd.read_csv(filepath)

    df = df.drop(columns=['ID'])
    # id-target-features
    y = df['Diagnosis'].map({'B': 0, 'M': 1})
    X = df.drop(columns=['Diagnosis'])
    print(f"Dataset caricato: {X.shape[0]} istanze, {X.shape[1]} caratteristiche.")

    # missing values, not founded

    return X,y



    
####################################
#### CLASSE MLP ####################
####################################

# CLASSE MLP che gestisce l'intero ciclo di vita di un singolo esperimento:
    # - Riceve i parametri specifici (architettura, attivazione, learning rate, ecc.)
    # - Metodo run(): esegue la standardizzazione dei dati e la cross-validation per 
class MLPAlgo: 
    def __init__(self, X, y, params, n_runs=30, seed=42):
        self.X = X
        self.y = y
        if params is None: 
            self.params = {}
        else:
            self.params = params
        self.n_runs = n_runs
        self.seed = seed

    def run_experiment(self):
        # Ci sono 3 livelli: Livello1 delle 30 run, livello 2 della cv con 5 fold, e livello 3 loop manuale sulle epoche
        # Invece di chiamare un semplice .fit() che esegue tutto l'addestramento internamente, il codice imposta max_iter=1 e warm_start=True. Questo permette di "entrare" dentro l'addestramento e registrare l'accuratezza e la loss dopo ogni singola epoca
        all_results = []
        # Silenziamo i ConvergenceWarning solo per questo blocco di codice
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
        for i in range(self.n_runs):
            print(f"Run: {i+1}/{self.n_runs}")
            current_seed = self.seed + i
            
            # 1. Estraiamo i parametri critici per il loop manuale
            run_params = self.params.copy()
            max_epochs = run_params.get('max_iter', 200)
            early_stop_enabled = run_params.get('early_stopping', False)
            n_iter_no_change = run_params.get('n_iter_no_change', 10)
            tol = run_params.get('tol', 1e-4)
            
            # Forziamo i parametri per il controllo manuale dell'epoca
            run_params['max_iter'] = 1 # così il .fit fa solo un epoca
            run_params['warm_start'] = True # warm_start=True 
            run_params['random_state'] = current_seed
            # Disabilitiamo l'early_stopping interno di sklearn perché viene gestito internamente
            # per poter registrare i dati a ogni step
            run_params['early_stopping'] = False 

            kf = KFold(n_splits=5, shuffle=True, random_state=current_seed)
            results = {
                'accuracies': [], 'y_pred': [], 'y_proba': [], 'y_true': [],
                'losses': [], 'val_scores': [], 'train_scores_per_epoch': [], 'fit_times': []
            }
            # 5-fold cross validation
            for train_index, val_index in kf.split(self.X):
                X_train_full, X_val_fold = self.X.iloc[train_index], self.X.iloc[val_index]
                y_train_full, y_val_fold = self.y.iloc[train_index], self.y.iloc[val_index]
                
                # applico standardizzzione sul training (e poi la applico anche sul test)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_full)
                X_val_scaled = scaler.transform(X_val_fold)

                # Istanzio un MLP con i parametri giusti
                model = MLPClassifier(**run_params)
                
                fold_losses, fold_train_acc, fold_val_acc = [], [], []
                best_val_score = -np.inf
                no_improvement_count = 0
                
                start_time = time.time()
                for epoch in range(max_epochs):
                    model.fit(X_train_scaled, y_train_full)
                    
                    # Registrazione metriche per epoca
                    current_loss = model.loss_
                    current_train_acc = model.score(X_train_scaled, y_train_full)
                    current_val_acc = model.score(X_val_scaled, y_val_fold)
                    
                    fold_losses.append(current_loss)
                    fold_train_acc.append(current_train_acc)
                    fold_val_acc.append(current_val_acc)
                    
                    # LOGICA EARLY STOPPING MANUALE
                    if early_stop_enabled:
                        if current_val_acc > best_val_score + tol:
                            best_val_score = current_val_acc
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                        
                        if no_improvement_count >= n_iter_no_change:
                            # print(f"Early stopping alla riga {epoch}")
                            break
                
                end_time = time.time()

                # Salvataggio dati fold
                results['fit_times'].append(end_time - start_time)
                results['losses'].append(fold_losses)
                results['train_scores_per_epoch'].append(fold_train_acc)
                results['val_scores'].append(fold_val_acc)
                
                # Metriche finali per l'analisi della stabilità (boxplot) e ROC
                y_proba = model.predict_proba(X_val_scaled)[:, 1]
                y_pred = model.predict(X_val_scaled)
                results['accuracies'].append(accuracy_score(y_val_fold, y_pred))
                results['y_proba'].append(y_proba)
                results['y_pred'].append(y_pred)
                results['y_true'].append(y_val_fold)
            
            all_results.append(results)
        return {
            'data' : all_results,  # I dati delle 30 run (accuracies, losses, ecc.)
            'params' : self.params # I parametri originali usati per questo scenario
        }




####################################
#### SOLVER SCENARIO ###############
####################################

# FUNZIONI SOLVER SCENARIO: per ogni scenario (Architettura, Learning Rate, Regolarizzazione), la funzione: 
    # - Itera sui parametri definiti dalla traccia, per quello scenario 
    # - Istanzia la classe per ogni combinazione 
    # - Esegue il ciclo delle 30 run 
    # - Raccoglie i dati per generare i Box plot delle accuratezze e l'analisi della stabilità richiesti

def scenario_architettura_att(X, y):
    results_scenario = {}
    architectures = [
        (50,), (100,), (200,),               
        (50, 25), (100, 50), (200, 100),     
        (100, 50, 25), (200, 100, 50)        
    ]
    activations = ['identity', 'logistic', 'tanh', 'relu']
    
    total_configs = len(architectures) * len(activations)
    count = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
        for arch in architectures:
            for act in activations:
                count += 1
                conf_name = f"Arch:{arch}_Act:{act}"
                
                print("\n" + "="*60)
                print(f" SCENARIO {count}/{total_configs} | {conf_name}")
                print(f"   Layer Nascosti: {arch}")
                print(f"   Attivazione:    {act}")
                print("="*60)
                
                params = {'hidden_layer_sizes': arch, 'activation': act}
                algo = MLPAlgo(X, y, params, n_runs=30, seed=42)
                results_scenario[conf_name] = algo.run_experiment()

    return results_scenario

def scenario_learning_rate_ott(X, y):
    results_scenario = {}
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    policies = ['constant', 'invscaling', 'adaptive']
    solvers = ['adam', 'sgd', 'lbfgs']
    batch_sizes = [32, 64, 128, 256]

    total_configs = len(learning_rates) * len(policies) * len(solvers) * len(batch_sizes)
    count = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for lr in learning_rates:
            for policy in policies:
                for solv in solvers:
                    for batch in batch_sizes:
                        count += 1
                        conf_name = f"LR:{lr}_Pol:{policy}_Solv:{solv}_Batch:{batch}"
                        
                        print("\n" + "="*60)
                        print(f"SCENARIO {count}/{total_configs} | {conf_name}")
                        print(f"   LR Init: {lr} | Policy: {policy}")
                        print(f"   Solver:  {solv} | Batch:  {batch}")
                        print("="*60)
                        
                        params = {
                            'learning_rate_init': lr,
                            'learning_rate': policy,
                            'solver': solv,
                            'batch_size': batch,
                        }
                        
                        algo = MLPAlgo(X, y, params, n_runs=30, seed=42)
                        results_scenario[conf_name] = algo.run_experiment()

    return results_scenario

def scenario_regolarizzazione(X, y):
    results_scenario = {}
    alpha_values = [0.0001, 0.001, 0.01, 0.1]
    validation_splits = [0.1, 0.2, 0.3]
    n_iter_no_change_list = [5, 10, 20]

    total_configs = len(alpha_values) * len(validation_splits) * len(n_iter_no_change_list)
    count = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for alpha in alpha_values:
            for split in validation_splits:
                for n_iter in n_iter_no_change_list:
                    count += 1
                    conf_name = f"Alpha:{alpha}_Split:{split}_IterNoCh:{n_iter}"
                    
                    print("\n" + "="*60)
                    print(f"SCENARIO {count}/{total_configs} | {conf_name}")
                    print(f"  Alpha (L2):      {alpha}")
                    print(f"  Validation Split: {split}")
                    print(f"   Patience (Iter):  {n_iter}")
                    print("="*60)
                    
                    params = {
                        'alpha': alpha,
                        'early_stopping': True,         
                        'validation_fraction': split,
                        'n_iter_no_change': n_iter,
                    }
                    
                    algo = MLPAlgo(X, y, params, n_runs=30, seed=42)
                    results_scenario[conf_name] = algo.run_experiment()

    return results_scenario



############################################
#### VISUALIZZAZIONE GENERALE###############
############################################

# train vs validation: curve di convergenza 
def plot_learning_curves(data_source, scenario_name, fix_epochs=None, plot_loss=False):
    """
    Mostra sempre Train (tratteggiato) e Val (continuo) per ogni scenario.
    Se plot_loss=True, mostra anche la Loss aggregata, usando lo stesso colore di VAL/TRAIN.
    """
    plt.figure(figsize=(12, 7))
    if not isinstance(data_source, dict):
        data_source = {scenario_name: data_source}

    colors = plt.cm.tab10(np.linspace(0, 1, len(data_source)))

    # 1. Calcolo lunghezza massima per il padding
    global_max_len = fix_epochs if fix_epochs else 0
    if not fix_epochs:
        for scenario_obj in data_source.values():
            for run in scenario_obj['data']:
                ref = run['losses'] if plot_loss else run['train_scores_per_epoch']
                for fold in ref:
                    global_max_len = max(global_max_len, len(fold))

    # 2. Loop sugli scenari
    for (name, scenario_obj), color in zip(data_source.items(), colors):
        all_train = []
        all_val = []
        all_loss = []  # Nuovo array per la loss complessiva
        
        for run in scenario_obj['data']:
            if plot_loss:
                all_train.extend(run.get('train_scores_per_epoch', []))
                val_data = run.get('val_scores', run.get('val_losses', run['losses']))
                all_val.extend(val_data)
                all_loss.extend(run.get('losses', []))
            else:
                all_train.extend(run['train_scores_per_epoch'])
                all_val.extend(run['val_scores'])
        
        # Funzione di padding
        def pad_to_max(curves, length):
            padded = []
            for c in curves:
                if len(c) == 0: continue
                last_val = c[-1]
                padded.append(list(c) + [last_val] * (length - len(c)))
            return np.array(padded)

        padded_train = pad_to_max(all_train, global_max_len)
        padded_val = pad_to_max(all_val, global_max_len)
        
        # Calcolo medie e deviazioni standard
        mean_train = np.mean(padded_train, axis=0)
        std_train = np.std(padded_train, axis=0)
        mean_val = np.mean(padded_val, axis=0)
        std_val = np.std(padded_val, axis=0)

        if plot_loss:
            padded_loss = pad_to_max(all_loss, global_max_len)
            mean_loss = np.mean(padded_loss, axis=0)
            std_loss = np.std(padded_loss, axis=0)

        epochs = range(1, global_max_len + 1)
        
        # 3. Plotting
        plt.plot(epochs, mean_val, label=f'VAL - {name}', color=color, lw=2)
        plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, color=color, alpha=0.1)

        plt.plot(epochs, mean_train, linestyle='--', color=color, alpha=0.6, label=f'TRAIN - {name}')

        # LOSS usa lo stesso colore, ma linea puntinata più spessa
        if plot_loss:
            plt.plot(epochs, mean_loss, linestyle=':', color=color, lw=2, label=f'LOSS - {name}')
            plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.05)
    
    metric_label = "Loss" if plot_loss else "Accuracy"
    if plot_loss == True:
        plt.title(f"Learning Curves (Acc-Loss): {scenario_name}", fontsize=14)
    else:
        plt.title(f"Learning Curves (Acc): {scenario_name}", fontsize=14)
    plt.xlabel("Epoche")
    plt.ylabel(metric_label)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
def plot_learning_curves_no_padding(data_source, scenario_name):
    """
    Versione SENZA padding - ogni configurazione mostra la sua lunghezza reale.
    Utile per le configurazioni con early stopping
    """
    plt.figure(figsize=(12, 7))
    if not isinstance(data_source, dict):
        data_source = {scenario_name: data_source}

    colors = plt.cm.tab10(np.linspace(0, 1, len(data_source)))

    for (name, scenario_obj), color in zip(data_source.items(), colors):
        all_train = []
        all_val = []
        
        for run in scenario_obj['data']:
            all_train.extend(run['train_scores_per_epoch'])
            all_val.extend(run['val_scores'])
        
        # Calcola la lunghezza MEDIA per questa specifica configurazione
        mean_len = int(np.mean([len(curve) for curve in all_train]))
        
        # Padding/troncamento SOLO per questa configurazione (non globale)
        def adjust_to_mean(curves, target_len):
            adjusted = []
            for c in curves:
                if len(c) >= target_len:
                    adjusted.append(c[:target_len])
                else:
                    # Padding se troppo corta
                    adjusted.append(list(c) + [c[-1]] * (target_len - len(c)))
            return adjusted
        
        adjusted_train = adjust_to_mean(all_train, mean_len)
        adjusted_val = adjust_to_mean(all_val, mean_len)
        
        mean_train = np.mean(adjusted_train, axis=0)
        std_train = np.std(adjusted_train, axis=0)
        mean_val = np.mean(adjusted_val, axis=0)
        std_val = np.std(adjusted_val, axis=0)
        
        epochs = range(1, mean_len + 1)
        
        # Plot con banda di confidenza
        plt.plot(epochs, mean_val, label=f'VAL - {name}', color=color, linewidth=2)
        plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.2, color=color)
        plt.plot(epochs, mean_train, label=f'TRAIN - {name}', color=color, linestyle='--', linewidth=2, alpha=0.7)
    
    plt.xlabel('Epoche', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Learning Curves: {scenario_name}', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# learning curve (variando la dimensione del dataset)
def plot_learning_curve_dataset_size(scenarios_dict, X, y, title="Learning Curve: Impatto dimensione Dataset"):
    plt.figure(figsize=(12, 8))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Iterazione su ogni scenario contenuto nel dizionario
    for conf_name, scenario_obj in scenarios_dict.items():
        # Recuperiamo i parametri salvati
        model_params = scenario_obj['params'].copy()
        
        # Prepariamo il modello per la learning_curve di sklearn
        # Ci si assicura che non erediti max_iter=1 dalla classe MLPAlgo
        if model_params.get('max_iter') == 1:
            model_params['max_iter'] = 1000
        
        m = MLPClassifier(**model_params)
        
        print(f"Calcolo learning_curve per: {conf_name}...")
        
        # Eseguiamo la learning curve di sistema
        train_sizes, train_scores, test_scores = learning_curve(
            m, X_scaled, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5), 
            scoring='accuracy'
        )
        
        # Calcolo medie e deviazioni standard
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot linea Validation (Continua con pallini)
        line, = plt.plot(train_sizes, test_mean, 'o-', label=f"VAL - {conf_name}", lw=2)
        color = line.get_color()
        
        # Plot linea Training (Tratteggiata, stesso colore, più chiara)
        plt.plot(train_sizes, train_mean, '--', color=color, alpha=0.5, label=f"TRAIN - {conf_name}")
        
        # Area di varianza (Incertezza) per la validazione
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                         alpha=0.1, color=color)

    plt.title(title, fontsize=14)
    plt.xlabel("Numero di campioni nel Training Set", fontsize=12)
    plt.ylabel("Accuracy Score", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# matrici di confusione
def plot_flexible_confusion_matrix(data_source, key=None, title="Matrice di Confusione"):
    """
    key = none => fa la media di tutto
    key not none => configurazione
    In questo modo si può passare o una singola configurazione, o una serie e in quel caso si fa la media
    """
    results = []
    final_title = title

    # 1. Selezione e Aggregazione dei risultati
    if isinstance(data_source, dict):
        if key is not None:
            # CASO SINGOLO: prendiamo solo la chiave specificata
            obj = data_source[key]
            results = obj['data'] if isinstance(obj, dict) and 'data' in obj else obj
            final_title = f"{title}: {key}"
        else:
            # CASO GLOBALE: key è None, cicliamo su TUTTE le configurazioni
            final_title = f"{title}: Media di tutti gli scenari"
            for k in data_source.keys():
                obj = data_source[k]
                # Estraiamo i dati da ogni configurazione e li aggiungiamo alla lista totale
                scenario_data = obj['data'] if isinstance(obj, dict) and 'data' in obj else obj
                results.extend(scenario_data) # extend aggiunge gli elementi della lista
    else:
        # Caso in cui viene passata direttamente una lista di run
        results = data_source
        final_title = title

    # 2. Logica di calcolo media
    all_matrices = []
    # Ora 'results' contiene o le 30 run di una config, o le (30 * N_config) run totali
    for run in results:
        for i in range(len(run['y_true'])):
            cm = confusion_matrix(run['y_true'][i], run['y_pred'][i])
            all_matrices.append(cm)
    
    if not all_matrices:
        print("Nessun dato trovato per generare la matrice.")
        return

    mean_cm = np.mean(all_matrices, axis=0)
    
    # 3. Visualizzazione
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm, display_labels=['Benigno', 'Maligno'])
    disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=ax) 
    ax.set_title(final_title)
    plt.show()

# curve roc

def plot_diagnostic_roc_comparison(diagnostic_data, title="Confronto Curve ROC"):
    """
    Genera un grafico ROC comparativo per le configurazioni chiave.
    Include l'area di confidenza (std dev) per mostrare la stabilità diagnostica.
    """
    plt.figure(figsize=(10, 8))
    mean_fpr = np.linspace(0, 1, 100)
    
    # Palette di colori accesa per distinguere bene le linee
    colors = plt.cm.tab10(np.linspace(0, 1, len(diagnostic_data)))
    
    for (name, results), color in zip(diagnostic_data.items(), colors):
        tprs = []
        aucs = []
        actual_data = results['data']
        
        for run in actual_data:
            for i in range(len(run['y_true'])):
                fpr, tpr, _ = roc_curve(run['y_true'][i], run['y_proba'][i])
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(auc(fpr, tpr))
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        plt.plot(mean_fpr, mean_tpr, color=color,
                 label=f'{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
                 lw=2.5, alpha=0.9)
        
        # Area di deviazione standard (trasparente)
        std_tpr = np.std(tprs, axis=0)
        plt.fill_between(mean_fpr, 
                         np.maximum(mean_tpr - std_tpr, 0), 
                         np.minimum(mean_tpr + std_tpr, 1), 
                         color=color, alpha=0.1)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Casuale', alpha=0.5)
    
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificità)')
    plt.ylabel('True Positive Rate (Sensibilità)')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# report temporale
def print_timing_report(results):
    # Appiattiamo tutti i tempi di tutti i fold
    all_times = [t for r in results for t in r['fit_times']]
    
    print("-" * 30)
    print("REPORT TEMPI DI ESECUZIONE")
    print("-" * 30)
    print(f"Tempo medio per Fold:   {np.mean(all_times):.4f} s")
    print(f"Tempo totale (30 run):  {np.sum(all_times):.2f} s")
    print(f"Deviazione Standard:    {np.std(all_times):.4f} s")
    print("-" * 30)

# box plot
def plot_flexible_boxplot(data_source, title="Confronto Performance"):
    """
    data_source: può essere una singola lista di risultati 
                 o un dizionario {'NomeScenario': lista_risultati}
    """
    data_to_plot = []
    labels = []

    # 1. Gestione flessibile dell'input
    if isinstance(data_source, dict):
        # Caso SCENARI: prendiamo medie per ogni configurazione
        for name, results in data_source.items():
            run_accs = [np.mean(run['accuracies']) for run in results]
            data_to_plot.append(run_accs)
            labels.append(name)
    else:
        # Caso SINGOLO (come quello fatto finora)
        run_accs = [np.mean(run['accuracies']) for run in data_source]
        data_to_plot.append(run_accs)
        labels.append("Configurazione Base")

    # 2. Plotting
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True,
                medianprops={'color': 'orange', 'linewidth': 2},
                boxprops={'facecolor': 'lightblue'})
    
    plt.title(title)
    plt.ylabel('Accuracy Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# barchar
def plot_validation_split_barchart(split_data, title="Impatto del Validation Split"):
    """
    Bar chart con media + std dev per validation_split.
    """
    # Estrai statistiche
    labels = []
    means = []
    stds = []
    
    # Ordina per valore di split (0.1, 0.2, 0.3)
    sorted_keys = sorted(split_data.keys(), key=lambda x: float(x.split('=')[1]))
    
    for label in sorted_keys:
        accuracies = split_data[label]
        labels.append(label)
        means.append(np.mean(accuracies))
        stds.append(np.std(accuracies))
    
    # Crea il grafico
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, 
                   capsize=7, alpha=0.7, 
                   color=['#3498db', '#e74c3c', '#2ecc71'],
                   edgecolor='black', linewidth=1.5)
    
    # Aggiungi valori sulle barre
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.002, f'{mean:.4f}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formattazione
    ax.set_xlabel('Validation Split', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy Media', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim([0.94, 1.0])  # Zoom sulla zona rilevante
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()




####################################
#### LOGGER ########################
####################################

# CLASSE LOGGER, da usare dopo lo scenarioscenario_architettura_att
class Logger:
    """
    Genera un file txt unico che contiene: 
    per ogni configurazione: 
        - Chiave della configurazione (che avrà dentro i dettagli della configurazione)
        - Accuracy per epoca
        - Tempi di training
        - Loss function
        - Curve di apprendimento
        - Acc per fold 
    Valori medi altrimenti diventa lunghissimo
    """
    def __init__(self):
        pass
    def generate_txt(self, filepath, scenario_results):
        """
            scenario_results è il Dict = {chiave_conf : [Dict]} 
        """
        with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("REPORT SPERIMENTALE MLP - ANALISI SCENARI\n")
                    f.write("="*60 + "\n\n")

                    for conf_name, scenario_obj in scenario_results.items():
                        runs = scenario_obj['data']
                        f.write(f"CONFIGURAZIONE: {conf_name}\n")
                        f.write("-" * 30 + "\n")

                        # 1. Estrazione Accuratezza (Acc per fold e per run)
                        # Calcoliamo la media dei 5 fold per ogni run
                        run_accuracies = [np.mean(run['accuracies']) for run in runs]
                        all_fold_accuracies = [acc for run in runs for acc in run['accuracies']]

                        f.write(f"[ACCURATEZZA]\n")
                        f.write(f"- Media Globale (30 run): {np.mean(run_accuracies):.4f}\n")
                        f.write(f"- Deviazione Standard:      {np.std(run_accuracies):.4f}\n")
                        f.write(f"- Range (Min - Max):       [{np.min(run_accuracies):.4f} - {np.max(run_accuracies):.4f}]\n")
                        f.write(f"- Dettaglio per Fold (tutti): {all_fold_accuracies[:10]}... (mostrati primi 10)\n\n")

                        # 2. Tempi di training
                        all_times = [t for run in runs for t in run['fit_times']]
                        f.write(f"[TEMPI DI TRAINING]\n")
                        f.write(f"- Tempo medio per Fold:    {np.mean(all_times):.4f} s\n")
                        f.write(f"- Tempo totale config:     {np.sum(all_times):.2f} s\n\n")

                        # 3. Analisi Curve (Loss e Validation Accuracy)
                        # Troviamo la lunghezza minima delle curve per fare la media
                        all_losses = [l for run in runs for l in run['losses']]
                        min_epochs = min(len(l) for l in all_losses)
                        
                        avg_loss_final = np.mean([l[min_epochs-1] for l in all_losses])
                        
                        f.write(f"[CURVE DI APPRENDIMENTO]\n")
                        f.write(f"- Epoche medie prima dello stop: {np.mean([len(l) for l in all_losses]):.1f}\n")
                        f.write(f"- Loss finale media:              {avg_loss_final:.6f}\n")
                        
                        if runs[0]['val_scores']: # Se abbiamo salvato i validation scores
                            all_val = [v for run in runs for v in run['val_scores']]
                            avg_val_final = np.mean([v[min_epochs-1] for v in all_val])
                            f.write(f"- Val Accuracy finale media:      {avg_val_final:.4f}\n")

                        f.write("\n" + "="*60 + "\n\n")

        print(f"Report generato con successo in: {filepath}")



###############################################
#### FUNZIONI AUSILIARIE ######################
#### ESTRAZIONE DATI - VISUALIZZAZIONE ########
###############################################

## FUNZIONI AUSILIARE-SCENARIO 1 ##

# risultati sintesi: differenziando solo per numero di layer e attivazione
def get_detailed_complexity_data(results_s1):
    """
    Raggruppa i risultati incrociando numero di layer e attivazione.
    Restituisce un dizionario compatibile con plot_flexible_boxplot.
    """
    grouped_data = {}
    
    # Ordine desiderato per una visualizzazione pulita nel plot
    layer_map = {1: "Singolo", 2: "Doppio", 3: "Triplo"}
    
    for conf_name, scenario_obj in results_s1.items():
        # Parsing: "Arch:(50,)_Act:relu" -> arch=(50,), act="relu"
        arch_str = conf_name.split("Arch:")[1].split("_")[0]
        activation = conf_name.split("Act:")[1]
        
        num_layers = len(eval(arch_str))
        layer_label = layer_map.get(num_layers, f"{num_layers} Layer")
        
        # Creiamo la chiave composta (es. "Singolo - relu")
        composite_key = f"{layer_label} - {activation}"
        
        # Estraiamo le run
        runs = scenario_obj['data'] if isinstance(scenario_obj, dict) and 'data' in scenario_obj else scenario_obj
        
        if composite_key not in grouped_data:
            grouped_data[composite_key] = []
        
        # Aggiungiamo le run alla chiave specifica
        grouped_data[composite_key].extend(runs)
            
    return grouped_data
# risultati single layer-Relu, raggruppando per numero di neuroni
def get_neurons_impact_data(results_s1):
    """
    Filtra solo le configurazioni con un singolo layer e attivazione ReLU,
    raggruppandole per numero di neuroni (50, 100, 200).
    """
    neurons_data = {}
    
    # Definiamo i target che vogliamo isolare (singolo layer con ReLU)
    # Le chiavi originali sono nel formato "Arch:(N,)_Act:relu"
    target_neurons = [50, 100, 200]
    
    for conf_name, scenario_obj in results_s1.items():
        # Verifichiamo se l'attivazione è relu
        if "_Act:relu" in conf_name:
            # Estraiamo l'architettura per contare i layer e i neuroni
            arch_str = conf_name.split("Arch:")[1].split("_")[0]
            layers = eval(arch_str)
            
            # Filtriamo: deve avere solo 1 layer e il numero di neuroni deve essere nei target
            if len(layers) == 1 and layers[0] in target_neurons:
                label = f"Single-ReLU-{layers[0]} Neuroni"
                
                # Estraiamo le run
                runs = scenario_obj['data'] if isinstance(scenario_obj, dict) and 'data' in scenario_obj else scenario_obj
                
                # Salviamo nel dizionario finale
                neurons_data[label] = runs
                
    # Opzionale: ordiniamo il dizionario per numero di neuroni crescente
    sorted_keys = sorted(neurons_data.keys(), key=lambda x: int(x.split('-')[2].split()[0]))
    return {k: neurons_data[k] for k in sorted_keys}
# risultati data un'architettura
def filter_results_by_arch(all_results, target_arch_str):
    """
    Filtra il dizionario dei risultati per una specifica architettura.
    Esempio target_arch_str: "(100, 50)"
    """
    filtered_data = {}
    
    for key, value in all_results.items():
        # Verifica se la stringa dell'architettura è presente nella chiave
        # Ad esempio se la chiave è "Arch: (100, 50) - Act: relu"
        if target_arch_str in key:
            # Puliamo il nome per la legenda (prendiamo solo l'attivazione)
            # "Arch: (100, 50) - Act: relu" -> "relu"
            activation_name = key.split("Act: ")[-1]
            filtered_data[activation_name] = value
            
    return filtered_data
# estrae i risultati delle due architetture "estreme", usato per analizzare overfitting
def extract_comparison_overfitting(all_results):
    """
    Estrae le due configurazioni chiave per mostrare l'overfitting:
    1. Single Layer ReLU (50) -> Modello equilibrato
    2. Triple Layer ReLU (200, 100, 50) -> Modello complesso propenso all'overfitting
    """
    comparison_data = {}
    
    # Chiavi tipiche basate sulla struttura del tuo dizionario
    target_keys = {
        "Single-ReLU-50": "Arch: (50,) - Act: relu",
        "Triple-ReLU-200": "Arch: (200, 100, 50) - Act: relu"
    }
    
    for label, full_key in target_keys.items():
        if full_key in all_results:
            comparison_data[label] = all_results[full_key]
        else:
            # Ricerca parziale nel caso le stringhe siano leggermente diverse
            for actual_key in all_results.keys():
                if ("(50,)" in actual_key and "relu" in actual_key) and label == "Single-ReLU-50":
                    comparison_data[label] = all_results[actual_key]
                elif ("(200, 100, 50)" in actual_key and "relu" in actual_key) and label == "Triple-ReLU-200":
                    comparison_data[label] = all_results[actual_key]
                    
    return comparison_data
# raggruppa i risultati per architettura
def extract_arch_tradeoff_data(all_results):
    """
    Raggruppa i risultati per Architettura.
    Estrae l'accuratezza per le barre e il tempo medio per la linea.
    """
    arch_groups = {}
    
    for key, value in all_results.items():
        # Parsing chiave: "Arch:(50,)_Act:relu" -> "(50,)"
        arch_name = key.split("_Act:")[0].replace("Arch:", "")
        
        if arch_name not in arch_groups:
            arch_groups[arch_name] = {'times': [], 'accuracies': []}
        
        runs = value['data']
        for run in runs:
            # Tempo medio per singola run (media dei tempi dei 5 fold)
            avg_run_time = np.mean(run['fit_times'])
            # Accuratezza media della run (media dei 5 fold)
            avg_run_acc = np.mean(run['accuracies'])
            
            arch_groups[arch_name]['times'].append(avg_run_time)
            arch_groups[arch_name]['accuracies'].append(avg_run_acc)
            
    return arch_groups
# estrae i risultati delle configurazioni usate per la curva ROC
def extract_diagnostic_configs(all_results):
    """
    Estrae le configurazioni chiave per l'analisi ROC e Matrice di Confusione:
    - (50,) con ReLU (La migliore)
    - (50,) con Logistic (Il confronto di efficienza)
    - (50,) con Identity (Il confronto di linearità)
    - (200, 100, 50) con ReLU (Il confronto sulla profondità)
    """
    target_keys = {
        "ReLU - (50,)": "Arch:(50,)_Act:relu",
        "Logistic - (50,)": "Arch:(50,)_Act:logistic",
        "Identity - (50,)": "Arch:(50,)_Act:identity",
        "ReLU - (200, 100, 50)": "Arch:(200, 100, 50)_Act:relu"
    }
    
    diagnostic_data = {}
    for label, key in target_keys.items():
        if key in all_results:
            diagnostic_data[label] = all_results[key]
        else:
            print(f"Attenzione: Chiave {key} non trovata nei risultati.")
            
    return diagnostic_data
# plot a doppio asse: acc e tempo, per architettura
def plot_arch_complexity_tradeoff(arch_results):
    """
    Grafico a doppio asse:
    - Barre: Accuratezza (la nostra 'fitness' del modello)
    - Linea: Tempo medio di esecuzione per run
    """
    order = [
        "(50,)", "(100,)", "(200,)", 
        "(50, 25)", "(100, 50)", "(200, 100)", 
        "(100, 50, 25)", "(200, 100, 50)"
    ]
    
    sorted_keys = [k for k in order if k in arch_results]
    x_labels = [k.replace(" ", "") for k in sorted_keys]
    
    # Calcolo medie e deviazioni standard per le barre di errore
    avg_accs = [np.mean(arch_results[k]['accuracies']) for k in sorted_keys]
    std_accs = [np.std(arch_results[k]['accuracies']) for k in sorted_keys]
    avg_times = [np.mean(arch_results[k]['times']) for k in sorted_keys]

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Asse 1: Accuratezza (Barre Blu) ---
    color_acc = '#4A90E2' # Un blu professionale
    ax1.set_xlabel('Configurazione Layer (Architettura)', fontsize=12)
    ax1.set_ylabel('Accuratezza Media (Validation)', color=color_acc, fontweight='bold', fontsize=12)
    
    # Disegniamo le barre con le linee di errore (std dev) per mostrare la stabilità
    ax1.bar(x_labels, avg_accs, yerr=std_accs, color=color_acc, alpha=0.7, 
            capsize=5, label="Accuratezza Media", edgecolor='navy')
    
    ax1.tick_params(axis='y', labelcolor=color_acc)
    # Range stretto per vedere le differenze tra 0.94 e 0.98
    ax1.set_ylim(min(avg_accs) - 0.02, 1.0) 
    plt.xticks(rotation=20)

    # --- Asse 2: Tempo (Linea Rossa) ---
    ax2 = ax1.twinx()
    color_time = '#D0021B' # Rosso scuro
    ax2.set_ylabel('Tempo Medio per Run (s)', color=color_time, fontweight='bold', fontsize=12)
    ax2.plot(x_labels, avg_times, color=color_time, marker='s', markersize=8, 
             linewidth=3, label="Tempo medio")
    ax2.tick_params(axis='y', labelcolor=color_time)

    plt.title("Scenario 1: Accuratezza vs Costo Temporale per Architettura", pad=25, fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Legenda combinata
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.tight_layout()
    plt.show()



## FUNZIONI AUSILIARE-SCENARIO 2 ##

# raggruppa i risultati per solvere
def extract_solver_comparison(all_results_s2):
    """
    Raggruppa tutte le run per solver.
    Media su learning_rate_init, learning_rate_policy, e batch_size.
    
    Formato chiave: "LR:0.001_Pol:adaptive_Solv:adam_Batch:64"
    Restituisce: {'Adam': [runs], 'SGD': [runs], 'LBFGS': [runs]}
    """
    solver_groups = {}
    
    for key, value in all_results_s2.items():
        # Estrai il solver dalla chiave
        if "_Solv:adam_" in key:
            solver_name = "Adam"
        elif "_Solv:sgd_" in key:
            solver_name = "SGD"
        elif "_Solv:lbfgs_" in key:
            solver_name = "LBFGS"
        else:
            print(f"Solver non riconosciuto in chiave: {key}")
            continue
        
        if solver_name not in solver_groups:
            solver_groups[solver_name] = []
        
        # Estrai le run e aggiungi al gruppo
        solver_groups[solver_name].extend(value['data'])
    
    return solver_groups
# stampa le configurazioni peggiori per sgd
def debug_sgd_failures(all_results_s2):
    """Trova le config SGD che falliscono miseramente"""
    bad_configs = []
    
    for key, value in all_results_s2.items():
        if "_Solv:sgd_" in key:
            for run in value['data']:
                avg_acc = np.mean(run['accuracies'])
                if avg_acc < 0.60:
                    bad_configs.append((key, avg_acc))
    
    # Ordina per accuracy
    bad_configs.sort(key=lambda x: x[1])
    
    print("Configurazioni SGD disastrose (accuracy < 0.60):")
    for config, acc in bad_configs[:10]:  # Prime 10
        print(f"  {config}: {acc:.3f}")
    
    return bad_configs
# estrazioni dati sgd variando LR, fisso il resto
def extract_sgd_lr_curves(all_results_s2):
    """
    Estrae curve di apprendimento per SGD con diversi LR.
    Fissa: Policy=adaptive, Batch=64
    Varia: LR = [0.0001, 0.001, 0.01, 0.1]
    """
    target_configs = {
        "SGD-LR=0.0001": "LR:0.0001_Pol:adaptive_Solv:sgd_Batch:64",
        "SGD-LR=0.001": "LR:0.001_Pol:adaptive_Solv:sgd_Batch:64",
        "SGD-LR=0.01": "LR:0.01_Pol:adaptive_Solv:sgd_Batch:64",
        "SGD-LR=0.1": "LR:0.1_Pol:adaptive_Solv:sgd_Batch:64"
    }
    
    sgd_lr_data = {}
    
    for label, key in target_configs.items():
        if key in all_results_s2:
            sgd_lr_data[label] = all_results_s2[key]
        else:
            print(f"Chiave non trovata: {key}")
    
    return sgd_lr_data
# estrazione dati variando batch size, resto fisso
def extract_batch_size_impact(all_results_s2):
    """
    Filtra i risultati per: Solver=Adam, LR=0.001, Policy=adaptive.
    Varia il Batch Size per analizzare l'impatto sulla stabilità e velocità.
    """
    batch_data = {}
    # Parametri fissi come da tua richiesta
    # Nota: Assicurati che le stringhe corrispondano esattamente ai nomi delle chiavi generate in scenario_learning_rate_ott
    target_lr = "LR:0.001"
    target_pol = "Pol:adaptive"
    target_solv = "Solv:adam"
    
    # Cerchiamo nel dizionario dei risultati
    for key, value in all_results_s2.items():
        if target_lr in key and target_pol in key and target_solv in key:
            # Estraiamo il batch size dalla chiave (es: "Batch:64")
            batch_label = key.split("_Batch:")[1] 
            
            # Calcoliamo le metriche medie per questa configurazione (media delle 30 run)
            runs = value['data']
            run_accs = [np.mean(run['accuracies']) for run in runs]
            run_times = [np.mean(run['fit_times']) for run in runs]
            
            batch_data[int(batch_label)] = {
                'accuracies': run_accs,
                'mean_acc': np.mean(run_accs),
                'std_acc': np.std(run_accs),
                'mean_time': np.mean(run_times)
            }
            
    # Ordiniamo per dimensione del batch (chiave intera)
    sorted_batches = sorted(batch_data.keys())
    return {b: batch_data[b] for b in sorted_batches}
# plot a due assi: acc (bar plot) e andamento temporale, varinado batch
def plot_batch_size_tradeoff(batch_results):
    """
    Bar Plot per Accuracy + Linea per Tempo di Training
    """
    batches = [str(b) for b in batch_results.keys()]
    means = [data['mean_acc'] for data in batch_results.values()]
    stds = [data['std_acc'] for data in batch_results.values()]
    times = [data['mean_time'] for data in batch_results.values()]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Asse 1: Accuratezza (Barre) ---
    color_acc = '#2ecc71' # Verde smeraldo
    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy Media', color=color_acc, fontsize=12, fontweight='bold')
    bars = ax1.bar(batches, means, yerr=stds, color=color_acc, alpha=0.6, 
                   capsize=7, label='Accuracy (± std)', edgecolor='black')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    
    # Zoom per apprezzare le differenze millesimali
    ax1.set_ylim([min(means) - 0.01, max(means) + 0.01])

    # --- Asse 2: Tempo (Linea) ---
    ax2 = ax1.twinx()
    color_time = '#e74c3c' # Rosso
    ax2.set_ylabel('Tempo Medio per Run (s)', color=color_time, fontsize=12, fontweight='bold')
    ax2.plot(batches, times, color=color_time, marker='o', linewidth=2.5, 
             markersize=8, label='Tempo di esecuzione')
    ax2.tick_params(axis='y', labelcolor=color_time)

    plt.title("Impatto del Batch Size: Accuratezza vs Efficienza\n(Adam, LR=0.001, Adaptive Policy)", 
              fontsize=14, pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    
    fig.tight_layout()
    plt.show()
# estrai i dati delle configurazioni passate come argomento, usato per le roc
def extract_roc_data_best_solvers(all_results, solver_names):
    """
    Estrae i dati ROC per i solver selezionati (configurazioni migliori).
    
    Parametri
    ----------
    all_results : dict
        Dizionario completo dei risultati (es. output di scenario_learning_rate_ott)
    solver_names : list[str]
        Lista dei solver da estrarre (es. ['adam', 'sgd', 'lbfgs'])
    
    Ritorna
    -------
    diagnostic_data : dict
        Dizionario pronto per plot_diagnostic_roc_comparison
    """
    diagnostic_data = {}

    for solver in solver_names:
        # Filtriamo tutte le configurazioni che usano quel solver
        matching_configs = {
            k: v for k, v in all_results.items()
            if f"Solv:{solver}" in k
        }

        if not matching_configs:
            print(f"Nessuna configurazione trovata per solver: {solver}")
            continue

        # Scegliamo la configurazione MIGLIORE in base all'accuracy media
        best_key = None
        best_score = -np.inf

        for key, scenario_obj in matching_configs.items():
            runs = scenario_obj['data']
            mean_acc = np.mean([np.mean(run['accuracies']) for run in runs])

            if mean_acc > best_score:
                best_score = mean_acc
                best_key = key

        diagnostic_data[f"Solver {solver.upper()}"] = matching_configs[best_key]

        print(f" Solver {solver.upper()} → Config migliore: {best_key}")
        print(f" Accuracy media: {best_score:.4f}")

    return diagnostic_data



## FUNZIONI AUSILIARE-SCENARIO 3 ##

# raggruppo i dati per alpha, su tutto
def extract_alpha_for_boxplot(all_results_s3):
    """
    Raggruppa le RUN intere per valore di Alpha.
    Restituisce un dizionario compatibile con plot_flexible_boxplot:
    {'Alpha:0.0001': [lista di 270 run], 'Alpha:0.001': [...]}
    (270 run = 30 run * 3 validation_splits * 3 n_iter_no_change)
    """
    alpha_groups = {}
    
    for key, value in all_results_s3.items():
        # Estraiamo la parte della stringa "Alpha:X.XXXX"
        # La tua chiave è: "Alpha:0.0001_Split:0.1_IterNoCh:5"
        alpha_label = key.split("_")[0] 
        
        if alpha_label not in alpha_groups:
            alpha_groups[alpha_label] = []
        
        # Estraiamo l'oggetto 'data' (che contiene le 30 run di quella config)
        # e lo aggiungiamo al gruppo alpha corrispondente
        runs = value['data']
        alpha_groups[alpha_label].extend(runs) # Usiamo extend per unire le liste di run
            
    return alpha_groups
# raggruppo i dati per alpha, su una configurazione fissa
def extract_alpha_for_boxplot_fixed(all_results_s3):
    """
    Isola l'effetto di Alpha fissando Split=0.2 e IterNoCh=10
    """
    alpha_groups = {}
    
    for key, value in all_results_s3.items():
        if "_Split:0.2_" in key and "_IterNoCh:10" in key:
            alpha_label = key.split("_")[0]  # "Alpha:0.0001"
            
            if alpha_label not in alpha_groups:
                alpha_groups[alpha_label] = []
            
            alpha_groups[alpha_label].extend(value['data'])
    
    return alpha_groups
# raggruppo i dati per n_iter_no_change, su tutto
def extract_early_stopping_all(all_results_s3):
    """
    Raggruppa TUTTE le run per valore di n_iter_no_change.
    Media su alpha e validation_split.
    Restituisce formato compatibile con plot_learning_curves.
    """
    n_iter_groups = {}
    
    for key, value in all_results_s3.items():
        # Estrai "IterNoCh:5", "IterNoCh:10", "IterNoCh:20"
        iter_label = key.split("_IterNoCh:")[1]  # "5", "10", "20"
        iter_label = f"n_iter={iter_label}"
        
        if iter_label not in n_iter_groups:
            n_iter_groups[iter_label] = {'data': [], 'params': {}}
        
        n_iter_groups[iter_label]['data'].extend(value['data'])
    
    return n_iter_groups
# raggruppo i dati per validation_split, su tutto
def extract_validation_split_comparison(all_results_s3):
    """
    Raggruppa le run per validation_split.
    Media su alpha e n_iter_no_change.
    Restituisce: {label: [lista di run con accuracies]}
    """
    split_groups = {}
    
    for key, value in all_results_s3.items():
        # Estrai "Split:0.1", "Split:0.2", "Split:0.3"
        split_value = key.split("_Split:")[1].split("_")[0]  # "0.1", "0.2", "0.3"
        split_label = f"Split={split_value}"
        
        if split_label not in split_groups:
            split_groups[split_label] = []
        
        # Estrai le accuracies finali da ogni run
        for run in value['data']:
            split_groups[split_label].extend(run['accuracies'])
    
    return split_groups
# raggruppo i dati per alpha, su tutto, per le curve roc
def extract_alpha_for_roc(all_results_s3):
    """
    Estrae i dati diagnostici (y_true, y_proba) raggruppandoli per valore di alpha.
    Aggrega su validation_split e n_iter_no_change.
    
    Output compatibile con plot_diagnostic_roc_comparison:
    {
        'Alpha=0.0001': {'data': [run, run, ...]},
        'Alpha=0.001':  {'data': [...]},
        ...
    }
    """
    alpha_groups = {}

    for key, scenario_obj in all_results_s3.items():
        # Chiave originale: "Alpha:0.0001_Split:0.2_IterNoCh:10"
        alpha_value = key.split("_")[0].split(":")[1]  # "0.0001"
        alpha_label = f"Alpha={alpha_value}"

        if alpha_label not in alpha_groups:
            alpha_groups[alpha_label] = {'data': []}

        # Estraiamo le 30 run di questa configurazione
        runs = scenario_obj['data']

        # Aggiungiamo tutte le run al gruppo alpha
        alpha_groups[alpha_label]['data'].extend(runs)

    return alpha_groups



####################################
#### MAIN ##########################
####################################
if __name__ == "__main__":
    X, y = load_dataset("Breast_Wisconsin_Dataset.csv")
    
    mode = 1 # 0 => run exsperiment, Altrimenti => plotting
    
    # ESPERIMENTI-RUN
    if mode == 0:
        # Scenario 1
        results_all_s1= scenario_architettura_att(X,y)
        with open("results_all_s1.pkl", "wb") as f: 
            pickle.dump(results_all_s1, f) 
        loggerS1 = Logger()
        loggerS1.generate_txt("report_esperimenti_scenario1.txt", results_all_s1) 

        # Scenario 2
        results_all_s2= scenario_learning_rate_ott(X,y)
        with open("results_all_s2.pkl", "wb") as f: 
            pickle.dump(results_all_s2, f) 
        loggerS2 = Logger()
        loggerS2.generate_txt("report_esperimenti_scenario2.txt", results_all_s2)     

        # Scenario 3
        results_all_s3= scenario_regolarizzazione(X,y)
        with open("results_all_s3.pkl", "wb") as f: 
            pickle.dump(results_all_s3, f) 
        loggerS3 = Logger()
        loggerS3.generate_txt("report_esperimenti_scenario3.txt", results_all_s3)
    # PLOTTING
    else: 
        # PLOTS - SCENARIO 1
        with open("results_all_s1.pkl", "rb") as f: 
            results_all_s1 = pickle.load(f)    
        # boxplot, per arch/attivazione
        result_1 = get_detailed_complexity_data(results_all_s1)
        plot_flexible_boxplot(result_1, "Confronto Performance: Architettura vs Attivazione")
        # boxplot, per single-Relu
        result_2 = get_neurons_impact_data(results_all_s1)
        plot_flexible_boxplot(result_2, "Confronto Performance: Best Arch-Attivazione vs # neuroni")    
        # learning curves per attivazione (arch fissa)
        result_3 = filter_results_by_arch(results_all_s1, "(50,)")
        plot_learning_curves(result_3, "Confronto Attivazioni su Architettura (50, )")
        # learning curve: singolo vs triplo
        result_4 = extract_comparison_overfitting(results_all_s1)
        plot_learning_curves(result_4, "Confronto Overfitting: Singolo vs Triplo Layer")
        # Acc vs Tempo variando architettura
        result_5 = extract_arch_tradeoff_data(results_all_s1) 
        plot_arch_complexity_tradeoff(result_5)
        # Curva ROC casi significativi
        result_6 = extract_diagnostic_configs(results_all_s1)
        plot_diagnostic_roc_comparison(result_6)
        # Matrice di confusione per la migliore
        miglior_config = "Arch:(50,)_Act:relu"
        plot_flexible_confusion_matrix(
            data_source=results_all_s1, 
            key=miglior_config, 
            title="Analisi Diagnostica - Configurazione Ottimale"
        )
    
        # PLOTS - SCENARIO 2
        with open("results_all_s2.pkl", "rb") as f: 
            results_all_s2 = pickle.load(f)

        # boxplot dividendo per solver
        results2_1 = extract_solver_comparison(results_all_s2)
        plot_flexible_boxplot(results2_1, "Confronto Performance: Solver (Adam vs SGD vs LBFGS)")
        debug_sgd_failures(results_all_s2)
        # learning curves variando LR, resto fisso
        results2_2 = extract_sgd_lr_curves(results_all_s2)
        plot_learning_curves(results2_2, "SGD: Impatto del Learning Rate (Policy=adaptive, Batch=64)", plot_loss=True) 
        # boxplot per policies, resto fisso
        results2_2_2 = {
            "Constant": results_all_s2["LR:0.01_Pol:constant_Solv:sgd_Batch:64"]['data'],
            "Invscaling": results_all_s2["LR:0.01_Pol:invscaling_Solv:sgd_Batch:64"]['data'],
            "Adaptive": results_all_s2["LR:0.01_Pol:adaptive_Solv:sgd_Batch:64"]['data']
        }
        plot_flexible_boxplot(
            data_source=results2_2_2, 
            title="Stabilità delle Learning Rate Policies (Solver: SGD, LR: 0.01)"
        )
        # acc vs tempo, variando batch, resto fisso
        results2_3 = extract_batch_size_impact(results_all_s2)
        plot_batch_size_tradeoff(results2_3)
        # curva roc per solver, conf migliori
        results2_4 = extract_roc_data_best_solvers(results_all_s2, solver_names=['adam', 'sgd', 'lbfgs'])
        plot_diagnostic_roc_comparison(results2_4, title="Confronto ROC - Solver Migliori")
        # matrice di confusione, conf migliore
        miglior_config_2 = "LR:0.001_Pol:adaptive_Solv:lbfgs_Batch:64"
        plot_flexible_confusion_matrix(
            data_source=results_all_s2, 
            key=miglior_config_2, 
            title="Analisi Diagnostica - Configurazione Ottimale"
        )

        

        # PLOTS - SCENARIO 3
        with open("results_all_s3.pkl", "rb") as f: 
            results_all_s3 = pickle.load(f) 

        # boxplot raggruppando per alpha
        results3_1 = extract_alpha_for_boxplot(results_all_s3)
        plot_flexible_boxplot(results3_1, "Confronto Performance: valore di Alpha")
        # boxplot raggruppando per alpha, resto fisso
        results3_1_1 = extract_alpha_for_boxplot_fixed(results_all_s3)
        plot_flexible_boxplot(results3_1_1, "Impatto di Alpha isolato (Split=0.2, IterNoCh=10)")
        # learning curve: variando n_iter_no_change, resto fisso
        results3_2 = extract_early_stopping_all(results_all_s3)
        plot_learning_curves_no_padding(results3_2, "Confronto n_iter_no_change, fissando altri parametri")
        # barplot variando validation split 
        results3_3 = extract_validation_split_comparison(results_all_s3)
        plot_validation_split_barchart(results3_3, "Impatto del Validation Split")
        # curve roc, variando alpha
        result4 = extract_alpha_for_roc(results_all_s3)
        plot_diagnostic_roc_comparison(result4)
        # matrice di confusione migliore
        miglior_config3 = "Alpha:0.1_Split:0.2_IterNoCh:5"
        plot_flexible_confusion_matrix(
            data_source=results_all_s3, 
            key=miglior_config3, 
            title="Analisi Diagnostica - Configurazione Ottimale"
        )
        
        
        # PLOT SCENARIO FINALE
        # Dizionario con la configurazione ottimale
        ideal_config = {
            "Single50_ReLU": {
                "params": {
                    "hidden_layer_sizes": (50,),
                    "activation": "relu",
                    "solver": "lbfgs",        # oppure "lbfgs" se vuoi LBFGS
                    "alpha": 0.1,
                    "learning_rate_init": 0.001,
                    "max_iter": 200,         # numero di epoche
                    "early_stopping": True,
                    "n_iter_no_change": 5,
                    "validation_fraction": 0.2,
                    "random_state": 42
                }
            }
        }
        # effetto dimensione dataset
        plot_learning_curve_dataset_size(ideal_config, X, y, title="Learning Curve - Configurazione Ideale")