"""
Genetic Algorithm for Feature Selection on DARWIN Dataset
==========================================================

Tesina 2 - Analisi Algoritmo Genetico come Feature Selection

HINT: Questo script fornisce una struttura base. Dovrete:
- Completare le funzioni con la logica appropriata
- Implementare i metodi di selezione richiesti
- Gestire correttamente i parametri e le metriche
"""

import random
import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Callable
import pickle
# HINT: Considerate quali altre librerie potrebbero essere utili per 
# la valutazione delle correlazioni e la visualizzazione

# =============================================================================
# CONFIGURAZIONE E SEED
# =============================================================================
SEED = 42  # HINT: Usare lo stesso seed per garantire riproducibilità


# =============================================================================
# CARICAMENTO DATASET
# =============================================================================
def load_darwin_dataset(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carica il dataset DARWIN e gestisce eventuali missing values.
    
    HINT: 
    - La prima colonna è l'ID, l'ultima è la classe (P/H)
    - Le colonne intermedie sono le 450 features (25 task × 18 features)
    - Considerate diverse strategie per i missing values
    
    Returns:
        X: DataFrame delle features
        y: Series delle classi
    """
    # TODO: Implementare il caricamento
    df= pd.read_csv(filepath)
    
    X=df.iloc[:,1:-1]
    y=df.iloc[:,-1]
    y = y.map({'H': 0, 'P': 1})

    print(X.shape)

    missing_percent = X.isna().mean()*100
   
    # isna -> dataframe boleano true se il valore non c'è false altrimenti, mean per fare la media dei valore true e false su ogni colonna
    # *100 si ottiene la percentuale dei valori mancanti per ogni colonna
    
    print(missing_percent[missing_percent>0]) 
    print(X.isna().sum().sum())
    cols_to_drop=missing_percent[missing_percent>50].index
    X=X.drop(columns=cols_to_drop)
    X=X.fillna(X.median())
   

    return X,y


# =============================================================================
# PRE-CALCOLO CORRELAZIONI
# =============================================================================
def precompute_correlations(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-calcola le correlazioni feature-classe e feature-feature.
    Si chiama una sola volta prima di eseguire il GA. Per ottimizzare i tempi di esecuzione
    
    Returns:
        r_cf: array delle correlazioni |corr(feature_i, classe)|
        r_ff: matrice delle correlazioni |corr(feature_i, feature_j)|
    """
    # Correlazione feature-classe
    r_cf = np.array([np.abs(np.corrcoef(X.iloc[:, i], y)[0, 1]) 
                     for i in range(X.shape[1])])
    r_cf = np.nan_to_num(r_cf, nan=0.0)  # gestione NaN
    
    # Correlazione feature-feature
    r_ff = np.abs(X.corr().values)
    
    return r_cf, r_ff


# =============================================================================
# RAPPRESENTAZIONE INDIVIDUO
# =============================================================================
class Individual:
    """
    Rappresenta un individuo nella popolazione.
    
    HINT: 
    - Codifica binaria: 1 = feature selezionata, 0 = feature non selezionata
    - Considerate come gestire il caso in cui nessuna feature sia selezionata
    """
    
    def __init__(self, n_features: int, chromosome: np.ndarray = None):
        # TODO: Inizializzare il cromosoma (random se non fornito)
        
        if chromosome is None:
            self.chromosome = np.random.randint(0,2,size=n_features)
        else:
            self.chromosome = chromosome

        if np.sum(self.chromosome) == 0:
            idx = np.random.randint(len(self.chromosome))
            self.chromosome[idx] = 1
    

    def count_selected_features(self) -> int:
        """Restituisce il numero di features selezionate."""
        # TODO: Implementare
        
        count = 0
        for i in range(len(self.chromosome)):
            if self.chromosome[i] == 1:
                count += 1
        return count 


# =============================================================================
# FUNZIONE FITNESS
# =============================================================================
def fitness_correlation_based(individual: Individual, r_cf: np.ndarray, r_ff: np.ndarray) -> float:
    """
    Calcola il fitness basato sulla correlation analysis.
    Usa correlazioni PRE-CALCOLATE per efficienza.
    
    HINT:
    - Considerate la correlazione features-classe e features-features
    - Un buon subset ha alta correlazione con la classe e bassa ridondanza
    - Formula suggerita: CFS (Correlation-based Feature Selection)
    
    Returns:
        float: valore di fitness (più alto = migliore)
    """
    # Indici delle feature selezionate
    selected_idx = np.where(individual.chromosome == 1)[0]
    k = len(selected_idx)

    # Nessuna feature selezionata -> fitness = 0
    if k == 0:
        return 0.0

    # ---------- Correlazione feature-classe ----------
    r_cf_mean = np.mean(r_cf[selected_idx])

    # ---------- Correlazione feature-feature ----------
    if k == 1:
        r_ff_mean = 0.0
    else:
        sub_matrix = r_ff[np.ix_(selected_idx, selected_idx)]
        # Formula: (somma totale - diagonale) / (k² - k)
        r_ff_mean = (np.sum(sub_matrix) - k) / (k ** 2 - k)

    # ---------- Formula CFS ----------
    denominator = np.sqrt(k + k * (k - 1) * r_ff_mean)
    if denominator == 0:
        return 0.0
    
    return (k * r_cf_mean) / denominator


# =============================================================================
# OPERATORI GENETICI
# =============================================================================

# --- Selezione ---
def tournament_selection(population: List[Individual], fitness_values: List[float], 
                         tournament_size: int = 3) -> Individual:
    """
    Selezione tramite torneo.
    
    HINT: Seleziona k individui random e restituisce il migliore.
    """
    # TODO: Implementare
    
    tournament_indices = random.sample(range(len(fitness_values)), tournament_size)
    best_idx = max(tournament_indices, key=lambda idx: fitness_values[idx])
    return population[best_idx]
def roulette_wheel_selection(population: List[Individual], fitness_values: List[float]) -> Individual:
    """
    Selezione a roulette wheel.
    
    HINT: Probabilità proporzionale al fitness.
    """
    # TODO: Implementare
    fitness = np.array (fitness_values)
    # calcolo delle probabilità proporzionali alla fitness
    probabilities = fitness / fitness.sum()

    selected_index = np.random.choice (len(population), p = probabilities)
   
    return population [selected_index]
# --- Crossover ---
def single_point_crossover(parent1: Individual, parent2: Individual, 
                           crossover_rate: float = 0.8) -> Tuple[Individual, Individual]:
    """
    Crossover a singolo punto.
    
    HINT: Con probabilità crossover_rate, eseguire il crossover.
    """
    # TODO: Implementare
    
    if random.random() > crossover_rate : 
       return parent1,parent2
    else:
        point= random.randint (1, len(parent1.chromosome)-1)
        offspring1_c=np.concatenate([parent1.chromosome[0:point],parent2.chromosome[point:]])
        offspring2_c=np.concatenate([parent2.chromosome[0:point],parent1.chromosome[point:]])

        offspring1=Individual(len(offspring1_c), offspring1_c)
        offspring2=Individual(len(offspring2_c), offspring2_c)

        return offspring1, offspring2
# --- Mutazione ---
def bit_flip_mutation(individual: Individual, mutation_rate: float = 0.1) -> Individual:
    """
    Mutazione bit-flip.
    
    HINT: Ogni bit ha probabilità mutation_rate di essere invertito.
    """
    # TODO: Implementare
    mutated = Individual(len(individual.chromosome), individual.chromosome.copy())
    for i in range(len( mutated.chromosome )):
        if random.random() < mutation_rate:
            mutated.chromosome[i] = 1 - mutated.chromosome[i]
    return mutated


# =============================================================================
# ALGORITMO GENETICO
# =============================================================================
class GeneticAlgorithm:
    """
    Implementazione dell'Algoritmo Genetico per Feature Selection.
    
    HINT: Questa classe dovrebbe essere modulare per permettere
    l'analisi parametrica richiesta dalla tesina.
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 selection_method: str = 'tournament',
                 tournament_size: int = 3,
                 max_generations: int = 100,
                 convergence_threshold: int = None,  # generazioni senza miglioramento
                 convergence_tolerance: float = 1e-5,
                 random_seed: int = SEED):
        
        # TODO: Inizializzare i parametri
        # Parametri della popolazione
        self.population_size = population_size  # numero di individui
        self.crossover_rate = crossover_rate    # probabilità di crossover
        self.mutation_rate = mutation_rate      # probabilità di mutazione

        # Metodo di selezione
        self.selection_method = selection_method  # 'tournament' o 'roulette'
        self.tournament_size = tournament_size    # solo se selection_method=='tournament'

        # Controllo delle generazioni
        self.max_generations = max_generations          # numero massimo di generazioni
        self.convergence_threshold = convergence_threshold  # stop se non c'è miglioramento
        self.convergence_tolerance = convergence_tolerance  # tolleranza per il miglioramento

        # Seed casuale per riproducibilità
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)



    def initialize_population(self, n_features: int) -> List[Individual]:
        """Inizializza la popolazione random."""
        # TODO: Implementare
        self.population = []
        for _ in range (self.population_size):
            self.population.append(Individual (n_features=n_features))
        
        return self.population

    
    def evaluate_population(self, population: List[Individual], 
                           r_cf: np.ndarray, r_ff: np.ndarray) -> List[float]:
        """Valuta il fitness di tutta la popolazione usando correlazioni pre-calcolate."""
        # TODO: Implementare

        # reset before the new gen
        self.fitness_values=[]
        for ind in population:
            self.fitness_values.append(fitness_correlation_based(ind, r_cf, r_ff))
        
        return self.fitness_values
    
    def run(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Esegue un GA
        Restituisce un dizionario con le informazioni della run
        """    
        start_time = time.time()
        r_cf, r_ff = precompute_correlations(X, y)
        population = self.initialize_population(X.shape[1])
        
        no_improvement = 0
        best_individual_global = None    
        best_fitness_global = -np.inf
        
        fitness_history = []
        avg_fitness_history = []
        diversity_history = []
        
        for gen in range(self.max_generations):
            fitness_values = self.evaluate_population(population, r_cf, r_ff)
            
            # Identifica il migliore della generazione attuale
            current_best_idx = np.argmax(fitness_values)
            current_best_ind = population[current_best_idx]
            current_best_fit = fitness_values[current_best_idx]
            
            # --- AGGIORNAMENTO BEST GLOBAL E ELITISMO ---
            # Se il migliore attuale supera il record storico, aggiorna
            if current_best_fit > best_fitness_global + self.convergence_tolerance:
                best_fitness_global = current_best_fit
                # Copia profonda per non corromperlo con mutazioni successive
                best_individual_global = Individual(len(current_best_ind.chromosome), current_best_ind.chromosome.copy())
                no_improvement = 0
            else: 
                no_improvement += 1

            # Logging
            fitness_history.append(best_fitness_global) # Logghiamo il record storico per curve pulite
            avg_fitness_history.append(np.mean(fitness_values))
            diversity_history.append(self.calculate_population_diversity(population))

            if self.convergence_threshold and no_improvement >= self.convergence_threshold:
                break

            # --- GENERAZIONE NUOVA POPOLAZIONE ---
            new_population = []
            
            # ELITISMO: Inseriamo il miglior individuo assoluto trovato finora
            # Questo garantisce che la best fitness non cali mai
            new_population.append(Individual(len(best_individual_global.chromosome), best_individual_global.chromosome.copy()))

            # Riempiamo il resto della popolazione
            while len(new_population) < self.population_size:
                if self.selection_method == "tournament":
                    p1 = tournament_selection(population, fitness_values, self.tournament_size)
                    p2 = tournament_selection(population, fitness_values, self.tournament_size)
                else: 
                    p1 = roulette_wheel_selection(population, fitness_values)
                    p2 = roulette_wheel_selection(population, fitness_values)

                off1, off2 = single_point_crossover(p1, p2, self.crossover_rate)
                
                # Mutazione e aggiunta
                new_population.append(bit_flip_mutation(off1, self.mutation_rate))
                if len(new_population) < self.population_size:
                    new_population.append(bit_flip_mutation(off2, self.mutation_rate))

            population = new_population
    
        execution_time = time.time() - start_time 
        return {
            "fitness_history": fitness_history,
            "avg_fitness_history": avg_fitness_history,
            "diversity_history": diversity_history,
            "best_individual": best_individual_global,
            "best_fitness": best_fitness_global,
            "best_features_idx": np.where(best_individual_global.chromosome == 1)[0],
            "n_selected_features": best_individual_global.count_selected_features(),
            "generations_completed": gen + 1,
            "execution_time": execution_time
        }
    
    def calculate_population_diversity(self, population: List[Individual]) -> float:
        """
        Calcola la diversità della popolazione.
        
        HINT: Diversità genetica = varianza nei cromosomi
        """
        # TODO: Implementare
        chromosomes = np.array([ind.chromosome for ind in population])
        
        diversity = np.mean(np.var(chromosomes, axis=0))
        
        return diversity


# =============================================================================
# LOGGING E METRICHE
# =============================================================================
class ExperimentLogger:
    """
    Logger per gli esperimenti.
    """
    
    def __init__(self):
        self.generations_data = []
        self.run_times = []
        self.feature_counts = {}  # frequenza selezione per feature
    
    def log_generation(self, generation: int, best_fitness: float, 
                       avg_fitness: float, diversity: float):
        """
        Logga i dati di una generazione (solo statistiche, niente feature counts qui).
        """
        self.generations_data.append({
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "diversity": diversity
        })
    
    def log_run(self, run_id: int, best_fitness : float, best_individual: Individual, 
                execution_time: float, generations_completed: int):
        """Logga i dati riassuntivi di un run completo."""
        self.run_times.append({
            "run_id": run_id,
            "best_fitness": best_fitness, 
            "n_features_selected": best_individual.count_selected_features(),
            "execution_time": execution_time,
            "generations_completed": generations_completed,
            "chromosome": best_individual.chromosome.copy()
        })

    def log_feature_selection(self, chromosome: np.ndarray):
        """
        Aggiorna la frequenza delle feature. Da chiamare alla fine del RUN
                        """
        for idx, val in enumerate(chromosome):
            if val == 1:
                self.feature_counts[idx] = self.feature_counts.get(idx, 0) + 1


# =============================================================================
# ESPERIMENTI PARAMETRICI
# =============================================================================

def run_experiment_population_size(X: pd.DataFrame, y: pd.Series, 
                                   n_runs: int = 30):
    """
    Scenario 1: Test dimensioni popolazione.
    Restituisce risultati separati per config+conteggio feature.
    """
    population_sizes = [20, 50, 100, 200, 500] 
    #population_sizes = [100]

    results = {} 
    # Dizionario per salvare i conteggi separati: {20: {feat_idx: count}, 50: {...}}
    feature_counts_by_config = {} 

    for pop_size in population_sizes:
        # Creiamo un logger nuovo per ogni configurazione (per non mischiare i dati)
        current_logger = ExperimentLogger()
        results[pop_size] = []
        
        pop_start_time = time.time()
        
        print(f"\n>>> INIZIO TEST POPOLAZIONE: {pop_size}")

        for run_id in range(n_runs):
            run_start = time.time()
            
            # Seed variabile per ogni run
            ga = GeneticAlgorithm(population_size=pop_size, random_seed=SEED + run_id)
            
            run_result = ga.run(X, y)
            results[pop_size].append(run_result)

            # --- LOGGING ---
            
            # 1. Log Run Stats
            current_logger.log_run(
                run_id=run_id,
                best_fitness=run_result["best_fitness"],
                best_individual=run_result["best_individual"],
                execution_time=run_result["execution_time"],
                generations_completed=run_result["generations_completed"],
            )

            # 2. Log History Generazioni (Per curve di convergenza)
            for g in range(len(run_result["fitness_history"])):
                current_logger.log_generation(
                    generation=g,
                    best_fitness=run_result["fitness_history"][g],
                    avg_fitness=run_result["avg_fitness_history"][g],
                    diversity=run_result["diversity_history"][g]
                ) 
            
            # 3. Log Features
            current_logger.log_feature_selection(run_result["best_individual"].chromosome)

            run_duration = time.time() - run_start
            # print(f"Run {run_id + 1}/{n_runs} completata ({run_duration:.2f}s)")

        # Salviamo i conteggi di questa configurazione
        feature_counts_by_config[pop_size] = current_logger.feature_counts
        
        pop_total_time = time.time() - pop_start_time
        print(f"FINE TEST POP {pop_size}. Tempo totale: {pop_total_time:.2f}s")
        
    return results, feature_counts_by_config

def run_experiment_genetic_operators(X: pd.DataFrame, y: pd.Series,
                                     n_runs: int = 30):
    """
    Scenario 2: Test operatori genetici
    Restituisce risultati e conteggi feature per configurazione
    """
    crossover_rates = [0.6, 0.7, 0.8, 0.9]
    mutation_rates = [0.01, 0.05, 0.1, 0.15]
    selection_methods = [
        ("tournament", 2),
        ("tournament", 3),
        ("tournament", 4),
        ("roulette", None),
    ]

    results = {}
    feature_counts_by_config = {}

    for cr in crossover_rates:
        for mr in mutation_rates:
            for sel_method, k in selection_methods:

                config_name = f"CR={cr}_MR={mr}_SEL={sel_method}_K={k}"
                print(f"\n>>> TEST CONFIGURAZIONE: {config_name}")
                print("-" * 50)

                current_logger = ExperimentLogger()
                results[config_name] = []

                for run_id in range(n_runs):
                    ga = GeneticAlgorithm(
                        population_size=100,
                        crossover_rate=cr,
                        mutation_rate=mr,
                        selection_method=sel_method,
                        tournament_size=k if sel_method == "tournament" else 3,
                        random_seed=SEED + run_id
                    )

                    run_result = ga.run(X, y)
                    results[config_name].append(run_result)

                    # --- LOGGING ---

                    current_logger.log_run(
                        run_id=run_id,
                        best_fitness=run_result["best_fitness"],
                        best_individual=run_result["best_individual"],
                        execution_time=run_result["execution_time"],
                        generations_completed=run_result["generations_completed"],
                    )

                    for g in range(len(run_result["fitness_history"])):
                        current_logger.log_generation(
                            generation=g,
                            best_fitness=run_result["fitness_history"][g],
                            avg_fitness=run_result["avg_fitness_history"][g],
                            diversity=run_result["diversity_history"][g]
                        )

                    # LOG FEATURE (una sola volta per run)
                    current_logger.log_feature_selection(
                        run_result["best_individual"].chromosome
                    )

                feature_counts_by_config[config_name] = current_logger.feature_counts

    return results, feature_counts_by_config

def run_experiment_stopping_criteria(X: pd.DataFrame, y: pd.Series,
                                     n_runs: int = 30):
    """
    Scenario 3: Test criteri di stop
    
    Testa tutte le combinazioni di:
    - max_generations: [50, 100, 200]
    - convergence_threshold: [10, 20, 30]
    - convergence_tolerance: [1e-4, 1e-5, 1e-6]

    Restituisce:
        results: dict con risultati per ogni combinazione
        feature_counts_by_config: dict con conteggi feature per ogni combinazione
    """
    fixed_generations = [50, 100, 200]
    convergence_thresholds = [10, 20, 30]
    tolerances = [1e-4, 1e-5, 1e-6]

    results = {}
    feature_counts_by_config = {}

    # Tutte le combinazioni
    for max_gen in fixed_generations:
        for conv_thresh in convergence_thresholds:
            for tol in tolerances:
                config_name = f"MaxGen={max_gen}_Conv={conv_thresh}_Tol={tol}"
                print(f"\n>>> TEST CRITERIO: {config_name}")
                print("-" * 50)

                current_logger = ExperimentLogger()
                results[config_name] = []

                for run_id in range(n_runs):
                    ga = GeneticAlgorithm(
                        population_size=100,
                        max_generations=max_gen,
                        convergence_threshold=conv_thresh,
                        convergence_tolerance=tol,
                        random_seed=SEED + run_id
                    )

                    run_result = ga.run(X, y)
                    results[config_name].append(run_result)

                    # Logging run
                    current_logger.log_run(
                        run_id=run_id,
                        best_fitness=run_result["best_fitness"],
                        best_individual=run_result["best_individual"],
                        execution_time=run_result["execution_time"],
                        generations_completed=run_result["generations_completed"],
                    )

                    # Logging generazioni
                    for g in range(len(run_result["fitness_history"])):
                        current_logger.log_generation(
                            generation=g,
                            best_fitness=run_result["fitness_history"][g],
                            avg_fitness=run_result["avg_fitness_history"][g],
                            diversity=run_result["diversity_history"][g]
                        )

                    # Logging feature (una sola volta per run)
                    current_logger.log_feature_selection(
                        run_result["best_individual"].chromosome
                    )

                feature_counts_by_config[config_name] = current_logger.feature_counts

    return results, feature_counts_by_config


    

# =============================================================================
# VISUALIZZAZIONE
# =============================================================================

# Curve di convergenza
def plot_convergence_curves(results: Dict, title: str = "Convergence Curves"):
    plt.figure(figsize=(10, 6))

    for pop_size, runs in results.items():
        # 1. Trova la lunghezza massima tra tutte le run di questa categoria
        max_len = max(len(run["fitness_history"]) for run in runs)
        
        padded_histories = []
        for run in runs:
            history = run["fitness_history"]
            # 2. Se la run è finita prima, riempiamo il resto con l'ultimo valore trovato
            if len(history) < max_len:
                padding = [history[-1]] * (max_len - len(history))
                history = list(history) + padding
            padded_histories.append(history)

        fitness_array = np.array(padded_histories)
        mean_fitness = np.mean(fitness_array, axis=0)
        std_fitness = np.std(fitness_array, axis=0)
        generations = np.arange(max_len)

        line, = plt.plot(generations, mean_fitness, label=f"Pop size = {pop_size}", linewidth=2)
        plt.fill_between(
            generations, 
            mean_fitness - std_fitness, 
            mean_fitness + std_fitness, 
            alpha=0.15,
            color=line.get_color() # Stesso colore della linea
        )

    plt.xlabel("Generations")
    plt.ylabel("Best Fitness (Global)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Boxplot 
def plot_fitness_boxplots(results: Dict, title: str = "Fitness Distribution"):
    """
    - Gestisce sia liste di numeri (Scenario 2 pulito)
    - Sia liste di dizionari run (Scenario 1 grezzo)
    """
    plt.figure(figsize=(10, 6))

    labels = []
    data = []

    # Ordiniamo le chiavi. Proviamo a ordinarle come numeri se possibile (es. 20, 50, 100)
    try:
        sorted_items = sorted(results.items(), key=lambda x: int(x[0].split('=')[1]) if '=' in str(x[0]) else int(x[0]))
    except:
        sorted_items = sorted(results.items()) 

    # in questo modo si gestiscono entrambi gli scenari (passaggio di dati grezzi, oppure direttametne di fitness) 
    for key, values in sorted_items:
        # Controlliamo cosa c'è dentro 'values'
        if isinstance(values, list) and len(values) > 0 and isinstance(values[0], dict):
            # Caso Scenario 1: Sono dizionari run! Estraiamo la fitness
            clean_values = [run['best_fitness'] for run in values]
            data.append(clean_values)
        else:
            # Caso Scenario 2: Sono già numeri!
            data.append(values)
        # ---------------------------
        
        labels.append(str(key))

    # Creazione Boxplot
    box = plt.boxplot(
        data,
        tick_labels=labels, 
        showmeans=True,
        meanline=True,
        patch_artist=True
    )
    
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')

    plt.ylabel("Best Fitness Finale")
    plt.xlabel("Configurazione")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Heatmap per analizzare variazione di CR e MR
def plot_heatmap_operators(results: Dict, title: str = "Heatmap: Crossover vs Mutation Impact") -> None:
    """
    Genera una Heatmap per analizzare l'interazione tra Crossover (CR) e Mutation (MR).
    
    Logica:
    1. Itera su tutte le configurazioni nel dizionario 'results'.
    2. Estrae CR e MR dalle chiavi (es. "CR=0.6_MR=0.01_SEL=tournament...").
    3. Calcola la fitness media per quella configurazione.
    4. Usa Pandas per creare una tabella pivot (Media di tutte le selezioni per quel CR/MR).
    5. Disegna la heatmap usando Seaborn.
    """
    
    data_points = []

    # 1. Parsing dei dati dal dizionario results
    for config_key, runs_list in results.items():
        # Esempio chiave: "CR=0.6_MR=0.01_SEL=tournament_K=2"
        parts = config_key.split('_')
        
        # Estrazione sicura dei valori CR e MR
        try:
            cr_val = float(parts[0].split('=')[1]) # Prende 0.6
            mr_val = float(parts[1].split('=')[1]) # Prende 0.01
        except IndexError:
            continue # Salta chiavi malformate se ce ne fossero

        # Calcoliamo la media della fitness delle 30 run per questa chiave
        # Usiamo 'best_fitness' che è salvato dentro ogni run
        fits = [run['best_fitness'] for run in runs_list]
        avg_fit = np.mean(fits)

        data_points.append({
            'Crossover Rate': cr_val,
            'Mutation Rate': mr_val,
            'Fitness': avg_fit
        })

    # 2. Creazione DataFrame e Pivot Table
    df = pd.DataFrame(data_points)
    
    # Raggruppa per CR e MR facendo la media (nel caso ci siano più metodi di selezione per lo stesso CR/MR)
    pivot_table = df.pivot_table(
        index='Mutation Rate', 
        columns='Crossover Rate', 
        values='Fitness', 
        aggfunc='mean'
    )

    # 3. Plotting
    plt.figure(figsize=(10, 8))
    
    # cmap="RdYlGn" -> Rosso (basso), Giallo (medio), Verde (alto)
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="RdYlGn", 
                linewidths=.5, cbar_kws={'label': 'Fitness Media'})
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Crossover Rate (CR)", fontsize=12)
    plt.ylabel("Mutation Rate (MR)", fontsize=12)
    
    # Invertiamo l'asse Y per avere i valori bassi di MR in basso (opzionale, standard nei grafici cartesiani)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# GRAFICO 1: Trade-off (Fitness vs Tempo) e GRAFICO 2: Efficacia criteri di stop (Max Gen vs Numero di generazioni effettive)
def plot_efficiency_and_stopping(results: Dict):
    """
    Crea 2 grafici:
    1. Trade-off: Fitness vs Tempo (per mostrare i rendimenti decrescenti)
    2. Stopping Check: MaxGenerations vs ActualGenerations (per vedere se si ferma prima)
    """
    # 1. Preparazione Dati: Raggruppiamo per MaxGen (50, 100, 200)
    groups = {50: [], 100: [], 200: []}
    
    for key, runs in results.items():
        # Estrai il numero 50, 100 o 200 dalla chiave
        parts = key.split('_')
        max_gen_val = int(parts[0].split('=')[1])
        
        # Calcola le medie per questa configurazione
        fits = [run['best_fitness'] for run in runs]
        times = [run['execution_time'] for run in runs]
        # Conta quanti elementi ha la history = generazioni fatte davvero
        actual_gens = [len(run['fitness_history']) for run in runs]
        
        # Salviamo tutto
        groups[max_gen_val].append({
            "fit": np.mean(fits),
            "time": np.mean(times),
            "actual_gen": np.mean(actual_gens)
        })

    # Calcoliamo le medie totali per ogni gruppo (50, 100, 200)
    x_labels = [50, 100, 200]
    avg_fitness = []
    avg_time = []
    avg_actual_gens = []

    for g in x_labels:
        data = groups[g]
        # Media delle medie delle varie configurazioni dentro quel gruppo
        avg_fitness.append(np.mean([d['fit'] for d in data]))
        avg_time.append(np.mean([d['time'] for d in data]))
        avg_actual_gens.append(np.mean([d['actual_gen'] for d in data]))

    # --- CREAZIONE GRAFICI ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # GRAFICO 1: Trade-off (Fitness vs Tempo)
    # Asse Y sinistro: Fitness (Barre)
    ax1.bar(x_labels, avg_fitness, width=20, color='skyblue', label='Fitness Media', alpha=0.8)
    ax1.set_xlabel("Max Generations Impostate")
    ax1.set_ylabel("Fitness Finale", color='blue')
    ax1.set_ylim(min(avg_fitness)*0.99, max(avg_fitness)*1.005) 
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(x_labels)

    # Asse Y destro: Tempo (Linea)
    ax1_bis = ax1.twinx()
    ax1_bis.plot(x_labels, avg_time, color='red', marker='o', linewidth=3, label='Tempo Esecuzione')
    ax1_bis.set_ylabel("Tempo Medio (s)", color='red')
    ax1_bis.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title("Costo/Beneficio: Più generazioni valgono la pena?")
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # GRAFICO 2: Funziona lo Stop? (Max vs Actual)
    # Barre che mostrano il limite impostato (vuote/tratteggiate)
    ax2.bar(x_labels, x_labels, width=20, color='white', edgecolor='black', linestyle='--', label='Limite Impostato (MaxGen)')
    
    # Barre che mostrano dove ci siamo fermati davvero (piene)
    ax2.bar(x_labels, avg_actual_gens, width=15, color='green', alpha=0.7, label='Generazioni Reali (Media)')
    
    # Aggiungiamo il valore sopra le barre
    for i, v in enumerate(avg_actual_gens):
        ax2.text(x_labels[i], v + 5, f"{v:.1f}", ha='center', fontweight='bold')

    ax2.set_xlabel("Max Generations Impostate")
    ax2.set_ylabel("Numero Generazioni")
    ax2.set_xticks(x_labels)
    ax2.set_title("Efficacia Criteri di Stop (Early Stopping)")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# Heatmap per analizzare variazione di # generazioni totale variando Tolerance e Convergence 
# Heatmap per analizzare variazione di best fitness variando Tolerance e Convergence
def plot_stopping_heatmap(results: Dict):
    """
    Analizza solo le run con MaxGen=100 per studiare
    l'impatto dei parametri di arresto anticipato.

    Genera 2 heatmap:
    1. Efficienza: numero medio di generazioni effettive
    2. Qualità: fitness media
    """
    data = []

    # 1. Filtering & data extraction
    for key, runs in results.items():
        # Key format: "MaxGen=100_Conv=10_Tol=0.0001"
        parts = key.split('_')
        max_gen_val = int(parts[0].split('=')[1])

        if max_gen_val != 100:
            continue

        conv_value = int(parts[1].split('=')[1])
        tol_value = float(parts[2].split('=')[1])

        actual_gens = [len(run['fitness_history']) for run in runs]
        fits = [run['best_fitness'] for run in runs]

        data.append({
            "Convergence": conv_value,
            "Tolerance": tol_value,
            "Real_Generations": np.mean(actual_gens),
            "Fitness": np.mean(fits)
        })

    if not data:
        print("Error: no data found for MaxGen=100")
        return

    df = pd.DataFrame(data)

    # 2. Pivot tables
    pivot_gens = df.pivot(
        index="Tolerance",
        columns="Convergence",
        values="Real_Generations"
    )

    pivot_fit = df.pivot(
        index="Tolerance",
        columns="Convergence",
        values="Fitness"
    )

    pivot_gens = pivot_gens.sort_index(ascending=False)
    pivot_fit = pivot_fit.sort_index(ascending=False)

    # 3. Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        pivot_gens,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        ax=ax1,
        cbar_kws={'label': 'Actual Generations'}
    )
    ax1.set_title("Efficiency: Required Generations (MaxGen=100)")
    ax1.set_xlabel("Convergence")
    ax1.set_ylabel("Tolerance")

    min_f, max_f = df["Fitness"].min(), df["Fitness"].max()
    sns.heatmap(
        pivot_fit,
        annot=True,
        fmt=".4f",
        cmap="OrRd",
        ax=ax2,
        vmin=min_f,
        vmax=max_f,
        cbar_kws={'label': 'Best Fitness'}
    )
    ax2.set_title("Solution Quality")
    ax2.set_xlabel("Convergence")
    ax2.set_ylabel("")

    plt.tight_layout()
    plt.show()


# Bar plot (popolazione vs fitness) + andamento temporale. Usato per lo scenario 1
def plot_population_tradeoff(results: Dict):
    """
    Scenario 1: Grafico a doppio asse per Population Size.
    """
    # 1. Identifichiamo il tipo di chiavi per ordinarle correttamente
    raw_keys = list(results.keys())
    
    # Caso A: Le chiavi sono NUMERI (es. 20, 50, 100) -> È il tuo caso attuale
    if all(isinstance(k, (int, float)) for k in raw_keys):
        sorted_keys = sorted(raw_keys)
        # Creiamo le etichette per l'asse X convertendo i numeri in stringhe
        x_labels = [str(k) for k in sorted_keys]
        
    # Caso B: Le chiavi sono STRINGHE (es. "Pop=20")
    else:
        try:
            # Proviamo a estrarre il numero dopo l'uguale
            sorted_keys = sorted(raw_keys, key=lambda x: int(str(x).split('=')[1]))
            x_labels = [str(x).split('=')[1] for x in sorted_keys]
        except:
            # Se fallisce, ordiniamo alfabeticamente
            sorted_keys = sorted(raw_keys)
            x_labels = [str(x) for x in sorted_keys]

    # 2. Estrazione Dati
    times = []    # Valori asse Y1 (Tempo)
    fits = []     # Valori asse Y2 (Fitness)

    for key in sorted_keys:
        runs = results[key]
        
        # Calcoliamo le medie
        avg_time = np.mean([run['execution_time'] for run in runs])
        avg_fit = np.mean([run['best_fitness'] for run in runs])
        
        times.append(avg_time)
        fits.append(avg_fit)

    # 3. Creazione grafico
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Asse 1: Tempo (Barre Rosse)
    ax1.set_xlabel('Dimensione Popolazione')
    ax1.set_ylabel('Tempo Medio (s)', color='tab:red')
    ax1.bar(x_labels, times, color='tab:red', alpha=0.6, label="Tempo")
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Asse 2: Fitness (Linea Blu)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Fitness Media', color='tab:blue')
    ax2.plot(x_labels, fits, color='tab:blue', marker='o', linewidth=3, label="Fitness")
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Trade-off Scenario 1: Popolazione vs Costi/Benefici")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# GRAFICO FINALE: Feature Selezionate dalla Config Ottimale
def plot_final_feature_selection(results_table, title="Feature Selection Finale"):
    """
    Plotta le feature selezionate con frequenza > 50%.
    """
    # Filtra solo feature stabili (>50% dei run)
    stable_features = [f for f in results_table if f['percentage'] >= 50]
    
    if not stable_features:
        print("Nessuna feature supera il 50% di frequenza!")
        return
    
    # Preparazione dati
    names = [f['feature_name'] for f in stable_features]
    freqs = [f['frequency'] for f in stable_features]
    percs = [f['percentage'] for f in stable_features]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.4)))
    
    bars = ax.barh(range(len(names)), freqs, color='teal', alpha=0.8, edgecolor='black')
    
    # Aggiungi percentuali sulle barre
    for i, (bar, perc) in enumerate(zip(bars, percs)):
        ax.text(bar.get_width() + 0.5, i, f'{perc:.1f}%', 
                va='center', fontsize=10, fontweight='bold')
    
    # Linea di riferimento al 50%
    ax.axvline(x=15, color='red', linestyle='--', linewidth=2, 
               label='Soglia Stabilità (50%)', alpha=0.7)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Frequenza di Selezione (n. di run su 30)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.legend(fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n {len(stable_features)} feature stabili identificate (freq >= 50%)")

# Bar plot confronto frequenza di selezioni delle feature tra due configutazioni
def plot_feature_frequency_comparison(results: Dict, keys_to_compare: List, feature_names: List[str], title: str):
    """
    Confronta la frequenza di selezione delle feature tra due configurazioni.
    Versione ROBUSTA: Gestisce oggetti Individual estraendo i geni (.genes o cast a list).
    """
    data_for_plot = []
    
    # 1. Calcoliamo le frequenze
    all_frequencies = {} 
    
    for key in keys_to_compare:
        if key not in results:
            print(f"Attenzione: Chiave '{key}' non trovata! Chiavi disponibili: {list(results.keys())}")
            continue
            
        runs = results[key]
        n_runs = len(runs)
        
        # Inizializziamo il contatore
        total_counts = np.zeros(len(feature_names))
        
        for run in runs:
            ind_obj = run['best_individual']
            if hasattr(ind_obj, 'genes'):
                mask = ind_obj.genes
            elif hasattr(ind_obj, 'chromosome'):
                mask = ind_obj.chromosome
            # Se è una lista o un oggetto DEAP che si comporta come lista
            else:
                mask = list(ind_obj)
            # ----------------------------------------

            total_counts += np.array(mask, dtype=int)
            
        # Convertiamo in percentuale
        freq_percent = (total_counts / n_runs) * 100
        all_frequencies[key] = freq_percent

    # Se non abbiamo trovato nulla, usciamo
    if not all_frequencies:
        print("Nessun dato da graficare.")
        return

    # 2. Identifichiamo le Top 15 feature GLOBALI (tra quelle selezionate)
    sum_freqs = np.zeros(len(feature_names))
    for k in all_frequencies:
        sum_freqs += all_frequencies[k]
        
    # Indici delle top 15
    top_indices = np.argsort(sum_freqs)[::-1][:15]
    
    # 3. Prepariamo i dati per il grafico
    for key in keys_to_compare:
        freqs = all_frequencies[key]
        
        # Etichetta pulita per la legenda
        clean_label = str(key).replace("MaxGen=", "").replace("Pop=", "") 
        
        for idx in top_indices:
            data_for_plot.append({
                "Feature": feature_names[idx],
                "Frequency (%)": freqs[idx],
                "Configuration": clean_label 
            })
            
    df = pd.DataFrame(data_for_plot)

    # 4. Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df, 
        x="Feature", 
        y="Frequency (%)", 
        hue="Configuration",
        palette="viridis"
    )
    
    plt.title(title)
    plt.ylabel("Frequenza di Selezione (su 30 Run)")
    plt.xlabel("Top 15 Features (Biomarcatori Più Comuni)")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Configurazione")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Convergence e diversity plot
def plot_convergence_and_diversity(results: Dict, title: str = "Convergence & Diversity Analysis"):
    """
    Grafico a doppio asse:
    - Asse SX (Solido): Fitness Media 
    - Asse DX (Tratteggiato): Diversità Media 
    """
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx() # Crea il secondo asse Y
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, (key, runs) in enumerate(results.items()):
        color = colors[i]
        
        # 1. Dati Fitness (Linea Solida)
        # (Codice di padding per gestire run di lunghezza diversa)
        histories_fit = [r['fitness_history'] for r in runs]
        max_len = max(len(h) for h in histories_fit)
        padded_fit = [list(h) + [h[-1]]*(max_len-len(h)) for h in histories_fit]
        mean_fit = np.mean(padded_fit, axis=0)
        
        # 2. Dati Diversità (Linea Tratteggiata)
        if 'diversity_history' not in runs[0]:
            print(f" Diversità non trovata per {key}")
            continue
            
        histories_div = [r['diversity_history'] for r in runs]
        padded_div = [list(h) + [h[-1]]*(max_len-len(h)) for h in histories_div]
        mean_div = np.mean(padded_div, axis=0)
        
        gens = range(max_len)
        
        # Plot Fitness (SX)
        ax1.plot(gens, mean_fit, color=color, linewidth=2, label=f"Fit: {key}")
        
        # Plot Diversità (DX)
        # Usa linestyle='--' per distinguerla
        ax2.plot(gens, mean_div, color=color, linestyle='--', alpha=0.6, linewidth=1.5) # label=f"Div: {key}" (opzionale)


    ax1.set_xlabel("Generazioni")
    ax1.set_ylabel("Best Fitness (Accuratezza)", fontweight='bold')
    ax2.set_ylabel("Diversità Popolazione", rotation=270, labelpad=15, fontweight='bold')
    
    ax1.set_title(title)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    plt.show()




# =============================================================================
# FUNZIONE DI SUPPORTO 
# =============================================================================

# Da "results", dizionario con tutti i risultati, in un dizionario semplice chiave-lista di fitness
# Per scenario due, si raggruppa per metodi di selezione
def prepare_data_by_selection(results: Dict) -> Dict[str, List[float]]:
    """
    Trasforma il dizionario complesso dei risultati in un dizionario semplice
    raggruppato per metodo di selezione.
    
    Return:
    {
        "Roulette": [0.85, 0.86, ...],
        "Tourn (k=2)": [0.88, 0.89, ...],
        ...
    }
    """
    # Inizializziamo i gruppi
    grouped_data = {
        "Roulette": [],
        "Tourn (k=2)": [],
        "Tourn (k=3)": [],
        "Tourn (k=4)": []
    }

    for key, runs in results.items():
        # Estraiamo tutte le fitness delle 30 run per questa configurazione
        fitness_values = [run['best_fitness'] for run in runs]
        
        # Smistiamo i dati nel cassetto giusto
        if "roulette" in key.lower():
            grouped_data["Roulette"].extend(fitness_values)
        elif "tournament" in key.lower():
            if "K=2" in key:
                grouped_data["Tourn (k=2)"].extend(fitness_values)
            elif "K=3" in key:
                grouped_data["Tourn (k=3)"].extend(fitness_values)
            elif "K=4" in key:
                grouped_data["Tourn (k=4)"].extend(fitness_values)
                
    return grouped_data

# Estrae migliore, peggiore, media da tutte le configurazioni
# Per scenario due 
def extract_best_mid_worst(results: Dict) -> Dict:
    """
    Analizza tutte le configurazioni, calcola la fitness media finale
    e restituisce un dizionario contenente solo le 3 configurazioni chiave.
    Le etichette vengono accorciate (Tournament->T, Roulette->R) per il grafico.
    """
    # 1. Calcoliamo la media finale per ogni configurazione
    leaderboard = []
    
    for key, runs in results.items():
        # Calcola media fitness finale delle 30 run
        final_fits = [run['best_fitness'] for run in runs]
        avg_fit = np.mean(final_fits)
        leaderboard.append((key, avg_fit))

    # 2. Ordiniamo dal migliore al peggiore
    leaderboard.sort(key=lambda x: x[1], reverse=True)

    # 3. Selezioniamo i 3 campioni
    best_key, best_val = leaderboard[0]
    worst_key, worst_val = leaderboard[-1]
    mid_index = len(leaderboard) // 2
    mid_key, mid_val = leaderboard[mid_index]

    # Funzione helper interna per accorciare le stringhe
    def clean_label(label):
        # Sostituisce i nomi lunghi con abbreviazioni
        label = label.replace("tournament", "T")
        label = label.replace("roulette", "R")
        # Sostituisce gli underscore con "a capo" per verticalizzare
        label = label.replace("_", "\n")
        return label

    # 4. Creiamo il dizionario filtrato con le etichette corte
    filtered_results = {
        f"BEST\n{clean_label(best_key)}": results[best_key],
        f"AVG\n{clean_label(mid_key)}": results[mid_key],
        f"WORST\n{clean_label(worst_key)}": results[worst_key]
    }
    
    return filtered_results

# Funzione per trovare il best assoluto
def find_global_best_config(results_pop, results_operators, results_criteria):
    """
    Trova la configurazione con la fitness media più alta tra TUTTI gli scenari.
    """
    all_configs = {}
    
    # Scenario 1
    for key, runs in results_pop.items():
        avg_fitness = np.mean([r['best_fitness'] for r in runs])
        all_configs[f"S1_Pop={key}"] = {
            'fitness': avg_fitness,
            'scenario': 1,
            'original_key': key,
            'runs': runs
        }
    
    # Scenario 2
    for key, runs in results_operators.items():
        avg_fitness = np.mean([r['best_fitness'] for r in runs])
        all_configs[f"S2_{key}"] = {
            'fitness': avg_fitness,
            'scenario': 2,
            'original_key': key,
            'runs': runs
        }
    
    # Scenario 3
    for key, runs in results_criteria.items():
        avg_fitness = np.mean([r['best_fitness'] for r in runs])
        all_configs[f"S3_{key}"] = {
            'fitness': avg_fitness,
            'scenario': 3,
            'original_key': key,
            'runs': runs
        }
    
    # Trova il massimo
    best_config_name = max(all_configs.items(), key=lambda x: x[1]['fitness'])[0]
    best_config_info = all_configs[best_config_name]
    
    return best_config_name, best_config_info

# Estrazione feature della configurazione ottimale
def extract_best_features(best_info, feature_names, top_k=20):
    """
    Estrae le feature più frequentemente selezionate dalla config vincitrice.
    """
    # Conta quante volte ogni feature appare nei 30 run
    feature_counter = {}
    
    for run in best_info['runs']:
        selected_features = run['best_features_idx']
        for feat_idx in selected_features:
            feature_counter[feat_idx] = feature_counter.get(feat_idx, 0) + 1
    
    # Ordina per frequenza decrescente
    sorted_features = sorted(feature_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Statistiche
    n_runs = len(best_info['runs'])
    
    print("\n" + "=" * 70)
    print("FEATURE SELECTION - RISULTATI FINALI")
    print("=" * 70)
    print(f"Configurazione: {best_info['original_key']}")
    print(f"Numero di run: {n_runs}")
    print(f"\nTop {top_k} Feature Selezionate:\n")
    print(f"{'Rank':<6} {'Feature':<40} {'Freq':<8} {'%':<8}")
    print("-" * 70)
    
    results_table = []
    
    for rank, (feat_idx, count) in enumerate(sorted_features[:top_k], 1):
        feat_name = feature_names[feat_idx]
        percentage = (count / n_runs) * 100
        print(f"{rank:<6} {feat_name:<40} {count:<8} {percentage:<8.1f}%")
        
        results_table.append({
            'rank': rank,
            'feature_idx': feat_idx,
            'feature_name': feat_name,
            'frequency': count,
            'percentage': percentage
        })
    
    # Numero medio di feature selezionate
    avg_n_features = np.mean([r['n_selected_features'] for r in best_info['runs']])
    std_n_features = np.std([r['n_selected_features'] for r in best_info['runs']])
    
    print("\n" + "-" * 70)
    print(f"Numero medio di feature selezionate: {avg_n_features:.1f} ± {std_n_features:.1f}")
    print("=" * 70)
    
    return results_table, sorted_features

# Funzione helper per filtrare il dizionario (per ottenere solo le configurazioni desiderate)
def filter_results(results, keys_to_keep):
    return {k: results[k] for k in keys_to_keep if k in results}




# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # HINT: Struttura suggerita per eseguire tutti gli esperimenti
    
    mode = 1 # 0 per eseguire gli scenari, 1 per ottenere i plot utilizzando i pkl

    # 1. Caricamento dati
    X, y = load_darwin_dataset("DARWIN.csv")
    

    print(X.dtypes)
    print(y.dtype)
    if mode == 0:
        # Pre-calcolo correlazioni per test
        r_cf, r_ff = precompute_correlations(X, y)
        
    
        ## Esecuzione scenari

        # 2. Esecuzione scenari-Si salvano i risultati i dei file .pkl da utilizzare poi per i grafici


        results_pop, counts_pop = run_experiment_population_size(X, y)
        with open("result_pop.pkl", "wb") as f: 
            pickle.dump(results_pop, f)
        with open("result_pop_counts.pkl", "wb") as f: 
            pickle.dump(counts_pop, f)
        
        
        results_operators, counts_operators = run_experiment_genetic_operators(X, y)
        with open("result_operators.pkl", "wb") as f: 
            pickle.dump(results_operators, f)
        with open("result_operators_counts.pkl", "wb") as f: 
            pickle.dump(counts_operators, f)

        
        results_stopping, counts_criteria = run_experiment_stopping_criteria(X, y)
        with open("results_criteria.pkl", "wb") as f: 
            pickle.dump(results_stopping, f)
        with open("result_criteria_count.pkl", "wb") as f: 
            pickle.dump(counts_criteria, f)

    else:
        ## Plots 


        #=============== PLOTS SCENARIO 2 ===============

        # Generare grafici -heatmap crossover vs mutation
        with open("result_operators.pkl", "rb") as f: 
            results_operators = pickle.load(f)
        # final
        plot_heatmap_operators(results_operators)
        results_operators_selection = prepare_data_by_selection(results_operators)
        # final
        plot_fitness_boxplots(results_operators_selection, "Tournament vs Roulette")
        # final
        results_operators_summury = extract_best_mid_worst(results_operators)
        plot_convergence_curves(results_operators_summury)

        #=============== PLOTS SCENARIO 3 ===============
        with open("results_criteria.pkl", "rb") as f: 
            result_criteria = pickle.load(f)
        # final
        plot_efficiency_and_stopping(result_criteria)
        # final
        plot_stopping_heatmap(result_criteria)


        #=============== PLOTS SCENARIO 1 ===============
        with open("result_pop.pkl", "rb") as f: 
            results_pop = pickle.load(f)
        # 2. Boxplot 
        #plot_fitness_boxplots(results_pop, title="Distribuzione Fitness per Dimensione Popolazione")
        # 3. Curve di Convergenza 
        #plot_convergence_curves(results_pop, title="Velocità di Convergenza per Popolazione")
    
        # 4. Trade-off Tempo/Fitness Final
        plot_population_tradeoff(results_pop)

        #=============== PLOTS CONTEGGIO FEATURE. Totale ===============
        feature_names = list(X.columns)
        best_name, best_info = find_global_best_config(results_pop, results_operators, result_criteria)
        results_table, all_features = extract_best_features(best_info, feature_names, top_k=20)
        #plot_final_feature_selection(results_table, title="Feature Ottimali per Classificazione Parkinson (Configurazione Best)")
        
        ##============================================
        ##========PLOT PER SCENARIO===================
        ##============================================

        #=============== PLOTS CONTEGGIO FEATURE X SCENARIO. ===============
        
        # plot-scenario pop
        #print(result_criteria.keys())

        # key configurazioni da plottare
        keys_scen1 = [20, 50, 100, 200, 500]
        keys_scen2 = [
            'CR=0.6_MR=0.01_SEL=roulette_K=None',  # La migliore Roulette
            'CR=0.9_MR=0.01_SEL=tournament_K=4'    # Il miglior Tournament
        ]
        keys_scen3 = [
            'MaxGen=100_Conv=10_Tol=1e-05',  # Impaziente (si ferma presto)
            'MaxGen=100_Conv=30_Tol=1e-05'   # Molto Paziente (insiste a cercare)
        ]
        
        ## FEATURE FREQUENCY
        # final
        plot_feature_frequency_comparison(
            results_pop, 
            keys_scen1, 
            feature_names, 
            title="Feature Stability: Popolazione"
        )
        # final
        plot_feature_frequency_comparison(
            results_operators,               # Il dizionario con i dati
            keys_scen2,                # Le due chiavi da confrontare
            feature_names,             # La lista dei nomi delle colonne (X.columns)
            title="Stabilità Feature: Roulette (0.65) vs Tournament (0.75)"
        )
        # final
        plot_feature_frequency_comparison(
            result_criteria, 
            keys_scen3, 
            feature_names, 
            title="Stabilità Feature: Convergence Bassa (10) vs Alta (30)"
        )

        results_ops_filtered = filter_results(results_operators, keys_scen2)
        results_crit_filtered = filter_results(result_criteria, keys_scen3)
        results_pop_filtered = results_pop

        ## BOX PLOT
        plot_fitness_boxplots(
            results_ops_filtered, 
            title="Distribuzione Fitness: Roulette vs Tournament"
        )
        # final
        plot_fitness_boxplots(
            results_crit_filtered, 
            title="Impatto della Convergenza sulla Fitness Finale (Conv=10 vs 30)"
        )
        # final
        plot_fitness_boxplots(
            results_pop_filtered,
            title="Impatto della popolazione"
        )

        ## CONVERGENCE PLOT
        """ plot_convergence_curves(
            results_ops_filtered, 
            title="Dinamica di Convergenza: Roulette vs Tournament"
        )
        plot_convergence_curves(
            results_crit_filtered, 
            title="Dinamica di Convergenza: Impaziente (10) vs Persistente (30)"
        )
        plot_convergence_curves(
            results_pop_filtered, 
            title="Dinamica di Convergenza: Popolazione"
        ) """
        
        ## CONVERGENCE & DIVERSITY PLOT
        # final
        plot_convergence_and_diversity(
            results_pop, 
            title="Dinamica Evolutiva: Fitness (Solida) vs Diversità (Tratteggiata)"
        )
        # final
        plot_convergence_and_diversity(
            results_crit_filtered, 
            title="Dinamica Evolutiva: Fitness (Solida) vs Diversità (Tratteggiata)"
        )
        """ plot_convergence_and_diversity(
            results_ops_filtered, 
            title="Dinamica Evolutiva: Fitness (Solida) vs Diversità (Tratteggiata)"
        ) """
    

  
        
