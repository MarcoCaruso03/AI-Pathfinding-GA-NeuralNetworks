# Tesina 2 - Ottimizzazione dei Percorsi di Distribuzione Energetica

## Problema

Una società di distribuzione energetica deve ottimizzare la manutenzione e il rifornimento di **20 stazioni di ricarica elettrica** in una rete urbana interconnessa.

---

## Fasi del Progetto

### Fase 1: Sviluppo dell'Euristica

Implementare una **heuristic function** personalizzata per l'algoritmo **A***.

Considerare criteri come:
- Minimizzazione dei costi di attraversamento
- Efficienza energetica
- Criticità delle stazioni

### Fase 2: Simulazione dei Percorsi

Verranno simulati **3 diversi scenari** di percorso:

---

#### 🚨 Scenario di Emergenza

| Parametro | Valore |
|-----------|--------|
| **Punto di partenza** | Stazione 1 (adiacente al centro) |
| **Obiettivo** | Raggiungere la stazione critica 17 (West) nel minor tempo possibile |
| **Vincoli** | Max consumo energetico: 800, Max tempo: 400 |

---

#### 🔧 Scenario di Manutenzione Programmata

| Parametro | Valore |
|-----------|--------|
| **Punto di partenza** | Stazione 6 (North-East) |
| **Obiettivo** | Visitare le stazioni **5 → 12 → 8** in ordine predefinito |
| **Vincoli** | Distanza totale massima: 1800 |

---

#### ⚡ Scenario di Bilanciamento Energetico

| Parametro | Valore |
|-----------|--------|
| **Punto di partenza** | Stazione 11 (South extreme) |
| **Obiettivo** | Collegare le 4 stazioni con basso livello di energia: **5, 10, 14, 18** |
| **Vincoli** | Distanza totale massima: 1400 |

---

## Struttura della Rete

### Stazioni Critiche
| ID Stazione | Livello Energia | Stato |
|-------------|-----------------|-------|
| 3 | 25% | Critica |
| 17 | 18% | Critica |

### Stazioni a Bassa Energia
| ID Stazione | Livello Energia |
|-------------|-----------------|
| 5 | 38% |
| 10 | 32% |
| 14 | 28% |
| 18 | 22% |

### Connessioni
- **Connessioni standard**: peso 120
- **Connessioni cross-area**: peso 180

---

## Output Richiesti per Ogni Scenario

- ✅ Percorso ottimale trovato
- 📊 Costo totale del percorso
- 🔢 Nodi attraversati in sequenza
- 🧠 Analisi delle scelte dell'euristica
- 🔄 Confronto con percorsi alternativi

---

## Criteri di Valutazione

- Correttezza implementazione **A***
- Efficacia della **heuristic function**
- Performance dei percorsi calcolati
- Capacità di adattamento a scenari diversi

---

## Requisiti Tecnici

- Implementare gestione dei pesi degli archi
- Dimostrare flessibilità dell'euristica
- Produrre una relazione dettagliata che spieghi le scelte implementative
- Documentare il processo di sviluppo dell'euristica
- Spiegare i criteri di ottimizzazione utilizzati
- Fornire visualizzazione grafica dei percorsi

---

## Da Fornire prima dell'Orale

- 📁 **Codice completo** (repository GitHub)
- 📝 **Report** con plot, scelte adottate e motivazioni

---

## Esecuzione

```bash
# Scenario di Emergenza
python main.py --scenario emergency

# Scenario di Manutenzione
python main.py --scenario maintenance

# Scenario di Bilanciamento
python main.py --scenario balancing

# Aiuto
python main.py --help
```