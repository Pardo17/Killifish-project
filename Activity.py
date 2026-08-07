import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import os #os è un modulo della libreria standard di Python
import datetime
import numpy as np
import math
import stumpy
import warnings

#todo: dopo aver fatto girare, stumpy mi dava dei "warnings" per alcuni segmenti => no motifs were found,
# you may consider increasing the 'cutoff' (quello impostato è di default). Ho aggiunto per questo un counter per
# segmenti con e senza motifs, e un modo per ignorare questi warnings.
warnings.filterwarnings('ignore', category=UserWarning)

files = [f for f in os.listdir(r"C:\Users\Giacomo\Desktop\IF, 6-6, 111\Hours 6") if f.startswith("hour") and "__" not in f]
infos = [
  f.replace("hour", "")\
   .replace("-", "")\
   .replace("_", "")\
   .replace("192.168.8.", "")\
   .replace("fishdetailedsizes.txt", ".csv")
 for f in files
]

infos = [
  [
    int(file_info[:2]), # hour
    int(file_info[2:6]), # year
    int(file_info[6:8]), # month
    int(file_info[8:10]), # day
    file_info[10:-4], #id
    datetime.datetime(int(file_info[2:6]), int(file_info[6:8]), int(file_info[8:10]), int(file_info[:2]))
  ]
  for file_info in infos
]

files_with_infos = [[f] + i for f, i in zip(files, infos)]
files_with_infos.sort(key=lambda x: (x[5], x[6])) #ordino per fish_id e datetime prima del processamento.
#con key = l gli dico che per ogni x nella lista, ordina in base alla tupla => prima guarda indice 5 (id), poi datetime

df_list_ = []
fish_dict = {
    "111": {"nutrition": "IF", "T-cycles": 6},
    "114": {"nutrition": "IF", "T-cycles": 6},
    "115": {"nutrition": "IF", "T-cycles": 6},
    "116": {"nutrition": "IF", "T-cycles": 6},
    "120": {"nutrition": "IF", "T-cycles": 6},
    "129": {"nutrition": "IF", "T-cycles": 12},
    "109": {"nutrition": "AL", "T-cycles": 6},
    "121": {"nutrition": "AL", "T-cycles": 12},
    "122": {"nutrition": "AL", "T-cycles": 12},
    "124": {"nutrition": "AL", "T-cycles": 12},
    "132": {"nutrition": "AL", "T-cycles": 6},
    "144": {"nutrition": "AL", "T-cycles": 18},
}

for file, hour, year, month, day, fish_id, dt in files_with_infos:
    path = os.path.join(r"C:\Users\Giacomo\Desktop\IF, 6-6, 111\Hours 6", file)
    df = pd.read_csv(path)

    df["file"] = file #aggiungo colonna con il nome del file
    df["hour"] = hour
    df["year"] = year
    df["month"] = month
    df["day"] = day
    df["fish_id"] = fish_id
    df["minute"] = df["frame"]//60 #con doppio slash ho divisione intera
    df["second"] = df["frame"] % 60  # % dà il resto della divisione direttamente (i secondi)

    df["datetime"] = dt
    df["datetime"] = df["datetime"] + pd.to_timedelta(df["minute"], unit='m') + pd.to_timedelta(df["second"], unit='s')

    if fish_id in fish_dict:
        df["nutrition"] = fish_dict[fish_id]["nutrition"]
        df["T-cycles"] = fish_dict[fish_id]["T-cycles"]
    else:
        df["nutrition"] = None
        df["T-cycles"] = None

    df_list_.append(df) #attacco tutti i dataset alla lista df_list, perché a quanto pare .concat legge le liste

dataset = pd.concat(df_list_).reset_index(drop=True) #ho ordinato il dataset sia per id che per datetime.

dataset["time_diff"] = dataset["datetime"].diff() #.diff() = calcola la differenza tra datetime consecutivi (è una funzione Pandas)
#potevo usare datetime.timedelta di Python invece di pd.Timedelta
dataset["hole"] = dataset["time_diff"] > pd.Timedelta("1sec") #creo una colonna booleana che indica i buchi > 1 secondo
dataset["missing_frames"] = dataset["time_diff"] / pd.Timedelta("1sec") #in questo modo vedo quanti frame mancano. se non c'è gap, sarà 1
dataset["segment_id"] = (dataset["missing_frames"] > 10).cumsum() # cumsum incrementa un contatore ogni volta che c'è un buco >10
#quindi segment_id = [0, 0, 1, 1, 1, 2, 2, 2]x es. e tutti i dati con lo stesso numero stanno nello stesso segmento continuo

dataset["travel_x"] = dataset["center_x"].diff()
dataset["travel_y"] = dataset["center_y"].diff()
dataset["travelled_distance"] = dataset[["travel_x", "travel_y"]].apply(
        lambda row: np.sqrt(row["travel_x"] ** 2 + row["travel_y"] ** 2),
        axis=1,)

# Dizionario: ogni chiave è (fish_id, segment_id), ogni valore è una time series
time_series_dict = {}

for fish_id in dataset['fish_id'].unique(): #unique estrae tutti i valori, non ripetuti, di una colonna
    for segment_id in dataset[dataset['fish_id'] == fish_id]['segment_id'].unique():
        segment_data = dataset[(dataset['fish_id'] == fish_id) & (dataset['segment_id'] == segment_id)] #non posso usare and, & è adatto a pandas
#quindi in questo modo ho filtrato il dataset e ottenuto un dataset con solo un fish_id e un segment_it
        ts = pd.Series(
            segment_data['travelled_distance'].values, #sull'asse y ho la distanza euclidea
            index=segment_data['datetime'].values, #indici i datetime, quindi sull'asse x c'è il tempo
            name=f"{fish_id}_{segment_id}" #dà il nome alla time series, che dovrebbe apparire anche come titolo del grafico
        )

        time_series_dict[(fish_id, segment_id)] = ts #aggiungo la time series al dizionario, usando la tupla id, segment_id come chiave

def plot_ts(time_series_dict, y_max, output_path, smooth_window): #y_max sarà dataset["travel distance"].max()
    """Plotta tutte le ts e le carica nel file => output_path"""

    for (fish_id, segment_id), ts in time_series_dict.items():
        ts = ts.dropna().rolling(window=smooth_window).mean() #dropna leva i Nan, poi smoothing

        plt.figure(figsize=(12, 5))
        plt.plot(ts.index, ts.values, color='blue')
        plt.title(f"Fish {fish_id}, Segment {segment_id}")
        plt.xlabel("Datetime")
        plt.ylabel("Travelled Distance")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.ylim(0, y_max)
        plt.tight_layout()

        filename = f"fish_{fish_id}_segment_{segment_id}.png" #.png salva come png
        filepath = os.path.join(output_path, filename)
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white') #facecolor white => sfondo bianco
        #bbox_inches tight => rimuove spazio bianco intorno al grafico
        #dpi 200 => qualità un po' più alta di quella di default, dice che è ottimale per la ricerca dei motif
        plt.close()

    return "Plots saved"

#print(plot_ts(time_series_dict, dataset["travelled_distance"].max(),
# r"C:\Users\Giacomo\Desktop\Projects\thesis\ts_prova_smooth", 10))

#todo: usare pickle

window_size = 10 #quindi il numero di frame che considero nella finestra
#se aumento w_s trovo pattern più lunghi

mp_results = {} #dict per risultati. key => tupla id, valore => matrix profile
for (fish_id, segment_id), ts in time_series_dict.items():
    ts_clean = ts.dropna().values #dropna toglie i Nan

    if len(ts_clean) < window_size * 2: #Ha bisogno di almeno 2 finestre per calcolare il MP (deve comparare)
        continue

    mp = stumpy.stump(ts_clean, m=window_size) #mp è una matrice con 2 colonne: distanze e indici
    mp_distance = mp[:, 0] #le disanze del matrix profile
    mp_results[(fish_id, segment_id)] = mp_distance #ne faccio un dict così come avevo fatto con le ts

print(f"Matrix Profile calcolati: {len(mp_results)} segmenti")


def plot_and_save_motifs(time_series_dict, mp_results, output_path):
    """Motif discovery usando stumpy, e salvataggio in file"""

    if len(mp_results) == 0: #se il dict è vuoto da errore
        raise ValueError("Nessun segmento valido per il motif discovery")

    found_count = 0 #numero di segmenti con motif trovati
    no_motif_count = 0 #numero di segmenti senza motif

    for (fish_id, segment_id), mp_distance in mp_results.items(): #mp_distance sono le distanze del matrix profile
        ts_clean = time_series_dict[(fish_id, segment_id)].dropna().values #dropna toglie i Nan dalle ts... #todo: prof

#con Ctrl + click posso aprire il link
        motif_distances, motif_indices = stumpy.motifs( #https://stumpy.readthedocs.io/en/latest/api.html#motifs
            ts_clean,
            mp_distance,
            max_motifs=2,  #prende solo i 2 pattern più comuni dentro una sola ts
            max_matches=4 #mi trova le 4 ripetizioni migliori.
        )
#motif_indices è una lista di liste, in cui ogni sottolista contiene gli indici (della ts_clean)
        # in cui si ripetono i motif. se ad esempio ts_clean ha 77 frame, gli indici vanno da 0 a 76.

        if len(motif_indices) == 0:
            no_motif_count += 1
            continue #passa al segmento successivo quando trova un segmento senza motif
            #in questo modo non salva nessun png per quel segmento
        found_count += 1

        plt.figure(figsize=(12, 5)) #le dimensioni della figura
        plt.plot(ts_clean, color='lightgray', linewidth=1)
        colors = ['red', 'green'] #colori dei motif

        for motif, (indices, color) in enumerate(zip(motif_indices, colors)): #accoppio indici motif ai colori
            for idx in indices:
            #controlli aggiunti perché a volte i limiti di lunghezza della ts venivano superati (vedi documentazione)
                if idx < 0 or idx + window_size > len(ts_clean):
                    continue

                motif_segment = ts_clean[idx:idx + window_size]
                if len(motif_segment) == window_size:
                    plt.plot(range(idx, idx + window_size), motif_segment,
                             color=color, linewidth=2)

        plt.title(f"Fish {fish_id}, Segment {segment_id}")
        plt.xlabel("Frame")
        plt.ylabel("Travelled Distance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"motifs_fish_{fish_id}_segment_{segment_id}.png"
        filepath = os.path.join(output_path, filename)
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

    return f"Elaborazione completata: {found_count} motif trovati, {no_motif_count} senza motif"

#print(plot_and_save_motifs(time_series_dict, mp_results,
                     #r"C:\Users\Giacomo\Desktop\Projects\thesis\mp_prova"))

"""Consensus motif"""
#todo: non posso usare mp_results e la funzione dei motif precedente perché:
# "per il consensus motif, dopo che stumpy.ostinato() trova il motif condiviso,
# tu hai bisogno di trovare dove quel motif appare in ogni segmento, e per farlo devi usare
# una ricerca di similarità (MASS) tra il consensus motif e ogni singola ts, e non una matrix profile
# come quella di prima


def fish_data_quality(dataset):
    """
    Valuta i frame mancanti per ogni pesce
    """
    quality = {} #dizionario dove metto le info e che ritorno

    for fish_id in dataset['fish_id'].unique():
        fish_data = dataset[dataset['fish_id'] == fish_id].sort_values('datetime')

        # Frame mancanti totali
        total_missing_frames = (fish_data[fish_data['missing_frames'] > 1]['missing_frames'] - 1).sum() #selezione righe e colonna, così ho solo i valori che mi interessano

        # Frame registrati
        total_frames = len(fish_data) #tutti i frame che ho registrato

        # Percentuale di frame mancanti
        missing_percentage = (total_missing_frames / (total_frames + total_missing_frames) * 100)

        quality[fish_id] = {
            "total_frames": int(total_frames),
            "missing_frames": float(total_missing_frames),
            "missing_percentage": float(missing_percentage),
        }

    return quality
#print(fish_data_quality(dataset))


"""PROVA CON PESCE 111, TIME SERIES LIFESPAN"""
filtered_111 = dataset[dataset["fish_id"] == "111"].sort_values('datetime')
ts_x = pd.Series(filtered_111["center_x"].values[:200], index=filtered_111["datetime"].values[:200]) #time series con x nel tempo
#plt.plot(ts_x) #plotta la time series
#plt.show() #senza questa il grafico non viene mostrato

ts_y = pd.Series(filtered_111["center_y"].values[:200], index=filtered_111["datetime"].values[:200]) #time series con y nel tempo
#plt.plot(ts_y)
#plt.show()


