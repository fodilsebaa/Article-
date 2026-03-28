import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Paramètres généraux
URGENCY_LEVELS = [1, 2, 3, 4]  # 1 = plus urgent, 4 = moins urgent
HOURS_PER_SHIFT = 4
SHIFTS = 6  # 6 groupes de 4 heures (00-04, 04-08, ..., 20-24)
ITALY_CONFIG = {
    # Pour chaque groupe (1 à 6) : [min_patients, max_patients]
    "patients_per_group": {
        1: [5, 17],   # 00:00 - 04:00
        2: [5, 17],   # 04:00 - 08:00
        3: [41, 62],  # 08:00 - 12:00
        4: [34, 57],  # 12:00 - 16:00
        5: [25, 49],  # 16:00 - 20:00
        6: [7, 37]    # 20:00 - 00:00
    },
    
    # Pourcentage de patients par niveau d'urgence (par groupe)
    "urgency_distribution": {
        1: [2.96, 3.13, 2.16, 3.04, 2.76, 2.58],  # % pour u=1 par groupe
        2: [34.02, 23.76, 24.84, 29.60, 28.62, 34.53],  # % pour u=2
        3: [51.78, 60.05, 57.37, 54.90, 59.73, 51.86],  # % pour u=3
        4: [11.24, 13.05, 15.63, 12.46, 8.89, 11.03]     # % pour u=4
    },
    
    # Temps d'attente maximum (minutes) par niveau d'urgence
    "max_waiting_time": {
        1: 0,
        2: 15,
        3: 60,
        4: 120
    },
    
    # Poids pour la fonction objectif
    "weights": {
        1: 4,
        2: 3,
        3: 2,
        4: 1
    },
    
    # Paramètres du temps de service (distribution normale tronquée)
    "service_time_params": {
        "min": 1,
        "max": 60,
        "mean_base": 9,  # 9 + 4*f_j
        "std_base": 3,   # 3 + 2*f_j
        "families": [0, 1, 2, 3, 4]  # f_j = famille de symptômes
    },
    
    # Taux d'arrivée par heure (exponentiel)
    "arrival_rates": {
        0: 0.5, 1: 0.4, 2: 0.3, 3: 0.3,   # Nuit
        4: 0.4, 5: 0.6, 6: 1.0, 7: 1.5,   # Matin
        8: 2.0, 9: 2.2, 10: 2.0, 11: 1.8, # Matinée
        12: 1.5, 13: 1.4, 14: 1.3, 15: 1.2, # Après-midi
        16: 1.2, 17: 1.3, 18: 1.4, 19: 1.2, # Soir
        20: 1.0, 21: 0.8, 22: 0.6, 23: 0.5  # Nuit
    }
}
def generate_italian_patients(group_id, seed=None):
    """
    Génère les patients pour un groupe horaire donné (1 à 6)
    """
    if seed:
        np.random.seed(seed)
    
    # 1. Déterminer le nombre de patients
    min_p, max_p = ITALY_CONFIG["patients_per_group"][group_id]
    n_patients = np.random.randint(min_p, max_p + 1)
    
    # 2. Définir l'intervalle de temps du groupe
    start_hour = (group_id - 1) * HOURS_PER_SHIFT
    end_hour = start_hour + HOURS_PER_SHIFT
    start_time = start_hour * 60  # en minutes
    end_time = end_hour * 60
    
    patients = []
    
    # 3. Générer chaque patient
    for i in range(n_patients):
        # Assigner le niveau d'urgence
        u = np.random.choice(
            URGENCY_LEVELS,
            p=[ITALY_CONFIG["urgency_distribution"][u][group_id-1] / 100 
               for u in URGENCY_LEVELS]
        )
        
        # Générer le temps d'arrivée (distribution exponentielle)
        hour_of_day = np.random.randint(start_hour, end_hour)
        rate = ITALY_CONFIG["arrival_rates"][hour_of_day]
        # Ajouter une variation intra-heure
        minute_offset = np.random.exponential(60 / rate) if rate > 0 else np.random.uniform(0, 60)
        arrival_minutes = hour_of_day * 60 + min(minute_offset, 59.9)
        
        # Générer la famille de patients (f_j)
        family = np.random.choice(ITALY_CONFIG["service_time_params"]["families"])
        
        # Générer le temps de service (distribution normale tronquée)
        mean_st = ITALY_CONFIG["service_time_params"]["mean_base"] + 4 * family
        std_st = ITALY_CONFIG["service_time_params"]["std_base"] + 2 * family
        service_time = np.random.normal(mean_st, std_st)
        service_time = np.clip(service_time, 
                               ITALY_CONFIG["service_time_params"]["min"],
                               ITALY_CONFIG["service_time_params"]["max"])
        service_time = int(round(service_time))
        
        # Créer le patient
        patient = {
            "id": i,
            "arrival_time": arrival_minutes,
            "urgency_level": u,
            "due_date": arrival_minutes + ITALY_CONFIG["max_waiting_time"][u],
            "weight": ITALY_CONFIG["weights"][u],
            "service_time_expected": service_time,
            "service_time_realized": service_time,  # Dans la version déterministe
            "family": family,
            "group": group_id
        }
        patients.append(patient)
    
    # Trier par temps d'arrivée
    patients.sort(key=lambda x: x["arrival_time"])
    return patients
HONGKONG_CONFIG = {
    "patients_per_group": {
        1: [35, 69],
        2: [47, 74],
        3: [140, 169],
        4: [112, 139],
        5: [102, 138],
        6: [68, 98]
    },
    
    "urgency_distribution": {
        1: [0.36, 0.96, 0.51, 0.31, 0.70, 0.48],
        2: [2.88, 2.25, 2.19, 3.40, 2.97, 4.78],
        3: [33.45, 27.33, 22.37, 32.92, 34.44, 36.36],
        4: [63.31, 69.45, 74.94, 63.37, 61.89, 58.37]
    },
    
    "max_waiting_time": {
        1: 0,
        2: 10,
        3: 30,
        4: 180
    },
    
    "weights": {
        1: 4,
        2: 3,
        3: 2,
        4: 1
    },
    
    # Service time: distribution de Weibull
    "service_time_weibull_params": {
        1: {"shape": 2.5, "scale": 20},   # u=1
        2: {"shape": 2.2, "scale": 15},
        3: {"shape": 2.0, "scale": 12},
        4: {"shape": 1.8, "scale": 10}
    },
    
    "early_information": {
        "percentage": 0.1862,  # 18.62% des patients
        "call_ahead_mean": 7.32,  # minutes en moyenne
        "call_ahead_std": 2.5
    },
    
    "arrival_rates": {
        # Heures d'arrivée pour Hong Kong (plus de patients en journée)
        0: 0.3, 1: 0.2, 2: 0.2, 3: 0.2,
        4: 0.3, 5: 0.5, 6: 1.0, 7: 1.8,
        8: 2.5, 9: 3.0, 10: 3.2, 11: 3.0,
        12: 2.8, 13: 2.5, 14: 2.2, 15: 2.0,
        16: 1.8, 17: 1.5, 18: 1.2, 19: 1.0,
        20: 0.8, 21: 0.6, 22: 0.4, 23: 0.3
    }
}

def generate_hongkong_patients(group_id, with_early_info=True, seed=None):
    """Génère les patients pour Hong Kong"""
    if seed:
        np.random.seed(seed)
    
    min_p, max_p = HONGKONG_CONFIG["patients_per_group"][group_id]
    n_patients = np.random.randint(min_p, max_p + 1)
    
    start_hour = (group_id - 1) * HOURS_PER_SHIFT
    end_hour = start_hour + HOURS_PER_SHIFT
    
    patients = []
    
    for i in range(n_patients):
        # Niveau d'urgence
        u = np.random.choice(
            URGENCY_LEVELS,
            p=[HONGKONG_CONFIG["urgency_distribution"][u][group_id-1] / 100 
               for u in URGENCY_LEVELS]
        )
        
        # Temps d'arrivée
        hour_of_day = np.random.randint(start_hour, end_hour)
        rate = HONGKONG_CONFIG["arrival_rates"][hour_of_day]
        minute_offset = np.random.exponential(60 / rate) if rate > 0 else np.random.uniform(0, 60)
        arrival_minutes = hour_of_day * 60 + min(minute_offset, 59.9)
        
        # Temps de service (Weibull)
        params = HONGKONG_CONFIG["service_time_weibull_params"][u]
        service_time = np.random.weibull(params["shape"]) * params["scale"]
        service_time = int(round(max(1, min(120, service_time))))  # entre 1 et 120 min
        
        patient = {
            "id": i,
            "arrival_time": arrival_minutes,
            "urgency_level": u,
            "due_date": arrival_minutes + HONGKONG_CONFIG["max_waiting_time"][u],
            "weight": HONGKONG_CONFIG["weights"][u],
            "service_time_expected": service_time,
            "service_time_realized": service_time,
            "group": group_id,
            "has_early_info": False,
            "early_info_time": None
        }
        
        # Ajouter l'information anticipée
        if with_early_info:
            if np.random.random() < HONGKONG_CONFIG["early_information"]["percentage"]:
                call_ahead = np.random.normal(
                    HONGKONG_CONFIG["early_information"]["call_ahead_mean"],
                    HONGKONG_CONFIG["early_information"]["call_ahead_std"]
                )
                call_ahead = max(1, min(15, call_ahead))
                patient["has_early_info"] = True
                patient["early_info_time"] = arrival_minutes - call_ahead
                patient["early_info_arrival"] = arrival_minutes - call_ahead
        
        patients.append(patient)
    
    patients.sort(key=lambda x: x["arrival_time"])
    return patients
def generate_all_instances(hospital="italy", n_days=31, output_file="instances.csv"):
    """
    Génère toutes les instances pour un hôpital donné
    """
    all_patients = []
    
    for day in range(n_days):
        for group in range(1, 7):
            if hospital == "italy":
                patients = generate_italian_patients(group, seed=day*10 + group)
            else:
                patients = generate_hongkong_patients(group, seed=day*10 + group)
            
            for p in patients:
                p["day"] = day
                p["hospital"] = hospital
                all_patients.append(p)
    
    df = pd.DataFrame(all_patients)
    
    # Ajouter des métadonnées
    df["instance_id"] = df["day"].astype(str) + "_" + df["group"].astype(str)
    
    # Sauvegarder
    df.to_csv(output_file, index=False)
    print(f"Généré {len(df)} patients pour {hospital}")
    
    return df

# Générer les instances
italy_instances = generate_all_instances("italy", n_days=31, output_file="italy_instances.csv")
hongkong_instances = generate_all_instances("hongkong", n_days=5, output_file="hongkong_instances.csv")