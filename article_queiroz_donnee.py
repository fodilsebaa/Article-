"""
Génération des instances selon Queiroz et al. (2023)
Dynamic scheduling of patients in emergency departments
European Journal of Operational Research, 310(1), 100-116

Basé sur:
- Dosi et al. (2019, 2020) pour l'Italie
- Kuo et al. (2016) pour Hong Kong
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMÈTRES GÉNÉRAUX
# ============================================================================

URGENCY_LEVELS = [1, 2, 3, 4]  # 1 = plus urgent, 4 = moins urgent
HOURS_PER_SHIFT = 4
SHIFTS = 6  # 6 groupes de 4 heures (00-04, 04-08, ..., 20-24)


# ============================================================================
# CONFIGURATION ITALIE (Dosi et al. 2019, 2020)
# ============================================================================

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
    # Source: Table 1 de l'article
    "urgency_distribution_percent": {
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


# ============================================================================
# CONFIGURATION HONG KONG (Kuo et al. 2016)
# ============================================================================

HONGKONG_CONFIG = {
    "patients_per_group": {
        1: [35, 69],
        2: [47, 74],
        3: [140, 169],
        4: [112, 139],
        5: [102, 138],
        6: [68, 98]
    },
    
    "urgency_distribution_percent": {
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
        0: 0.3, 1: 0.2, 2: 0.2, 3: 0.2,
        4: 0.3, 5: 0.5, 6: 1.0, 7: 1.8,
        8: 2.5, 9: 3.0, 10: 3.2, 11: 3.0,
        12: 2.8, 13: 2.5, 14: 2.2, 15: 2.0,
        16: 1.8, 17: 1.5, 18: 1.2, 19: 1.0,
        20: 0.8, 21: 0.6, 22: 0.4, 23: 0.3
    }
}


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_urgency_probabilities(config: dict, group_id: int) -> List[float]:
    """
    Retourne les probabilités normalisées pour chaque niveau d'urgence.
    CORRECTION : Normalisation pour que la somme = 1
    """
    probs = []
    for u in URGENCY_LEVELS:
        p = config["urgency_distribution_percent"][u][group_id - 1]
        probs.append(p)
    
    # Normalisation (correction du bug)
    total = sum(probs)
    if abs(total - 100) > 0.01:
        print(f"  Attention: somme des probabilités = {total}% (devrait être 100%)")
    
    normalized = [p / total for p in probs]
    return normalized


def generate_arrival_time(start_hour: int, end_hour: int, arrival_rates: dict) -> float:
    """
    Génère un temps d'arrivée selon une distribution exponentielle.
    """
    hour_of_day = np.random.randint(start_hour, end_hour)
    rate = arrival_rates.get(hour_of_day, 1.0)
    
    if rate > 0:
        # Distribution exponentielle pour l'inter-arrivée
        minute_offset = np.random.exponential(60 / rate)
    else:
        minute_offset = np.random.uniform(0, 60)
    
    # Limiter à l'heure courante
    minute_offset = min(minute_offset, 59.9)
    arrival_minutes = hour_of_day * 60 + minute_offset
    
    return arrival_minutes


# ============================================================================
# GÉNÉRATION PATIENTS ITALIE
# ============================================================================

def generate_italian_patients(group_id: int, seed: Optional[int] = None) -> List[dict]:
    """
    Génère les patients pour un groupe horaire donné (1 à 6)
    selon les spécifications de Dosi et al. (2019, 2020)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Déterminer le nombre de patients
    min_p, max_p = ITALY_CONFIG["patients_per_group"][group_id]
    n_patients = np.random.randint(min_p, max_p + 1)
    
    # 2. Définir l'intervalle de temps du groupe
    start_hour = (group_id - 1) * HOURS_PER_SHIFT
    end_hour = start_hour + HOURS_PER_SHIFT
    
    # 3. Obtenir les probabilités d'urgence normalisées
    urgency_probs = get_urgency_probabilities(ITALY_CONFIG, group_id)
    
    patients = []
    
    for i in range(n_patients):
        # Assigner le niveau d'urgence
        u = np.random.choice(URGENCY_LEVELS, p=urgency_probs)
        
        # Générer le temps d'arrivée
        arrival_minutes = generate_arrival_time(
            start_hour, end_hour, ITALY_CONFIG["arrival_rates"]
        )
        
        # Générer la famille de patients (f_j) - distribution uniforme
        family = np.random.choice(ITALY_CONFIG["service_time_params"]["families"])
        
        # Générer le temps de service (distribution normale tronquée)
        mean_st = ITALY_CONFIG["service_time_params"]["mean_base"] + 4 * family
        std_st = ITALY_CONFIG["service_time_params"]["std_base"] + 2 * family
        service_time = np.random.normal(mean_st, std_st)
        service_time = np.clip(
            service_time,
            ITALY_CONFIG["service_time_params"]["min"],
            ITALY_CONFIG["service_time_params"]["max"]
        )
        service_time = int(round(service_time))
        
        # Créer le dictionnaire du patient
        patient = {
            "patient_id": i,
            "arrival_time": arrival_minutes,
            "urgency_level": u,
            "due_date": arrival_minutes + ITALY_CONFIG["max_waiting_time"][u],
            "weight": ITALY_CONFIG["weights"][u],
            "service_time": service_time,
            "family": family,  # Italie uniquement
            "group": group_id
        }
        patients.append(patient)
    
    # Trier par temps d'arrivée
    patients.sort(key=lambda x: x["arrival_time"])
    
    # Réassigner les IDs dans l'ordre d'arrivée
    for idx, p in enumerate(patients):
        p["patient_id"] = idx
    
    return patients


# ============================================================================
# GÉNÉRATION PATIENTS HONG KONG
# ============================================================================

def generate_hongkong_patients(
    group_id: int, 
    with_early_info: bool = True, 
    seed: Optional[int] = None
) -> List[dict]:
    """
    Génère les patients pour Hong Kong selon Kuo et al. (2016)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Déterminer le nombre de patients
    min_p, max_p = HONGKONG_CONFIG["patients_per_group"][group_id]
    n_patients = np.random.randint(min_p, max_p + 1)
    
    # 2. Définir l'intervalle de temps du groupe
    start_hour = (group_id - 1) * HOURS_PER_SHIFT
    end_hour = start_hour + HOURS_PER_SHIFT
    
    # 3. Obtenir les probabilités d'urgence normalisées
    urgency_probs = get_urgency_probabilities(HONGKONG_CONFIG, group_id)
    
    patients = []
    
    for i in range(n_patients):
        # Niveau d'urgence
        u = np.random.choice(URGENCY_LEVELS, p=urgency_probs)
        
        # Temps d'arrivée
        arrival_minutes = generate_arrival_time(
            start_hour, end_hour, HONGKONG_CONFIG["arrival_rates"]
        )
        
        # Temps de service (distribution de Weibull)
        params = HONGKONG_CONFIG["service_time_weibull_params"][u]
        service_time = np.random.weibull(params["shape"]) * params["scale"]
        service_time = int(round(max(1, min(120, service_time))))
        
        # Créer le dictionnaire du patient (pas de 'family' pour Hong Kong)
        patient = {
            "patient_id": i,
            "arrival_time": arrival_minutes,
            "urgency_level": u,
            "due_date": arrival_minutes + HONGKONG_CONFIG["max_waiting_time"][u],
            "weight": HONGKONG_CONFIG["weights"][u],
            "service_time": service_time,
            "group": group_id,
            "has_early_info": False,
            "early_info_time": None
        }
        
        # Ajouter l'information anticipée (spécifique à Hong Kong)
        if with_early_info:
            if np.random.random() < HONGKONG_CONFIG["early_information"]["percentage"]:
                call_ahead = np.random.normal(
                    HONGKONG_CONFIG["early_information"]["call_ahead_mean"],
                    HONGKONG_CONFIG["early_information"]["call_ahead_std"]
                )
                call_ahead = max(1, min(15, call_ahead))
                patient["has_early_info"] = True
                patient["early_info_time"] = arrival_minutes - call_ahead
        
        patients.append(patient)
    
    # Trier par temps d'arrivée
    patients.sort(key=lambda x: x["arrival_time"])
    
    # Réassigner les IDs dans l'ordre d'arrivée
    for idx, p in enumerate(patients):
        p["patient_id"] = idx
    
    return patients


# ============================================================================
# GÉNÉRATION DE TOUTES LES INSTANCES
# ============================================================================

def generate_all_instances(
    hospital: str = "italy", 
    n_days: int = 31, 
    output_file: str = None
) -> pd.DataFrame:
    """
    Génère toutes les instances pour un hôpital donné.
    
    Args:
        hospital: "italy" ou "hongkong"
        n_days: Nombre de jours à générer (31 pour Italie, 5 pour Hong Kong)
        output_file: Fichier CSV de sortie (optionnel)
    
    Returns:
        DataFrame contenant tous les patients
    """
    all_patients = []
    
    print(f"\n{'='*60}")
    print(f"Génération des instances pour {hospital.upper()}")
    print(f"Nombre de jours: {n_days}")
    print(f"{'='*60}")
    
    for day in range(n_days):
        for group in range(1, 7):
            seed = day * 100 + group * 10
            
            if hospital.lower() == "italy":
                patients = generate_italian_patients(group, seed=seed)
            else:
                patients = generate_hongkong_patients(group, with_early_info=True, seed=seed)
            
            for p in patients:
                p["day"] = day
                p["hospital"] = hospital
                all_patients.append(p)
            
            # Afficher la progression
            if group == 6:
                total_in_group = len(patients)
                print(f"  Jour {day+1:2d}: {total_in_group:3d} patients")
    
    # Créer le DataFrame
    df = pd.DataFrame(all_patients)
    
    # Ajouter un identifiant unique d'instance
    df["instance_id"] = df["day"].astype(str) + "_" + df["group"].astype(str)
    
    # Définir l'ordre des colonnes en fonction de l'hôpital
    base_columns = [
        "instance_id", "day", "group", "patient_id",
        "arrival_time", "service_time", "due_date",
        "urgency_level", "weight", "hospital"
    ]
    
    if hospital.lower() == "italy":
        # Italie: inclure la colonne 'family'
        column_order = base_columns.copy()
        # Insérer 'family' après 'weight'
        idx = column_order.index("weight")
        column_order.insert(idx + 1, "family")
    else:
        # Hong Kong: inclure les colonnes d'information anticipée
        column_order = base_columns.copy()
        column_order.extend(["has_early_info", "early_info_time"])
    
    # Sélectionner les colonnes qui existent réellement
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    # Sauvegarder
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\n✓ Fichier sauvegardé: {output_file}")
    
    print(f"\nTotal patients générés: {len(df)}")
    print(f"Nombre d'instances: {df['instance_id'].nunique()}")
    
    return df


# ============================================================================
# VALIDATION DES INSTANCES
# ============================================================================

def validate_instances(df: pd.DataFrame, hospital: str):
    """
    Valide que les instances respectent les distributions attendues.
    """
    print(f"\n{'='*60}")
    print(f"VALIDATION - {hospital.upper()}")
    print(f"{'='*60}")
    
    # 1. Vérification des niveaux d'urgence par groupe
    print("\n1. Distribution des niveaux d'urgence par groupe:")
    print("-" * 50)
    
    if hospital.lower() == "italy":
        expected = ITALY_CONFIG["urgency_distribution_percent"]
    else:
        expected = HONGKONG_CONFIG["urgency_distribution_percent"]
    
    for group in range(1, 7):
        group_df = df[df["group"] == group]
        if len(group_df) == 0:
            continue
        
        dist = group_df["urgency_level"].value_counts(normalize=True).sort_index()
        
        # Convertir en pourcentage
        dist_pct = {u: dist.get(u, 0) * 100 for u in URGENCY_LEVELS}
        
        print(f"\n  Groupe {group} (n={len(group_df)}):")
        for u in URGENCY_LEVELS:
            exp = expected[u][group-1]
            actual = dist_pct[u]
            diff = abs(exp - actual)
            ok = "✓" if diff < 5 else "⚠️"
            print(f"    u={u}: {actual:.1f}% (attendu: {exp}%) {ok}")
    
    # 2. Statistiques des temps de service
    print("\n2. Statistiques des temps de service:")
    print("-" * 50)
    print(f"  Moyenne: {df['service_time'].mean():.1f} min")
    print(f"  Médiane: {df['service_time'].median():.1f} min")
    print(f"  Écart-type: {df['service_time'].std():.1f} min")
    print(f"  Min: {df['service_time'].min()} min")
    print(f"  Max: {df['service_time'].max()} min")
    
    # 3. Information anticipée (Hong Kong uniquement)
    if hospital.lower() == "hongkong" and "has_early_info" in df.columns:
        ei_percentage = df["has_early_info"].mean() * 100
        print(f"\n3. Information anticipée (Early Information):")
        print("-" * 50)
        print(f"  Pourcentage: {ei_percentage:.1f}% (attendu: 18.62%)")
        
        ei_times = df[df["has_early_info"]]["early_info_time"].dropna()
        if len(ei_times) > 0:
            avg_call_ahead = (df[df["has_early_info"]]["arrival_time"] - ei_times).mean()
            print(f"  Temps moyen d'anticipation: {avg_call_ahead:.1f} min (attendu: 7.32 min)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("GÉNÉRATION DES INSTANCES SELON QUEIROZ ET AL. (2023)")
    print("="*60)
    
    # 1. Générer les instances Italie (31 jours × 6 groupes = 186 instances)
    print("\n" + "="*60)
    print("HÔPITAL D'ITALIE")
    print("="*60)
    italy_df = generate_all_instances(
        hospital="italy",
        n_days=31,
        output_file="italy_instances.csv"
    )
    
    # 2. Générer les instances Hong Kong (5 jours × 6 groupes = 30 instances)
    print("\n" + "="*60)
    print("HÔPITAL DE HONG KONG")
    print("="*60)
    hongkong_df = generate_all_instances(
        hospital="hongkong",
        n_days=5,
        output_file="hongkong_instances.csv"
    )
    
    # 3. Validation
    if italy_df is not None and len(italy_df) > 0:
        validate_instances(italy_df, "italy")
    
    if hongkong_df is not None and len(hongkong_df) > 0:
        validate_instances(hongkong_df, "hongkong")
    
    # 4. Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    if italy_df is not None:
        print(f"✓ Italie: {italy_df['instance_id'].nunique()} instances, {len(italy_df)} patients")
    if hongkong_df is not None:
        print(f"✓ Hong Kong: {hongkong_df['instance_id'].nunique()} instances, {len(hongkong_df)} patients")
    print(f"\nFichiers générés:")
    print(f"  - italy_instances.csv")
    print(f"  - hongkong_instances.csv")