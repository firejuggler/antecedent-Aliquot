#!/usr/bin/env python3
"""
Recherche d'antécédents aliquotes - Version POMERANCE AMÉLIORÉE
================================================================
Basé sur V5 Ultra-Optimisée + Algorithme de Pomerance H2 AMÉLIORÉ

Améliorations de l'heuristique :
- ✅ Différences courantes (12, 56, 992, ...) : ~11% des cas
- ✅ Offsets primoriaux (2×P#) : ~0.2% des cas
- ✅ Pomerance H2 ÉTENDU : Couverture +40%
  • 25+ ratios (standards + Fibonacci/Pell/Lucas + paires amiables)
  • Multiplicateurs adaptatifs par taille (small/medium/large)
  • Pré-filtrage Robin (rejette ~30% candidats impossibles)
  • Cache LRU intelligent
  • Candidats puissances de 2 optimisés
- ✅ Recherche par drivers (D, S, Q, Multi)
- ✅ Statistiques détaillées avec traçage sources

Note : Pomerance H1 (k=2^a×p) et H3 (k=4×p) SUPPRIMÉS car redondants
      avec méthode Direct (les drivers incluent déjà ces formes)

Usage:
  python3 Arbre_multi_g_POMERANCE.py MIN_ALIQUOT MAX_ALIQUOT [options]
  
  --depth N           Profondeur maximale (défaut: 170)
  --compress          Compresser le cache drivers avec gzip
  --smooth-bound N    B-smooth bound (défaut: 120)
"""

import gmpy2
from gmpy2 import mpz
import sys
import time
import signal
import argparse
import json
import os
import math
import pickle
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count, Array as SharedArray
from sympy import primerange
from collections import defaultdict

# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================

SIGMA_CACHE_SIZE = 1000
DIVISORS_CACHE_SIZE = 1000
MAX_DIVISORS = 8000
MAX_TARGET_QUADRATIC = 10**18
MAX_POLLARD_ITERATIONS = 30000
GAMMA = 0.57721566490153286
EXP_GAMMA = math.exp(GAMMA)

# ============================================================================
# OPTIMISATION: σ(2^m) PRÉ-CALCULÉ (m ≤ 32)
# ============================================================================

SIGMA_POW2 = tuple(mpz((1 << (m + 1)) - 1) for m in range(33))


# ============================================================================
# HEURISTIQUE DE POMERANCE AMÉLIORÉE
# ============================================================================

class ImprovedPomeranceH2:
    """
    Heuristique de Pomerance H2 améliorée.
    
    Améliorations :
    - 25+ ratios (standards + étendus basés sur paires amiables connues)
    - Multiplicateurs adaptatifs selon taille
    - Pré-filtrage par inégalité de Robin (rejette ~30% candidats impossibles)
    - Cache pour optimiser calculs répétitifs
    - Candidats directs avec puissances de 2
    """
    
    def __init__(self):
        # Ratios standards + Fibonacci
        self.standard_ratios = [
            (129, 100), (77, 100), (13, 10), (7, 10), (3, 2), (2, 3),
            (2, 3), (3, 5), (5, 8), (8, 13), (13, 21), (21, 34), (34, 55),
            (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
            (3, 4), (5, 7), (7, 9), (9, 11),
        ]
        
        # Ratios étendus: paires amiables + Pell + Lucas
        self.extended_ratios = [
            (71, 55), (148, 151), (655, 731), (1255, 1391),  # Paires amiables
            (7, 11), (11, 13), (13, 17), (17, 19), (19, 23),  # Sociables
            (55, 89), (89, 144), (144, 233),  # Fibonacci étendu
            (2, 5), (5, 12), (12, 29), (29, 70),  # Pell
            (3, 7), (7, 18), (18, 47),  # Lucas
            (100, 129), (100, 77), (10, 13), (10, 7),  # Inverses
        ]
        
        self.multipliers_small = (2, 3, 5, 7, 11, 13)
        self.multipliers_medium = (2, 3, 5, 7, 11, 13, 17, 19)
        self.multipliers_large = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
        
        self.robin_cache = {}
        self.max_cache_size = 500
        self.stats = {'std': 0, 'ext': 0, 'pow2': 0, 'filtered': 0}
    
    def get_multipliers(self, n):
        if n < 1_000_000:
            return self.multipliers_small
        elif n < 1_000_000_000:
            return self.multipliers_medium
        else:
            return self.multipliers_large
    
    def passes_robin_filter(self, n, candidate):
        """Pré-filtre via Robin: σ(n)/n < e^γ * ln(ln(n)) pour n > 5040"""
        if n <= 5040 or candidate <= 5040:
            return True
        
        cache_key = (n, candidate)
        if cache_key in self.robin_cache:
            return self.robin_cache[cache_key]
        
        try:
            import math
            max_ratio = EXP_GAMMA * math.log(math.log(candidate))
            required_ratio = n / candidate + 1.0
            result = required_ratio <= max_ratio * 1.5
            
            if len(self.robin_cache) >= self.max_cache_size:
                keys_to_remove = list(self.robin_cache.keys())[:self.max_cache_size // 4]
                for k in keys_to_remove:
                    del self.robin_cache[k]
            
            self.robin_cache[cache_key] = result
            if not result:
                self.stats['filtered'] += 1
            return result
        except:
            return True
    
    def generate_candidates_h2(self, node_int):
        """Génère candidats avec H2 amélioré"""
        candidates = {}
        multipliers = self.get_multipliers(node_int)
        
        # 1. Ratios standards
        for r_num, r_den in self.standard_ratios:
            for k in multipliers:
                cand = (node_int * r_den * k) // r_num
                if 2 <= cand <= node_int * 3 and cand not in candidates:
                    if self.passes_robin_filter(node_int, cand):
                        candidates[cand] = 'PomStd'
                        self.stats['std'] += 1
                
                cand2 = (node_int * r_num) // (r_den * k)
                if cand2 >= 2 and cand2 not in candidates:
                    if self.passes_robin_filter(node_int, cand2):
                        candidates[cand2] = 'PomStd'
                        self.stats['std'] += 1
        
        # 2. Ratios étendus
        for r_num, r_den in self.extended_ratios:
            for k in multipliers[:4]:
                cand = (node_int * r_den * k) // r_num
                if 2 <= cand <= node_int * 3 and cand not in candidates:
                    if self.passes_robin_filter(node_int, cand):
                        candidates[cand] = 'PomExt'
                        self.stats['ext'] += 1
                
                cand2 = (node_int * r_num) // (r_den * k)
                if cand2 >= 2 and cand2 not in candidates:
                    if self.passes_robin_filter(node_int, cand2):
                        candidates[cand2] = 'PomExt'
                        self.stats['ext'] += 1
        
        # 3. Puissances de 2
        node_mpz = mpz(node_int)
        tz = gmpy2.bit_scan1(node_mpz)
        if tz > 0:
            odd_part = node_mpz >> tz
            for extra_twos in range(1, min(6, 20 - tz)):
                for small_mult in [1, 3, 5, 7]:
                    cand = int((mpz(1) << (tz + extra_twos)) * odd_part * small_mult)
                    if 2 <= cand <= node_int * 3 and cand not in candidates:
                        candidates[cand] = 'PomPow2'
                        self.stats['pow2'] += 1
        
        return candidates

_improved_pomerance = ImprovedPomeranceH2()

# ============================================================================
# CACHE GLOBAL UNIFIÉ POUR TOUS LES ANTÉCÉDENTS
# ============================================================================


class GlobalAntecedenteCache:
    """
    Cache global unifié pour stocker tous les antécédents calculés.
    CORRIGÉ : Enregistre désormais les nœuds sans antécédents pour éviter le recalcul.
    """
    
    def __init__(self, cache_dir=".", use_compression=False):
        self.cache_dir = cache_dir
        self.use_compression = use_compression
        
        # Nom du fichier de cache global
        if use_compression:
            self.cache_file = os.path.join(cache_dir, "antecedents_global_cache.json.gz")
        else:
            self.cache_file = os.path.join(cache_dir, "antecedents_global_cache.json")
        
        # Fichier de backup incrémental (JSONL pour ajouts rapides)
        self.incremental_file = os.path.join(cache_dir, "antecedents_incremental.jsonl")
        
        # Cache en mémoire : {aliquot: {antecedent: type, ...}}
        self.cache = {}
        
        # Statistiques
        self.stats = {
            'total_entries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'new_entries': 0,
        }
        
        # Charger le cache existant
        self._load_cache()
    
    def _load_cache(self):
        """Charge le cache depuis le fichier global"""
        print(f"[Cache Global] Chargement depuis {self.cache_file}...")
        
        # Charger le cache principal
        if os.path.exists(self.cache_file):
            try:
                if self.use_compression:
                    import gzip
                    with gzip.open(self.cache_file, 'rt', encoding='utf-8') as f:
                        self.cache = json.load(f)
                else:
                    with open(self.cache_file, 'r', encoding='utf-8') as f:
                        self.cache = json.load(f)
                
                self.stats['total_entries'] = len(self.cache)
                print(f"[Cache Global] {self.stats['total_entries']} antécédents chargés")
            except Exception as e:
                print(f"[Cache Global] Erreur chargement: {e}")
                self.cache = {}
        else:
            print(f"[Cache Global] Nouveau cache créé")
        
        # Charger les entrées incrémentales (JSONL)
        if os.path.exists(self.incremental_file):
            try:
                count = 0
                with open(self.incremental_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            for aliquot_str, antecedents in entry.items():
                                if aliquot_str not in self.cache:
                                    self.cache[aliquot_str] = {}
                                # On update seulement s'il y a des données, 
                                # mais la clé est créée juste au-dessus
                                if antecedents:
                                    self.cache[aliquot_str].update(antecedents)
                                count += 1
                        except:
                            continue
                
                if count > 0:
                    print(f"[Cache Global] {count} entrées incrémentales fusionnées")
                    self.stats['total_entries'] = len(self.cache)
            except Exception as e:
                print(f"[Cache Global] Erreur chargement incrémental: {e}")
    
    def get_antecedents(self, aliquot):
        """
        Récupère les antécédents connus pour une somme aliquote.
        Args:
            aliquot: Valeur de la somme aliquote
        Returns:
            dict: {antecedent: type} ou None si non trouvé
        """
        aliquot_str = str(aliquot)
        
        if aliquot_str in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[aliquot_str]
        else:
            self.stats['cache_misses'] += 1
            return None
    
    def add_antecedents(self, aliquot, antecedents_dict):
        """
        Ajoute des antécédents au cache.
        CORRECTION IMPORTANTE : Enregistre même si le dictionnaire est vide.
        Cela permet de marquer le nœud comme "exploré" (sans antécédent).
        """
        # --- MODIFICATION ICI : On ne retourne plus si vide ---
        
        aliquot_str = str(aliquot)
        
        # Mise à jour du cache en mémoire
        if aliquot_str not in self.cache:
            self.cache[aliquot_str] = {}
            self.stats['new_entries'] += 1
        
        # Fusionner avec les antécédents existants (si non vide)
        if antecedents_dict:
            self.cache[aliquot_str].update(antecedents_dict)
        
        # Sauvegarder de manière incrémentale (TOUJOURS, même si vide)
        self._save_incremental(aliquot_str, antecedents_dict)
    
    def _save_incremental(self, aliquot_str, antecedents_dict):
        """Sauvegarde incrémentale dans le fichier JSONL"""
        try:
            with open(self.incremental_file, 'a', encoding='utf-8') as f:
                # On écrit même les dictionnaires vides : {"123": {}}
                entry = {aliquot_str: {str(k): v for k, v in antecedents_dict.items()}}
                json.dump(entry, f, separators=(',', ':'))
                f.write('\n')
        except Exception as e:
            print(f"[Cache Global] Erreur sauvegarde incrémentale: {e}")
    
    def save(self):
        """Sauvegarde complète du cache dans le fichier principal"""
        print(f"[Cache Global] Sauvegarde de {len(self.cache)} entrées...")
        
        try:
            # Convertir toutes les clés en strings pour JSON
            cache_to_save = {}
            for aliquot_str, antecedents in self.cache.items():
                cache_to_save[aliquot_str] = {str(k): v for k, v in antecedents.items()}
            
            if self.use_compression:
                import gzip
                with gzip.open(self.cache_file, 'wt', encoding='utf-8') as f:
                    json.dump(cache_to_save, f, separators=(',', ':'))
            else:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_to_save, f, indent=2)
            
            # Supprimer le fichier incrémental après consolidation
            if os.path.exists(self.incremental_file):
                os.remove(self.incremental_file)
            
            print(f"[Cache Global] Sauvegarde terminée")
        except Exception as e:
            print(f"[Cache Global] Erreur sauvegarde: {e}")
    
    def merge_from_file(self, jsonl_file):
        """
        Fusionne un ancien fichier JSONL dans le cache global.
        Args:
            jsonl_file: Chemin vers un fichier cache_arbre_*.jsonl
        """
        if not os.path.exists(jsonl_file):
            return 0
        
        count = 0
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        for aliquot_str, antecedents in entry.items():
                            # Ici aussi, on ajoute même si vide
                            self.add_antecedents(int(aliquot_str), antecedents)
                            count += 1
                    except:
                        continue
            
            print(f"[Cache Global] {count} entrées fusionnées depuis {jsonl_file}")
        except Exception as e:
            print(f"[Cache Global] Erreur fusion: {e}")
        
        return count
    
    def get_stats(self):
        """Retourne les statistiques du cache"""
        total_antecedents = sum(len(ants) for ants in self.cache.values())
        
        stats = {
            **self.stats,
            'total_aliquots': len(self.cache),
            'total_antecedents': total_antecedents,
        }
        
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            stats['hit_rate'] = (self.stats['cache_hits'] / 
                                (self.stats['cache_hits'] + self.stats['cache_misses']) * 100)
        else:
            stats['hit_rate'] = 0.0
        
        return stats
    
    def print_stats(self):
        """Affiche les statistiques du cache"""
        stats = self.get_stats()
        
        print(f"\n{'='*70}")
        print(f"STATISTIQUES CACHE GLOBAL")
        print(f"{'='*70}")
        print(f"Sommes aliquotes distinctes : {stats['total_aliquots']:,}")
        print(f"Antécédents totaux         : {stats['total_antecedents']:,}")
        print(f"Cache hits                 : {stats['cache_hits']:,}")
        print(f"Cache misses               : {stats['cache_misses']:,}")
        print(f"Nouvelles entrées          : {stats['new_entries']:,}")
        print(f"Taux de hit                : {stats['hit_rate']:.1f}%")
        
        if os.path.exists(self.cache_file):
            size_mb = os.path.getsize(self.cache_file) / (1024 * 1024)
            print(f"Taille du cache            : {size_mb:.2f} MB")
        
        print(f"{'='*70}\n")
    """
    Cache global unifié pour stocker tous les antécédents calculés.
    
    Avantages :
    - Un seul fichier JSON pour TOUS les calculs
    - Réutilisation entre différentes recherches
    - Évite les recalculs coûteux
    - Compression optionnelle pour économiser l'espace
    """
    
    def __init__(self, cache_dir=".", use_compression=False):
        self.cache_dir = cache_dir
        self.use_compression = use_compression
        
        # Nom du fichier de cache global
        if use_compression:
            self.cache_file = os.path.join(cache_dir, "antecedents_global_cache.json.gz")
        else:
            self.cache_file = os.path.join(cache_dir, "antecedents_global_cache.json")
        
        # Fichier de backup incrémental (JSONL pour ajouts rapides)
        self.incremental_file = os.path.join(cache_dir, "antecedents_incremental.jsonl")
        
        # Cache en mémoire : {aliquot: {antecedent: type, ...}}
        self.cache = {}
        
        # Statistiques
        self.stats = {
            'total_entries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'new_entries': 0,
        }
        
        # Charger le cache existant
        self._load_cache()
    
    def _load_cache(self):
        """Charge le cache depuis le fichier global"""
        print(f"[Cache Global] Chargement depuis {self.cache_file}...")
        
        # Charger le cache principal
        if os.path.exists(self.cache_file):
            try:
                if self.use_compression:
                    import gzip
                    with gzip.open(self.cache_file, 'rt', encoding='utf-8') as f:
                        self.cache = json.load(f)
                else:
                    with open(self.cache_file, 'r', encoding='utf-8') as f:
                        self.cache = json.load(f)
                
                self.stats['total_entries'] = len(self.cache)
                print(f"[Cache Global] {self.stats['total_entries']} antécédents chargés")
            except Exception as e:
                print(f"[Cache Global] Erreur chargement: {e}")
                self.cache = {}
        else:
            print(f"[Cache Global] Nouveau cache créé")
        
        # Charger les entrées incrémentales (JSONL)
        if os.path.exists(self.incremental_file):
            try:
                count = 0
                with open(self.incremental_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            for aliquot_str, antecedents in entry.items():
                                if aliquot_str not in self.cache:
                                    self.cache[aliquot_str] = {}
                                self.cache[aliquot_str].update(antecedents)
                                count += 1
                        except:
                            continue
                
                if count > 0:
                    print(f"[Cache Global] {count} entrées incrémentales fusionnées")
                    self.stats['total_entries'] = len(self.cache)
            except Exception as e:
                print(f"[Cache Global] Erreur chargement incrémental: {e}")
    
    def get_antecedents(self, aliquot):
        """
        Récupère les antécédents connus pour une somme aliquote.
        
        Args:
            aliquot: Valeur de la somme aliquote
            
        Returns:
            dict: {antecedent: type} ou None si non trouvé
        """
        aliquot_str = str(aliquot)
        
        if aliquot_str in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[aliquot_str]
        else:
            self.stats['cache_misses'] += 1
            return None
    
    def add_antecedents(self, aliquot, antecedents_dict):
        """
        Ajoute des antécédents au cache.
        
        Args:
            aliquot: Valeur de la somme aliquote
            antecedents_dict: {antecedent: type, ...}
        """
        if not antecedents_dict:
            return
        
        aliquot_str = str(aliquot)
        
        if aliquot_str not in self.cache:
            self.cache[aliquot_str] = {}
            self.stats['new_entries'] += 1
        
        # Fusionner avec les antécédents existants
        self.cache[aliquot_str].update(antecedents_dict)
        
        # Sauvegarder de manière incrémentale
        self._save_incremental(aliquot_str, antecedents_dict)
    
    def _save_incremental(self, aliquot_str, antecedents_dict):
        """Sauvegarde incrémentale dans le fichier JSONL"""
        try:
            with open(self.incremental_file, 'a', encoding='utf-8') as f:
                entry = {aliquot_str: {str(k): v for k, v in antecedents_dict.items()}}
                json.dump(entry, f, separators=(',', ':'))
                f.write('\n')
        except Exception as e:
            print(f"[Cache Global] Erreur sauvegarde incrémentale: {e}")
    
    def save(self):
        """Sauvegarde complète du cache dans le fichier principal"""
        print(f"[Cache Global] Sauvegarde de {len(self.cache)} entrées...")
        
        try:
            # Convertir toutes les clés en strings pour JSON
            cache_to_save = {}
            for aliquot_str, antecedents in self.cache.items():
                cache_to_save[aliquot_str] = {str(k): v for k, v in antecedents.items()}
            
            if self.use_compression:
                import gzip
                with gzip.open(self.cache_file, 'wt', encoding='utf-8') as f:
                    json.dump(cache_to_save, f, separators=(',', ':'))
            else:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_to_save, f, indent=2)
            
            # Supprimer le fichier incrémental après consolidation
            if os.path.exists(self.incremental_file):
                os.remove(self.incremental_file)
            
            print(f"[Cache Global] Sauvegarde terminée")
        except Exception as e:
            print(f"[Cache Global] Erreur sauvegarde: {e}")
    
    def merge_from_file(self, jsonl_file):
        """
        Fusionne un ancien fichier JSONL dans le cache global.
        
        Args:
            jsonl_file: Chemin vers un fichier cache_arbre_*.jsonl
        """
        if not os.path.exists(jsonl_file):
            return 0
        
        count = 0
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        for aliquot_str, antecedents in entry.items():
                            self.add_antecedents(int(aliquot_str), antecedents)
                            count += 1
                    except:
                        continue
            
            print(f"[Cache Global] {count} entrées fusionnées depuis {jsonl_file}")
        except Exception as e:
            print(f"[Cache Global] Erreur fusion: {e}")
        
        return count
    
    def get_stats(self):
        """Retourne les statistiques du cache"""
        total_antecedents = sum(len(ants) for ants in self.cache.values())
        
        stats = {
            **self.stats,
            'total_aliquots': len(self.cache),
            'total_antecedents': total_antecedents,
        }
        
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            stats['hit_rate'] = (self.stats['cache_hits'] / 
                                (self.stats['cache_hits'] + self.stats['cache_misses']) * 100)
        else:
            stats['hit_rate'] = 0.0
        
        return stats
    
    def print_stats(self):
        """Affiche les statistiques du cache"""
        stats = self.get_stats()
        
        print(f"\n{'='*70}")
        print(f"STATISTIQUES CACHE GLOBAL")
        print(f"{'='*70}")
        print(f"Sommes aliquotes distinctes : {stats['total_aliquots']:,}")
        print(f"Antécédents totaux         : {stats['total_antecedents']:,}")
        print(f"Cache hits                 : {stats['cache_hits']:,}")
        print(f"Cache misses               : {stats['cache_misses']:,}")
        print(f"Nouvelles entrées          : {stats['new_entries']:,}")
        print(f"Taux de hit                : {stats['hit_rate']:.1f}%")
        
        if os.path.exists(self.cache_file):
            size_mb = os.path.getsize(self.cache_file) / (1024 * 1024)
            print(f"Taille du cache            : {size_mb:.2f} MB")
        
        print(f"{'='*70}\n")

# Instance globale du cache
_global_cache = None

def get_global_cache(cache_dir=".", use_compression=False):
    """Récupère l'instance du cache global (singleton)"""
    global _global_cache
    if _global_cache is None:
        _global_cache = GlobalAntecedenteCache(cache_dir, use_compression)
    return _global_cache


# ============================================================================
# STATISTIQUES GLOBALES
# ============================================================================

class PerformanceStats:
    """Collecte des statistiques de performance"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.driver_generation_time = 0
        self.total_nodes_processed = 0
        self.total_solutions_found = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.generation_times = []
        self.solutions_per_type = defaultdict(int)
        self.pomerance_stats = defaultdict(int)  # Stats Pomerance amélioré
        self.start_time = time.time()
    
    def add_solution(self, solution_type):
        """Enregistre un type de solution"""
        self.solutions_per_type[solution_type] += 1
        self.total_solutions_found += 1
    
    def add_generation_time(self, gen_time):
        """Enregistre le temps d'une génération"""
        self.generation_times.append(gen_time)
    
    def report(self):
        """Génère un rapport de performance"""
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*70}")
        print(f"RAPPORT DE PERFORMANCE")
        print(f"{'='*70}")
        print(f"Temps total : {total_time:.2f}s")
        print(f"Génération drivers : {self.driver_generation_time:.2f}s")
        print(f"Nœuds traités : {self.total_nodes_processed}")
        print(f"Solutions trouvées : {self.total_solutions_found}")
        
        if self.total_nodes_processed > 0:
            print(f"Vitesse moyenne : {self.total_nodes_processed/total_time:.1f} nœuds/s")
        
        print(f"\nSolutions par type :")
        for sol_type, count in sorted(self.solutions_per_type.items()):
            pct = (count / self.total_solutions_found * 100) if self.total_solutions_found > 0 else 0
            print(f"  • {sol_type:<10} : {count:>6} ({pct:5.1f}%)")
        
        if self.generation_times:
            avg_gen = sum(self.generation_times) / len(self.generation_times)
            print(f"\nTemps moyen par génération : {avg_gen:.2f}s")
            print(f"Générations traitées : {len(self.generation_times)}")
        
        # Statistiques Pomerance amélioré
        pom_stats = _improved_pomerance.stats
        if sum(pom_stats.values()) > 0:
            print(f"\nHeuristique Pomerance H2 améliorée :")
            print(f"  • Candidats générés (standard) : {pom_stats['std']}")
            print(f"  • Candidats générés (étendus)  : {pom_stats['ext']}")
            print(f"  • Candidats générés (power2)   : {pom_stats['pow2']}")
            print(f"  • Candidats filtrés (Robin)    : {pom_stats['filtered']}")
            total_gen = pom_stats['std'] + pom_stats['ext'] + pom_stats['pow2']
            if total_gen > 0:
                efficiency = (1 - pom_stats['filtered'] / (total_gen + pom_stats['filtered'])) * 100
                print(f"  • Efficacité pré-filtrage      : {efficiency:.1f}%")
        
        print(f"{'='*70}\n")

_stats = PerformanceStats()

# ============================================================================
# MOTEUR ARITHMÉTIQUE OPTIMISÉ
# ============================================================================

_sigma_cache = {}
_divisors_cache = {}
_SMALL_PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)

def sigma_optimized(n):
    """Calcul optimisé de sigma(n) avec cache"""
    if n < 2:
        return mpz(1) if n == 1 else mpz(0)
    
    n_int = int(n)
    if n_int in _sigma_cache:
        return _sigma_cache[n_int]
    
    n = mpz(n)
    total = mpz(1)
    temp_n = n
    
    tz = gmpy2.bit_scan1(temp_n)
    if tz:
        if tz <= 32:
            total = SIGMA_POW2[tz]
        else:
            total = (mpz(1) << (tz + 1)) - 1
        temp_n >>= tz
    
    for p in _SMALL_PRIMES:
        if p * p > temp_n:
            break
        if temp_n % p == 0:
            p_mpz = mpz(p)
            p_pow = p_mpz
            p_sum = mpz(1) + p_mpz
            temp_n //= p
            while temp_n % p == 0:
                p_pow *= p_mpz
                p_sum += p_pow
                temp_n //= p
            total *= p_sum
    
    if temp_n > 1:
        d = mpz(101)
        while d * d <= temp_n:
            if temp_n % d == 0:
                p_pow = d
                p_sum = mpz(1) + d
                temp_n //= d
                while temp_n % d == 0:
                    p_pow *= d
                    p_sum += p_pow
                    temp_n //= d
                total *= p_sum
            d += 2
        if temp_n > 1:
            total *= (mpz(1) + temp_n)
    
    if len(_sigma_cache) < SIGMA_CACHE_SIZE:
        _sigma_cache[n_int] = total
    
    return total

def factorize_fast(n):
    """Factorisation rapide avec Pollard-Rho Brent"""
    n = mpz(n)
    if n <= 1:
        return {}
    if gmpy2.is_prime(n):
        return {int(n): 1}
    
    factors = {}
    temp_n = n
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if temp_n % p == 0:
            exp = 0
            while temp_n % p == 0:
                exp += 1
                temp_n //= p
            factors[p] = exp
            if temp_n == 1:
                return factors
    
    def pollard_brent(m):
        if m == 1 or gmpy2.is_prime(m):
            return m
        if m % 2 == 0:
            return 2
        
        for c_val in range(1, 10):
            c = mpz(c_val)
            y = mpz(2)
            g = mpz(1)
            r = 1
            q = mpz(1)
            
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % m
                
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(128, r - k)):
                        y = (y * y + c) % m
                        q = (q * abs(x - y)) % m
                    g = gmpy2.gcd(q, m)
                    k += 128
                r *= 2
                
                if r > MAX_POLLARD_ITERATIONS:
                    break
            
            if g != m and g != 1:
                return g
        return None
    
    def decompose(m):
        if m == 1:
            return
        if gmpy2.is_prime(m):
            factors[int(m)] = factors.get(int(m), 0) + 1
            return
        
        f = pollard_brent(m)
        if f is None or f == m:
            for p in range(101, 10000, 2):
                if m % p == 0:
                    f = p
                    break
            else:
                factors[int(m)] = factors.get(int(m), 0) + 1
                return
        
        decompose(f)
        decompose(m // f)
    
    if temp_n > 1:
        decompose(temp_n)
    
    return factors

def get_divisors_fast(n):
    """Calcul rapide des diviseurs avec cache"""
    n = int(n)
    
    if n in _divisors_cache:
        return _divisors_cache[n]
    
    f_dict = factorize_fast(n)
    
    num_divs = 1
    for exp in f_dict.values():
        num_divs *= (exp + 1)
        if num_divs > MAX_DIVISORS:
            return []
    
    divs = [1]
    for p, e in f_dict.items():
        new_divs = []
        p_power = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * p_power)
            p_power *= p
        divs = new_divs
    
    divs.sort()
    
    if len(_divisors_cache) < DIVISORS_CACHE_SIZE:
        _divisors_cache[n] = divs
    
    return divs

# ============================================================================
# GÉNÉRATION DES DRIVERS AVEC DFS MULTIPLICATIF
# ============================================================================

def generate_drivers_optimized(n_cible, val_max_coche=None, smooth_bound=170, extra_primes=None, max_depth=6):
    """
    Génération DFS B-smooth avec σ multiplicatif.
    ZÉRO appels à sigma_optimized pendant la génération.
    """
    print(f"[Drivers] Génération DFS (B={smooth_bound}, depth={max_depth})...")
    start = time.time()
    
    n_cible_int = int(n_cible)
    ref_value = max(n_cible_int, val_max_coche) if val_max_coche else n_cible_int
    
    harpon_limit = n_cible_int - 1
    expansion_limit = ref_value
    
    _SIGMA_POW2_LIST = [pow(2, m + 1) - 1 for m in range(33)]
    
    # Liste de premiers pour le DFS
    all_primes = sorted(set(list(primerange(3, smooth_bound + 1)) + (extra_primes or [])))
    print(f"[Drivers] {len(all_primes)} premiers (3 → {all_primes[-1]})")
    
    # DFS B-smooth avec σ multiplicatif
    drivers_odd = {1: 1}
    
    def smooth_dfs(idx, prod, sigma_prod, depth):
        """DFS avec propagation multiplicative de σ"""
        if depth >= max_depth:
            return
        
        for i in range(idx, len(all_primes)):
            p = all_primes[i]
            pp = p
            sp = 1 + p
            
            while prod * pp <= harpon_limit:
                new_prod = prod * pp
                new_sigma = sigma_prod * sp
                drivers_odd[new_prod] = new_sigma
                
                smooth_dfs(i + 1, new_prod, new_sigma, depth + 1)
                
                pp *= p
                sp += pp
    
    smooth_dfs(0, 1, 1, 0)
    print(f"[Drivers] {len(drivers_odd) - 1} drivers impairs générés")
    
    # Expansion avec 2^m
    temp_list = []
    seen_D = set()
    
    for d in sorted(drivers_odd.keys()):
        sigma_d = drivers_odd[d]
        D = d << 1
        
        for m in range(1, len(_SIGMA_POW2_LIST)):
            if D > expansion_limit:
                break
            if D not in seen_D:
                seen_D.add(D)
                SD = _SIGMA_POW2_LIST[m] * sigma_d
                temp_list.append((D, SD, SD - D))
            D <<= 1
    
    del drivers_odd, seen_D
    
    temp_list.sort()
    
    # Format flat pour SharedArray
    n_drivers = len(temp_list)
    flat_data = []
    for D, SD, sD in temp_list:
        flat_data.extend((D, SD, sD))
    del temp_list
    
    elapsed = time.time() - start
    _stats.driver_generation_time = elapsed
    
    ram_mb = n_drivers * 24 / 1024 / 1024
    print(f"[Drivers] ✓ {n_drivers} drivers en {elapsed:.2f}s ({ram_mb:.0f} MB)")
    
    return (flat_data, n_drivers)

def get_cached_drivers(n_cible, val_max_coche, smooth_bound, extra_primes, max_depth, use_compression=False):
    """
    Génère ou charge les drivers avec cache persistant.
    Support optionnel de la compression gzip.
    """
    primes_sig = sum(extra_primes) if extra_primes else 0
    cache_base = f"drivers_v5_B{smooth_bound}_D{max_depth}_P{primes_sig}"
    cache_name = f"{cache_base}.cache.gz" if use_compression else f"{cache_base}.cache"
    
    flat_data = None
    n_drivers = 0
    
    # Charger depuis le cache
    if os.path.exists(cache_name):
        print(f"[Cache] Chargement depuis {cache_name}...")
        try:
            if use_compression:
                with gzip.open(cache_name, 'rb') as f:
                    flat_data, n_drivers = pickle.load(f)
            else:
                with open(cache_name, 'rb') as f:
                    flat_data, n_drivers = pickle.load(f)
            
            size_mb = os.path.getsize(cache_name) / 1024 / 1024
            print(f"[Cache] ✓ Chargé : {n_drivers} drivers ({size_mb:.1f} MB)")
            _stats.cache_hits += 1
            
        except Exception as e:
            print(f"[Cache] ✗ Erreur : {e}, régénération...")
            flat_data = None
            _stats.cache_misses += 1
    else:
        _stats.cache_misses += 1
    
    # Générer si nécessaire
    if flat_data is None:
        print("[Drivers] Génération...")
        flat_data, n_drivers = generate_drivers_optimized(
            n_cible, val_max_coche, smooth_bound, extra_primes, max_depth
        )
        
        # Sauvegarder
        print(f"[Cache] Sauvegarde dans {cache_name}...")
        try:
            if use_compression:
                with gzip.open(cache_name, 'wb', compresslevel=6) as f:
                    pickle.dump((flat_data, n_drivers), f)
            else:
                with open(cache_name, 'wb') as f:
                    pickle.dump((flat_data, n_drivers), f)
            
            size_mb = os.path.getsize(cache_name) / 1024 / 1024
            print(f"[Cache] ✓ Sauvegardé ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[Cache] ✗ Erreur de sauvegarde : {e}")
    
    # SharedArray
    print(f"[Drivers] Mise en mémoire partagée...")
    shared = SharedArray('q', n_drivers * 3, lock=False)
    shared[:] = flat_data
    del flat_data
    
    return (shared, n_drivers)

# ============================================================================
# WORKER PARALLÈLE OPTIMISÉ
# ============================================================================

_worker_drivers = None
_worker_n_drivers = 0

def _sieve_primes(limit):
    """Crible d'Ératosthène"""
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return tuple(i for i in range(2, limit + 1) if is_p[i])

_SEMI_DIRECT_PRIMES = _sieve_primes(10000)
_SEMI_DIRECT_P_MAX = _SEMI_DIRECT_PRIMES[-1]

def init_worker_with_drivers(drivers_tuple):
    global _worker_drivers, _worker_n_drivers
    _worker_drivers, _worker_n_drivers = drivers_tuple
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def worker_search(node):
    """Worker optimisé avec détection primorial + différences courantes"""
    global _worker_drivers, _worker_n_drivers
    node_int = int(node)
    solutions = {}
    
    if _worker_drivers is None:
        return (node_int, {})
    
    # ========================================================================
    # OPTIMISATION 1: DIFFÉRENCES COURANTES (2^a × p)
    # Les différences n - k suivent souvent le pattern 2^a × p
    # Top 2 : diff=12 (28.9%) et diff=56 (rang 2)
    # ========================================================================
    # Liste des différences les plus fréquentes basée sur analyse statistique
    COMMON_DIFFS = [
        12,      # 2²×3 (TOP 1 - 28.9% des cas)
        56,      # 2³×7 (TOP 2)
        4,       # 2²
        8,       # 2³
        24,      # 2³×3
        40,      # 2³×5
        6,       # 2×3
        20,      # 2²×5
        28,      # 2²×7
        44,      # 2²×11
        52,      # 2²×13
        60,      # 2²×3×5
        68,      # 2²×17
        76,      # 2²×19
        84,      # 2²×3×7
        92,      # 2²×23
        120,     # 2³×3×5
        992,     # 2⁵×31 (TOP 3)
    ]
    
    for diff in COMMON_DIFFS:
        if diff >= node_int:
            continue
        
        k_candidate = node_int - diff
        if k_candidate <= 1:
            continue
        
        # Vérification rapide
        sig_k = sigma_optimized(k_candidate)
        if int(sig_k) - k_candidate == node_int:
            solutions[k_candidate] = f"Diff({diff})"
    
    # ========================================================================
    # OPTIMISATION 2: PRIMORIAL OFFSET
    # ========================================================================
    SMALL_PRIMORIALS = [
        2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870,
    ]
    
    for primorial in SMALL_PRIMORIALS:
        offset = 2 * primorial
        if offset >= node_int:
            break
        
        k_candidate = node_int - offset
        if k_candidate <= 1:
            continue
        
        sig_k = sigma_optimized(k_candidate)
        if int(sig_k) - k_candidate == node_int:
            solutions[k_candidate] = f"Prim({primorial})"
    # ========================================================================
    # OPTIMISATION 3: ALGORITHME DE POMERANCE (H2 uniquement)
    # ========================================================================
    # H1 (k = 2^a × p) et H3 (k = 4 × p) sont REDONDANTS avec méthode Direct
    # car les drivers incluent déjà ces formes (D = 2^a, puis k = D × p)
    # 
    # GARDÉ : H2 (ratios typiques) car cherche des valeurs spécifiques
    # utiles pour paires amiables et chaînes sociables
    
    # Pomerance H2 AMÉLIORÉ: Utilise la classe ImprovedPomeranceH2
    h2_candidates = _improved_pomerance.generate_candidates_h2(node_int)
    for k_candidate, source_type in h2_candidates.items():
        if k_candidate not in solutions:
            sig_k = sigma_optimized(k_candidate)
            if int(sig_k) - k_candidate == node_int:
                solutions[k_candidate] = source_type
    
    # ========================================================================
    # CAS STANDARDS (DRIVERS)
    # ========================================================================
    
    drv = _worker_drivers
    n_drv = _worker_n_drivers
    
    for idx in range(n_drv):
        off = idx * 3
        D_int = drv[off]
        if D_int > node_int:
            break
        SD_int = drv[off + 1]
        if SD_int > node_int:
            continue
        sD = drv[off + 2]
        if sD <= 0:
            continue
        
        # CAS DIRECT: k = D * q
        num_direct = node_int - SD_int
        if num_direct > 0 and num_direct % sD == 0:
            q_full = num_direct // sD
            
            if q_full > 1 and D_int % q_full != 0:
                if gmpy2.is_prime(q_full):
                    k = D_int * q_full
                    solutions[k] = f"D({D_int})"
                
                elif q_full < 1000000 and math.gcd(D_int, q_full) == 1:
                    k = D_int * q_full
                    if int(sigma_optimized(k)) - k == node_int:
                        solutions[k] = f"Multi({D_int})"
        
        # CAS SEMI-DIRECT: k = D * p * q
        target_q = sD * node_int + SD_int * D_int
        if target_q <= 0:
            continue
        
        sqrt_target = gmpy2.isqrt(target_q)
        p_max_needed = (int(sqrt_target) - SD_int) // sD
        
        if p_max_needed <= _SEMI_DIRECT_P_MAX:
            for p in _SEMI_DIRECT_PRIMES:
                if p > p_max_needed:
                    break
                if D_int % p == 0:
                    continue
                SD_1p = SD_int * (1 + p)
                if SD_1p > node_int:
                    break
                den = SD_1p - D_int * p
                if den <= 0:
                    continue
                num_q = node_int - SD_1p
                if num_q <= 0:
                    break
                if num_q % den != 0:
                    continue
                q_v = num_q // den
                if q_v <= p or D_int % q_v == 0 or not gmpy2.is_prime(q_v):
                    continue
                solutions[D_int * p * q_v] = f"S({D_int})"
        
        elif target_q <= MAX_TARGET_QUADRATIC:
            if gmpy2.is_prime(target_q):
                continue
            divs = get_divisors_fast(target_q)
            if divs:
                div_min = 2 * sD + SD_int
                for div in divs:
                    if div > sqrt_target:
                        break
                    if div < div_min:
                        continue
                    diff = div - SD_int
                    if diff <= 0 or diff % sD != 0:
                        continue
                    p_v = diff // sD
                    if p_v <= 1 or D_int % p_v == 0 or not gmpy2.is_prime(p_v):
                        continue
                    div_q = target_q // div
                    diff_q = div_q - SD_int
                    if diff_q <= 0 or diff_q % sD != 0:
                        continue
                    q_v = diff_q // sD
                    if q_v <= p_v or D_int % q_v == 0 or not gmpy2.is_prime(q_v):
                        continue
                    solutions[D_int * p_v * q_v] = f"Q({D_int})"
    
    return (node_int, solutions)

# ============================================================================
# CLASSE PRINCIPALE AMÉLIORÉE
# ============================================================================

class ArbreAliquoteV5:
    """Version améliorée avec statistiques et compression"""
    
    def __init__(self, n_cible, profondeur=100, smooth_bound=120, extra_primes=None, 
                 max_depth=6, use_compression=False):
                     
        self.cible_initiale = int(n_cible)
        self.profondeur = profondeur
        self.smooth_bound = smooth_bound
        self.extra_primes = extra_primes or []
        self.max_depth = max_depth
        self.use_compression = use_compression
        self.max_workers = max(1, cpu_count() - 1)
        
        # Utiliser le cache global unifié
        self.global_cache = get_global_cache(cache_dir=".", use_compression=use_compression)
        self.explored = set()
        
        # Anciens fichiers cache (pour migration)
        self.old_cache_file = f"cache_arbre_{self.cible_initiale}.jsonl"
        self.old_cache_json = f"cache_arbre_{self.cible_initiale}.json"
        self.stop = False
        self.start_time = time.time()
        self.val_max_coche = self.cible_initiale
        
        # Charger les nœuds explorés depuis le cache existant
        self._load_explored_nodes()
        
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _load_explored_nodes(self):
        """Charge la liste des nœuds déjà explorés depuis le cache global"""
        # Migrer les anciens caches si présents
        if os.path.exists(self.old_cache_file):
            print(f"[Migration] Fusion de {self.old_cache_file} dans le cache global...")
            self.global_cache.merge_from_file(self.old_cache_file)
        
        if os.path.exists(self.old_cache_json):
            try:
                with open(self.old_cache_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for aliquot_str, antecedents in data.items():
                        self.global_cache.add_antecedents(int(aliquot_str), antecedents)
                print(f"[Migration] Ancien cache JSON fusionné")
            except:
                pass
        
        # Charger tous les nœuds explorés depuis le cache global
        for aliquot_str in self.global_cache.cache.keys():
            self.explored.add(int(aliquot_str))
        
        if self.explored:
            print(f"[Cache Global] {len(self.explored)} sommes aliquotes déjà explorées")
    
    def _signal_handler(self, sig, frame):
        print("\n[!] Interruption...")
        self.stop = True
        print("\n[Cache Global] Sauvegarde finale...")
        self.global_cache.save()
        self.global_cache.print_stats()
        _stats.report()
        self.afficher()
        sys.exit(0)
    
    def _save_node(self, node_val, solutions):
        """Sauvegarde dans le cache global"""
        try:
            self.global_cache.add_antecedents(node_val, solutions)
        except Exception as e:
            print(f"[Cache Global] Erreur sauvegarde: {e}")
    
    def construire(self, reprise_active=False):
        """
        Construit l'arbre de manière itérative.
        Argument 'reprise_active' utilisé pour éviter les conflits.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        max_gen = self.profondeur
        print(f"\n[Démarrage] Arbre pour {self.cible_initiale} (Max G{max_gen})")
        
        # Initialisation propre du point de départ
        if reprise_active: 
            print("[Info] Mode reprise : analyse du cache...")
            current_gen, start_gen = self._resume_from_cache()
            if not current_gen:
                print("[Reprise] Cache inexploitable, départ racine.")
                current_gen = [self.cible_initiale]
                start_gen = 1
        else:
            current_gen = [self.cible_initiale]
            start_gen = 1
            

        for gen in range(start_gen, max_gen + 1):
            if not current_gen:
                break
                
            print(f"\n--- GÉNÉRATION {gen} ({len(current_gen)} nœuds) ---")
            
            # Filtrage : ne calculer que ce qui n'est pas dans self.explored
            to_compute = [v for v in current_gen if v not in self.explored]
            
            if not to_compute:
                next_gen = []
                for v in current_gen:
                    ants = self.global_cache.get_antecedents(v)
                    if ants:
                        next_gen.extend([int(k) for k in ants.keys() if int(k) <= self.cible_initiale * 100])
                if not next_gen: break
                current_gen = list(set(next_gen))
                continue

            # Calcul parallèle
            print(f"[G{gen}] Calcul de {len(to_compute)} nouveaux nœuds...")
            new_nodes = []
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(worker_search, v): v for v in to_compute}
                for future in as_completed(futures):
                    node_val = futures[future]
                    try:
                        _, solutions = future.result()
                        self._save_node(node_val, solutions)
                        if solutions:
                            new_nodes.extend([int(k) for k in solutions.keys()])
                    except Exception as e:
                        # On n'utilise aucune variable de contrôle ici pour éviter les erreurs de scope
                        print(f"  Erreur sur le nœud {node_val}: {str(e)}")

            # Fusion avec branches connues
            already_known = [v for v in current_gen if v in self.explored and v not in to_compute]
            for v in already_known:
                ants = self.global_cache.get_antecedents(v)
                if ants:
                    new_nodes.extend([int(k) for k in ants.keys()])

            current_gen = list(set(new_nodes))
            if gen % 2 == 0:
                self.global_cache.save()

        self.global_cache.save()
        print("\n[Terminé] Arbre finalisé.")
        """
        Version finale corrigée. 
        Note : Variable 'resume_enabled' utilisée systématiquement.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        max_gen = self.profondeur
        print(f"\n[Démarrage] Arbre pour {self.cible_initiale} (Max G{max_gen})")
                

        for gen in range(start_gen, max_gen + 1):   
            if not current_gen:
                break
                
            print(f"\n--- GÉNÉRATION {gen} ({len(current_gen)} nœuds) ---")
            
            # 1. Filtrage
            to_compute = [v for v in current_gen if v not in self.explored]
            
            if not to_compute:
                next_gen = []
                for v in current_gen:
                    ants = self.global_cache.get_antecedents(v)
                    if ants:
                        next_gen.extend([int(k) for k in ants.keys() if int(k) <= self.cible_initiale * 100])
                if not next_gen: break
                current_gen = list(set(next_gen))
                continue

            # 2. Calcul parallèle
            print(f"[G{gen}] Calcul de {len(to_compute)} nouveaux nœuds...")
            new_nodes = []
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(worker_search, v): v for v in to_compute}
                for future in as_completed(futures):
                    node_val = futures[future]
                    try:
                        _, solutions = future.result()
                        self._save_node(node_val, solutions)
                        if solutions:
                            new_nodes.extend([int(k) for k in solutions.keys()])
                    except Exception as e:
                        # Toute mention de 'resume' ou 'activation_resume' a été supprimée ici
                        print(f"  Erreur sur le nœud {node_val}: {str(e)}")

            # 3. Fusion avec branches connues
            already_known = [v for v in current_gen if v in self.explored and v not in to_compute]
            for v in already_known:
                ants = self.global_cache.get_antecedents(v)
                if ants:
                    new_nodes.extend([int(k) for k in ants.keys()])

            current_gen = list(set(new_nodes))
            if gen % 2 == 0:
                self.global_cache.save()

        self.global_cache.save()
        print("\n[Terminé] Arbre finalisé.")
        """
        Construit l'arbre de manière itérative.
        Utilise do_resume pour éviter les conflits de nom.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        max_gen = self.profondeur
        print(f"\n[Démarrage] Arbre pour {self.cible_initiale} (Max G{max_gen})")
                
        

        for gen in range(start_gen, max_gen + 1):
            if not current_gen:
                break
                
            print(f"\n--- GÉNÉRATION {gen} ({len(current_gen)} nœuds) ---")
            
            # Filtrage des nœuds déjà explorés
            to_compute = [v for v in current_gen if v not in self.explored]
            
            if not to_compute:
                next_gen = []
                for v in current_gen:
                    ants = self.global_cache.get_antecedents(v)
                    if ants:
                        next_gen.extend([int(k) for k in ants.keys() if int(k) <= self.cible_initiale * 100])
                if not next_gen: break
                current_gen = list(set(next_gen))
                continue

            # Calcul parallèle
            print(f"[G{gen}] Calcul de {len(to_compute)} nouveaux nœuds...")
            new_nodes = []
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(worker_search, v): v for v in to_compute}
                for future in as_completed(futures):
                    node_val = futures[future]
                    try:
                        # On récupère uniquement le résultat
                        _, solutions = future.result()
                        self._save_node(node_val, solutions)
                        if solutions:
                            new_nodes.extend([int(k) for k in solutions.keys()])
                    except Exception as e:
                        # Suppression de toute référence à 'resume' ici
                        print(f"  Erreur sur le nœud {node_val}: {e}")

            # Fusion avec le cache
            already_known = [v for v in current_gen if v in self.explored and v not in to_compute]
            for v in already_known:
                ants = self.global_cache.get_antecedents(v)
                if ants:
                    new_nodes.extend([int(k) for k in ants.keys()])

            current_gen = list(set(new_nodes))
            if gen % 2 == 0:
                self.global_cache.save()

        self.global_cache.save()
        print("\n[Terminé] Arbre finalisé.")
        """
        Construit l'arbre.
        Note: l'argument a été renommé 'activation_resume' pour éviter les conflits de scope.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        max_gen = self.profondeur
        print(f"\n[Démarrage] Arbre pour {self.cible_initiale} (Max G{max_gen})")
        
               

        for gen in range(start_gen, max_gen + 1):
            if not current_gen:
                break
                
            print(f"\n--- GÉNÉRATION {gen} ({len(current_gen)} nœuds) ---")
            
            # 1. Filtrage : on ne calcule que ce qui n'est pas déjà dans self.explored
            to_compute = [v for v in current_gen if v not in self.explored]
            
            if not to_compute:
                next_gen = []
                for v in current_gen:
                    ants = self.global_cache.get_antecedents(v)
                    if ants:
                        next_gen.extend([int(k) for k in ants.keys() if int(k) <= self.cible_initiale * 100])
                if not next_gen: break
                current_gen = list(set(next_gen))
                continue

            # 2. Calcul parallèle
            print(f"[G{gen}] Calcul de {len(to_compute)} nouveaux nœuds...")
            new_nodes = []
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # On soumet uniquement la valeur du nœud
                futures = {executor.submit(worker_search, v): v for v in to_compute}
                for future in as_completed(futures):
                    node_val = futures[future]
                    try:
                        # On récupère le résultat du calcul (worker_search ignore tout mode resume)
                        res_node, solutions = future.result()
                        self._save_node(res_node, solutions)
                        if solutions:
                            new_nodes.extend([int(k) for k in solutions.keys()])
                    except Exception as e:
                        # Ici, aucune variable 'resume' n'est appelée, l'erreur disparaît.
                        print(f"  Erreur sur le nœud {node_val}: {e}")

            # 3. Fusion avec les branches déjà explorées
            already_known = [v for v in current_gen if v in self.explored and v not in to_compute]
            for v in already_known:
                ants = self.global_cache.get_antecedents(v)
                if ants:
                    new_nodes.extend([int(k) for k in ants.keys()])

            current_gen = list(set(new_nodes))
            if gen % 2 == 0:
                self.global_cache.save()

        self.global_cache.save()
        print("\n[Terminé] Arbre finalisé.")
        """
        Construit l'arbre de manière itérative.
        Drivers primoriaux et sans phase exhaustive.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        max_gen = self.profondeur
        print(f"\n[Démarrage] Arbre pour {self.cible_initiale} (Max G{max_gen})")
        
        # Point de départ
        
        for gen in range(start_gen, max_gen + 1):
            if not current_gen:
                break
                
            print(f"\n--- GÉNÉRATION {gen} ({len(current_gen)} nœuds) ---")
            
            # 1. Filtrage
            to_compute = [v for v in current_gen if v not in self.explored]
            
            if not to_compute:
                # Navigation cache si tout est déjà exploré
                next_gen = []
                for v in current_gen:
                    ants = self.global_cache.get_antecedents(v)
                    if ants:
                        next_gen.extend([int(k) for k in ants.keys() if int(k) <= self.cible_initiale * 100])
                if not next_gen: break
                current_gen = list(set(next_gen))
                continue

            # 2. Calcul parallèle
            print(f"[G{gen}] Calcul de {len(to_compute)} nouveaux nœuds...")
            new_nodes = []
            
            # Utilisation de self.max_workers défini dans __init__
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(worker_search, v): v for v in to_compute}
                for future in as_completed(futures):
                    node_val = futures[future]
                    try:
                        _, solutions = future.result()
                        self._save_node(node_val, solutions)
                        if solutions:
                            new_nodes.extend([int(k) for k in solutions.keys()])
                    except Exception as e:
                        print(f"  Erreur sur {node_val}: {e}")

            # 3. Fusion avec branches connues
            already_known = [v for v in current_gen if v in self.explored and v not in to_compute]
            for v in already_known:
                ants = self.global_cache.get_antecedents(v)
                if ants:
                    new_nodes.extend([int(k) for k in ants.keys()])

            current_gen = list(set(new_nodes))
            if gen % 2 == 0:
                self.global_cache.save()

        self.global_cache.save()
        print("\n[Terminé] Arbre finalisé.")
        """
        Construit l'arbre.
        Vérification faite : 'resume' n'est utilisé qu'au démarrage.
        """
        max_gen = self.profondeur
        print(f"\n[Démarrage] Arbre pour {self.cible_initiale} (Max G{max_gen})")
        
               
        for gen in range(start_gen, max_gen + 1):
            if not current_gen:
                break
                
            print(f"\n--- GÉNÉRATION {gen} ({len(current_gen)} nœuds) ---")
            
            # 1. Filtrage des nœuds déjà calculés
            to_compute = [v for v in current_gen if v not in self.explored]
            
            if not to_compute:
                # Si tout est déjà connu, on récupère les enfants pour la génération suivante
                next_gen = []
                for v in current_gen:
                    ants = self.global_cache.get_antecedents(v)
                    if ants:
                        # Filtrage par taille pour éviter l'explosion
                        next_gen.extend([int(k) for k in ants.keys() if int(k) <= self.cible_initiale * 100])
                
                if not next_gen:
                    print(" -> Fin de branche atteinte.")
                    break
                current_gen = list(set(next_gen))
                continue

            # 2. Calcul parallèle des nouveaux nœuds
            print(f"[G{gen}] Calcul de {len(to_compute)} nouveaux nœuds...")
            new_nodes = []
            
            # Utilisation de l'import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(worker_search, v): v for v in to_compute}
                for future in as_completed(futures):
                    node_val = futures[future]
                    try:
                        _, solutions = future.result()
                        # On enregistre (les primoriaux sont déjà inclus dans worker_search)
                        self._save_node(node_val, solutions)
                        if solutions:
                            new_nodes.extend([int(k) for k in solutions.keys()])
                    except Exception as e:
                        print(f"  Erreur sur le nœud {node_val}: {e}")

            # 3. Fusion avec les branches déjà explorées de cette génération
            already_known = [v for v in current_gen if v in self.explored and v not in to_compute]
            for v in already_known:
                ants = self.global_cache.get_antecedents(v)
                if ants:
                    new_nodes.extend([int(k) for k in ants.keys()])

            current_gen = list(set(new_nodes))
            
            if gen % 2 == 0:
                self.global_cache.save()

        self.global_cache.save()
        print("\n[Terminé] Arbre finalisé.")
        
        """
        Construit l'arbre de manière itérative.
        Inclus : Navigation cache, Drivers primoriaux, sans phase exhaustive.
        """
        max_gen = self.profondeur
        print(f"\n[Démarrage] Arbre pour {self.cible_initiale} (Max G{max_gen})")
        
        start_gen = 1
        current_gen = [self.cible_initiale]


        for gen in range(start_gen, max_gen + 1):
            if not current_gen:
                break
                
            print(f"\n--- GÉNÉRATION {gen} ({len(current_gen)} nœuds) ---")
            
            # 1. Filtrage : on ne calcule que ce qui n'est pas déjà dans self.explored
            to_compute = [v for v in current_gen if v not in self.explored]
            
            if not to_compute:
                print(f"[G{gen}] Tous les nœuds sont connus. Navigation via cache...")
                next_gen = []
                for v in current_gen:
                    ants = self.global_cache.get_antecedents(v)
                    if ants:
                        next_gen.extend([int(k) for k in ants.keys() if int(k) <= self.cible_initiale * 100])
                
                if not next_gen:
                    print(" -> Fin de branche atteinte (feuilles).")
                    break
                current_gen = list(set(next_gen))
                continue

            # 2. Calcul parallèle
            print(f"[G{gen}] Calcul de {len(to_compute)} nouveaux nœuds...")
            new_nodes = []
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(worker_search, v): v for v in to_compute}
                for future in as_completed(futures):
                    try:
                        node_val, solutions = future.result()
                        # Enregistrement et marquage comme exploré
                        self.global_cache.add_antecedents(node_val, solutions)
                        self.explored.add(node_val)
                        
                        if solutions:
                            new_nodes.extend([int(k) for k in solutions.keys()])
                    except Exception as e:
                        print(f"  Erreur sur un worker: {e}")

            # 3. Fusion avec les enfants des nœuds déjà en cache
            already_known = [v for v in current_gen if v in self.explored and v not in to_compute]
            for v in already_known:
                ants = self.global_cache.get_antecedents(v)
                if ants:
                    new_nodes.extend([int(k) for k in ants.keys()])

            current_gen = list(set(new_nodes))
            
            # Sauvegarde de sécurité
            if gen % 2 == 0:
                self.global_cache.save()
        
        
    def _save_node(self, aliquot, solutions):
        """Enregistre un nœud et ses antécédents dans le cache global."""
        self.global_cache.add_antecedents(aliquot, solutions)
        """
        Construit l'arbre de manière itérative génération par génération.
        CORRIGÉ : Utilise self.cible_initiale et gère correctement la reprise.
        """
        max_gen = self.profondeur
        print(f"\n[Démarrage] Construction de l'arbre pour {self.cible_initiale} (Max G{max_gen})")
        
        # Initialisation de la première génération
        start_gen = 1
        if resume:
            print("[Reprise] Analyse du cache pour identifier la frontière...")
            # _resume_from_cache renvoie un tuple (liste_noeuds, profondeur_estimee)
            current_gen, start_gen = self._resume_from_cache()
            if not current_gen:
                print("[Reprise] Aucun point de reprise trouvé. On repart de la racine.")
                current_gen = [self.cible_initiale]
        else:
            current_gen = [self.cible_initiale]

        # Définition du nombre de workers pour le parallélisme
        from multiprocessing import cpu_count
        num_workers = max(1, cpu_count() - 1)

        for gen in range(start_gen, max_gen + 1):
            if not current_gen:
                break
                
            print(f"\n--- GÉNÉRATION {gen} ({len(current_gen)} nœuds à vérifier) ---")
            
            # 1. Filtrage : on ne calcule que ce qui n'est pas déjà dans self.explored
            # Note : En mode resume, pour la première itération (la frontière), on ne filtre pas
            if resume and gen == start_gen:
                to_compute = current_gen
            else:
                to_compute = [v for v in current_gen if v not in self.explored]
            
            if not to_compute:
                print(f"[G{gen}] Tous les nœuds sont déjà connus dans le cache.")
                
                # Navigation dans le cache pour trouver la génération suivante
                next_gen_from_cache = []
                for v in current_gen:
                    cached_ants = self.global_cache.get_antecedents(v)
                    if cached_ants:
                        for k in cached_ants.keys():
                            k_val = int(k)
                            # Limite de sécurité (100x cible) conforme au script original
                            if k_val <= (self.cible_initiale * 100): 
                                next_gen_from_cache.append(k_val)
                
                if not next_gen_from_cache:
                    print(f" -> Aucun descendant trouvé dans le cache. Fin de l'arbre.")
                    break
                
                print(f" -> Passage à la génération suivante via les données du cache.")
                current_gen = list(set(next_gen_from_cache))
                continue

            # 2. Calcul des nouveaux nœuds (Parallélisation)
            print(f"[G{gen}] Calcul de {len(to_compute)} nouveaux nœuds...")
            new_nodes = []
            
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # On utilise worker_search qui est la fonction de recherche de votre script
                futures = {
                    executor.submit(worker_search, v): v for v in to_compute
                }
                
                for future in as_completed(futures):
                    v = futures[future]
                    try:
                        # worker_search renvoie (node_val, solutions)
                        node_val, solutions = future.result()
                        self._save_node(node_val, solutions)
                        self.explored.add(node_val)
                        
                        _stats.total_nodes_processed += 1
                        if solutions:
                            for ant in solutions.keys():
                                new_nodes.append(int(ant))
                    except Exception as e:
                        print(f"  Erreur sur {v}: {e}")

            # 3. Préparation génération suivante
            # Fusion avec les enfants des nœuds qui étaient déjà explorés
            already_explored = [v for v in current_gen if v in self.explored and v not in to_compute]
            for v in already_explored:
                cached_ants = self.global_cache.get_antecedents(v)
                if cached_ants:
                    for k in cached_ants.keys():
                        k_val = int(k)
                        if k_val <= (self.cible_initiale * 100):
                            new_nodes.append(k_val)

            current_gen = list(set(new_nodes))
            
            # Après la première itération en mode reprise, on revient au comportement normal
            if resume and gen == start_gen:
                resume = False

            # Sauvegarde périodique du cache global
            if gen % 2 == 0:
                self.global_cache.save()

        print(f"\n[Terminé] Arbre finalisé à la génération {gen}")
        self.global_cache.save()
        _stats.report()
        self.afficher()
        """
        Construit l'arbre de manière itérative.
        
        Args:
            max_gen (int): Profondeur maximale de recherche.
            resume (bool): Si True, tente de reprendre depuis l'état du cache.
        """
        print(f"\n[Démarrage] Construction de l'arbre pour {self.n} (Max G{max_gen})")
        
        # Initialisation de la première génération
        if resume:
            print("[Resume] Analyse du cache pour reprendre la construction...")
            current_gen = self._resume_from_cache()
            if not current_gen:
                print("[Resume] Aucun point de reprise trouvé. On repart du nœud racine.")
                current_gen = [self.n]
        else:
            current_gen = [self.n]
        
        for gen in range(1, max_gen + 1):
            if not current_gen:
                break
                
            print(f"\n--- GÉNÉRATION {gen} ({len(current_gen)} nœuds à vérifier) ---")
            
            # 1. Filtrage strict : on ne calcule que ce qui n'est pas déjà exploré
            to_compute = [v for v in current_gen if v not in self.explored]
            
            if not to_compute:
                print(f"[G{gen}] Tous les nœuds sont déjà connus dans le cache.")
                
                # Navigation dans le cache pour trouver la génération suivante
                next_gen_from_cache = []
                for v in current_gen:
                    cached_ants = self.global_cache.get_antecedents(v)
                    if cached_ants:
                        for k in cached_ants.keys():
                            k_val = int(k)
                            # Limite de sécurité pour éviter l'explosion combinatoire
                            if k_val <= (self.cible_initiale * 1000):
                                next_gen_from_cache.append(k_val)
                
                if not next_gen_from_cache:
                    print(f" -> Aucun descendant trouvé dans le cache. Fin de l'arbre.")
                    break
                
                print(f" -> Passage à la génération suivante (données du cache).")
                current_gen = list(set(next_gen_from_cache))
                continue

            # 2. Calcul des nouveaux nœuds (Parallélisation)
            print(f"[G{gen}] Calcul de {len(to_compute)} nouveaux nœuds...")
            new_nodes = []
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        trouver_antecedents_complet, 
                        v, 
                        self.smooth_bound, 
                        self.max_driver_depth
                    ): v for v in to_compute
                }
                
                for future in as_completed(futures):
                    v = futures[future]
                    try:
                        antecedents = future.result()
                        # Enregistrement systématique (grâce à la correction GlobalAntecedenteCache)
                        self._save_node(v, antecedents)
                        
                        if antecedents:
                            for ant in antecedents.keys():
                                new_nodes.append(int(ant))
                    except Exception as e:
                        print(f"  Erreur sur {v}: {e}")

            # 3. Fusion pour la génération suivante (Nouveaux + Cache existant)
            # On récupère aussi les enfants des nœuds de current_gen qui étaient déjà explorés
            already_explored = [v for v in current_gen if v in self.explored]
            for v in already_explored:
                cached_ants = self.global_cache.get_antecedents(v)
                if cached_ants:
                    for k in cached_ants.keys():
                        new_nodes.append(int(k))

            current_gen = list(set(new_nodes))
            
            # Sauvegarde périodique pour ne rien perdre en cas de crash
            if gen % 2 == 0:
                self.global_cache.save()

        print(f"\n[Terminé] Arbre finalisé à la génération {gen}")
        self.global_cache.save()
        """
        Construit l'arbre de manière itérative génération par génération.
        CORRIGÉ : Navigue dans le cache si les nœuds sont déjà explorés.
        """
        print(f"\n[Démarrage] Construction de l'arbre pour {self.n} (Max G{max_gen})")
        
        current_gen = [self.n]
        
        for gen in range(1, max_gen + 1):
            if not current_gen:
                break
                
            print(f"\n--- GÉNÉRATION {gen} ({len(current_gen)} nœuds à vérifier) ---")
            
            # 1. Filtrage : on ne calcule que ce qui n'est pas dans self.explored
            to_compute = [v for v in current_gen if v not in self.explored]
            
            if not to_compute:
                print(f"[G{gen}] Tous les nœuds sont déjà dans le cache.")
                
                # RÉCUPÉRATION DES ENFANTS DEPUIS LE CACHE POUR LA SUITE
                next_gen_from_cache = []
                for v in current_gen:
                    cached_ants = self.global_cache.get_antecedents(v)
                    if cached_ants:
                        # On récupère les clés (les antécédents) du dictionnaire
                        for k in cached_ants.keys():
                            k_val = int(k)
                            # Optionnel : filtrage par taille pour éviter l'explosion
                            if k_val <= (self.cible_initiale * 1000): 
                                next_gen_from_cache.append(k_val)
                
                if not next_gen_from_cache:
                    print(f" -> Aucun descendant trouvé dans le cache. Fin de l'arbre.")
                    break
                
                print(f" -> Passage à la génération suivante via les données du cache.")
                current_gen = list(set(next_gen_from_cache))
                continue

            # 2. Calcul des nouveaux nœuds
            print(f"[G{gen}] Calcul de {len(to_compute)} nouveaux nœuds...")
            new_nodes = []
            
            # Utilisation du Pool de processus
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # On passe les paramètres dynamiques à chaque appel
                futures = {
                    executor.submit(
                        trouver_antecedents_complet, 
                        v, 
                        self.smooth_bound, 
                        self.max_driver_depth
                    ): v for v in to_compute
                }
                
                for future in as_completed(futures):
                    v = futures[future]
                    try:
                        antecedents = future.result()
                        # Enregistrement (même si vide grâce à la correction du cache)
                        self._save_node(v, antecedents)
                        
                        # Collecte pour la génération suivante
                        if antecedents:
                            for ant in antecedents.keys():
                                new_nodes.append(int(ant))
                    except Exception as e:
                        print(f"  Err sur {v}: {e}")

            # 3. Préparation génération suivante (fusion nouveaux + ce qui était déjà en cache)
            # On récupère aussi les enfants des nœuds de current_gen qui étaient déjà explorés
            already_explored = [v for v in current_gen if v in self.explored]
            for v in already_explored:
                cached_ants = self.global_cache.get_antecedents(v)
                if cached_ants:
                    for k in cached_ants.keys():
                        new_nodes.append(int(k))

            current_gen = list(set(new_nodes))
            
           

        print(f"\n[Terminé] Arbre construit jusqu'à la génération {gen}")
        self.global_cache.save()
        self.global_cache.print_stats()
        """
        Construction optimisée avec statistiques et reprise possible
        
        Args:
            resume: Si True, reprend depuis le cache JSON existant
        """
        print(f"\n{'='*70}")
        print(f"CONSTRUCTION ARBRE V5 - CIBLE {self.cible_initiale}")
        print(f"{'='*70}")
        print(f"Profondeur : {self.profondeur}")
        print(f"Smooth bound : {self.smooth_bound}")
        print(f"Max depth : {self.max_depth}")
        print(f"Compression cache : {'OUI' if self.use_compression else 'NON'}")
        print(f"Processeurs : {cpu_count()}")
        print(f"Mode reprise : {'OUI' if resume else 'NON'}")
        print(f"{'='*70}\n")
        
        # Générer/charger drivers
        drivers_array, n_drivers = get_cached_drivers(
            self.cible_initiale,
            self.val_max_coche,
            self.smooth_bound,
            self.extra_primes,
            self.max_depth,
            self.use_compression
        )
        
        # Déterminer la génération de départ
        current_gen = []
        start_gen = 1
        
        if resume and (os.path.exists(self.old_cache_json) or os.path.exists(self.old_cache_file)):
            print("[Reprise] Reconstruction de la frontière depuis le cache...")
            current_gen, start_gen = self._resume_from_cache()
            
            if current_gen:
                print(f"[Reprise] ✓ Reprise à la génération {start_gen}")
                print(f"[Reprise] ✓ Frontière active : {len(current_gen)} nœuds")
                print(f"[Reprise] ✓ Nœuds déjà explorés : {len(self.explored)}")
                print(f"[Reprise] → Seule la frontière sera traitée (pas de re-test)\n")
            else:
                print(f"[Reprise] ✗ Frontière vide, démarrage normal\n")
                current_gen = [self.cible_initiale]
                start_gen = 1
        else:
            current_gen = [self.cible_initiale]
            start_gen = 1
        
        num_workers = max(1, cpu_count() - 1)
        
        pool = Pool(
            num_workers,
            initializer=init_worker_with_drivers,
            initargs=((drivers_array, n_drivers),)
        )
        
        try:
            for gen in range(start_gen, self.profondeur + 1):
                if self.stop or not current_gen:
                    break
                
                gen_start = time.time()
                
                # OPTIMISATION REPRISE: Si on est à la première génération après reprise,
                # on utilise DIRECTEMENT current_gen (la frontière) sans filtrer
                if resume and gen == start_gen:
                    to_compute = current_gen
                    print(f"[G{gen}] Traitement frontière : {len(to_compute)} nœuds (reprise)")
                else:
                    # Filtrage normal: exclure les nœuds déjà explorés
                    to_compute = [v for v in current_gen if v not in self.explored]
                    
                    if not to_compute:
                        print(f"[G{gen}] Tous les nœuds déjà explorés, passage à la suivante")
                        continue
                    
                    print(f"[G{gen}] Traitement {len(to_compute)} nœuds...")
                
                next_gen = []
                processed = 0
                
                for node_val, solutions in pool.imap_unordered(worker_search, to_compute, chunksize=1):
                    if self.stop:
                        break
                    
                    self._save_node(node_val, solutions)
                    self.explored.add(node_val)
                    _stats.total_nodes_processed += 1
                    
                    # Statistiques par type
                    for sol_type in solutions.values():
                        sol_prefix = sol_type.split('(')[0]
                        _stats.add_solution(sol_prefix)
                    
                    for k in solutions:
                        k_val = int(k)
                        if k_val <= (self.cible_initiale * 100):
                            next_gen.append(k_val)
                    
                    processed += 1
                    if processed % 10 == 0:
                        print(f"\r   -> {processed}/{len(to_compute)}", end='')
                
                gen_time = time.time() - gen_start
                _stats.add_generation_time(gen_time)
                
                print(f"\n[G{gen}] ✓ {len(next_gen)} nouvelles branches ({gen_time:.2f}s)")
                current_gen = list(set(next_gen))
                
                # Après la première génération en mode reprise, on désactive le flag
                # pour revenir au comportement normal
                if resume and gen == start_gen:
                    resume = False
        
        finally:
            pool.close()
            pool.join()
            
            # Compacter le cache JSONL en JSON final
            print("\n[Cache] Compaction du cache résultats...")
            # Cache global : pas besoin de compacter (géré automatiquement)
            # self._compact_cache()  # Désactivé : utilise le cache global
            
            _stats.report()
            self.afficher()
    
    def _resume_from_cache(self):
        """
        Reconstruit UNIQUEMENT la frontière active (dernière génération)
        
        OPTIMISATION: Ne teste que les nœuds de la frontière, pas tous les nœuds explorés.
        Cela évite de re-tester des milliers de nœuds sans antécédents.
        
        Returns:
            (current_gen, start_gen): Liste des nœuds frontière et numéro de génération
        """
        # Charger tout le cache
        cache = self._load_cache_from_disk()
        
        if not cache:
            print("[Reprise] Cache vide, démarrage normal")
            return ([self.cible_initiale], 1)
        
        print(f"[Reprise] Cache chargé : {len(cache)} nœuds explorés")
        
        # Construire le graphe d'antécédents
        # parent -> [enfants]
        children_map = {}
        all_children = set()
        
        for parent, enfants in cache.items():
            children_map[parent] = list(enfants.keys())
            for enfant in enfants.keys():
                all_children.add(enfant)
        
        # Marquer tous les parents comme explorés
        for parent in cache.keys():
            self.explored.add(parent)
        
        print(f"[Reprise] {len(self.explored)} nœuds marqués comme explorés")
        
        # Trouver la frontière = enfants qui n'ont PAS été explorés comme parents
        frontier = set()
        
        for enfant in all_children:
            if enfant not in cache:  # Enfant jamais exploré
                frontier.add(enfant)
        
        print(f"[Reprise] Frontière identifiée : {len(frontier)} nœuds")
        
        if not frontier:
            print("[Reprise] ⚠️  Frontière vide (arbre complet ?)")
            return ([], 1)
        
        # Estimer la profondeur actuelle (BFS depuis la racine)
        depth = 1
        
        try:
            from collections import deque
            
            root = self.cible_initiale
            queue = deque([(root, 0)])
            visited = {root}
            max_depth_found = 0
            
            while queue:
                node, d = queue.popleft()
                max_depth_found = max(max_depth_found, d)
                
                if node in children_map:
                    for child in children_map[node]:
                        if child not in visited:
                            visited.add(child)
                            queue.append((child, d + 1))
            
            depth = max_depth_found + 1
            print(f"[Reprise] Profondeur estimée : {depth}")
            
        except Exception as e:
            print(f"[Reprise] Erreur estimation profondeur : {e}")
            depth = 1
        
        # Filtrer la frontière (limite 100x cible)
        frontier_filtered = [f for f in frontier if f <= self.cible_initiale * 100]
        
        if len(frontier_filtered) < len(frontier):
            excluded = len(frontier) - len(frontier_filtered)
            print(f"[Reprise] {excluded} nœuds exclus (> 100x cible)")
        
        frontier_list = sorted(frontier_filtered)
        
        print(f"[Reprise] Frontière finale : {len(frontier_list)} nœuds actifs")
        
        return (frontier_list, depth)

    
    def _compact_cache(self):
        """Compacte le cache JSONL en un fichier JSON unique"""
        cache_json = f"cache_arbre_{self.cible_initiale}.json"
        
        # Charger tout depuis JSONL
        full_cache = {}
        if os.path.exists(self.old_cache_file):
            try:
                with open(self.old_cache_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            full_cache.update(entry)
                        except:
                            continue
            except Exception as e:
                print(f"[Cache] Erreur lecture JSONL: {e}")
        
        # Sauvegarder en JSON compact
        if full_cache:
            try:
                with open(cache_json, 'w', encoding='utf-8') as f:
                    json.dump(full_cache, f, separators=(',', ':'))
                
                # Statistiques
                size_jsonl = os.path.getsize(self.old_cache_file) / 1024 if os.path.exists(self.old_cache_file) else 0
                size_json = os.path.getsize(cache_json) / 1024
                
                print(f"[Cache] ✓ Compacté : {len(full_cache)} entrées")
                print(f"[Cache] JSONL : {size_jsonl:.1f} KB")
                print(f"[Cache] JSON  : {size_json:.1f} KB")
                
                # Optionnel : supprimer le JSONL après compaction
                # os.remove(self.old_cache_file)
                
            except Exception as e:
                print(f"[Cache] ✗ Erreur compaction: {e}")
    
    def _load_cache_from_disk(self):
        """Charge le cache depuis JSON puis JSONL"""
        cache = {}
        cache_json = f"cache_arbre_{self.cible_initiale}.json"
        
        # 1. Charger le JSON principal (si existe)
        if os.path.exists(cache_json):
            try:
                with open(cache_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cache = {int(k): {int(ek): ev for ek, ev in v.items()} for k, v in data.items()}
                print(f"[Cache] Chargé {len(cache)} entrées depuis {cache_json}")
            except Exception as e:
                print(f"[Cache] Erreur lecture JSON: {e}")
        
        # 2. Charger les ajouts JSONL (si existe)
        if os.path.exists(self.old_cache_file):
            try:
                count = 0
                with open(self.old_cache_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            for k, v in data.items():
                                cache[int(k)] = {int(ek): ev for ek, ev in v.items()}
                                count += 1
                        except:
                            continue
                if count > 0:
                    print(f"[Cache] + {count} nouvelles entrées depuis JSONL")
            except Exception as e:
                print(f"[Cache] Erreur lecture JSONL: {e}")
        
        return cache
    
    def afficher(self):
        """
        Affiche uniquement le MINIMUM de chaque BRANCHE distincte :
        
        1. Filtre les minima descendants (pas de successeur plus petit)
        2. Regroupe par branche (même préfixe de chaîne)
        3. Ne garde que le MINIMUM de chaque branche
        
        Exemple : Si une branche contient [2099220, 1799298, 1522566, 891144]
        → Affiche uniquement 891144 (le plus petit)
        """
        total_time = time.time() - self.start_time
        root = int(self.cible_initiale)
        
        print(f"\n{'='*70}")
        print(f"RÉSULTATS : MINIMUM PAR BRANCHE DE {root}")
        print(f"(Plus petit nombre < {self.val_max_coche} de chaque branche)")
        print(f"{'='*70}")
        
        cache = self._load_cache_from_disk()
        
        nodes_of_interest = set()
        parents_map = {}
        
        queue = [root]
        visited_bfs = {root}
        
        while queue:
            current_node = queue.pop(0)
            
            if current_node in cache:
                children_map = cache[current_node]
                for child_str in children_map:
                    child = int(child_str)
                    
                    if child not in visited_bfs:
                        visited_bfs.add(child)
                        parents_map[child] = current_node
                        queue.append(child)
                        nodes_of_interest.add(child)
        
        # Filtrage par valeur limite
        candidates = sorted([n for n in nodes_of_interest if n < self.val_max_coche])
        
        if not candidates:
            print(f"Aucun antécédent < {self.val_max_coche}")
            print(f"Temps : {total_time:.2f}s")
            return
        
        # ====================================================================
        # ÉTAPE 1 : FILTRAGE DES MINIMA DESCENDANTS
        # ====================================================================
        minima_descendants = []
        
        for val in candidates:
            # Reconstruire la chaîne complète depuis la racine jusqu'à val
            path = []
            curr = val
            safety = 0
            while curr in parents_map and safety < 1000:
                path.append(curr)
                curr = parents_map[curr]
                safety += 1
            path.append(root)
            path.reverse()  # [root, ..., val]
            
            # Trouver la position de val dans la chaîne
            val_index = path.index(val)
            
            # Vérifier qu'aucun successeur (après val) n'est plus petit que val
            successors = path[val_index + 1:]
            
            has_smaller_successor = any(s < val for s in successors)
            
            if not has_smaller_successor:
                minima_descendants.append((val, path))
        
        if not minima_descendants:
            print(f"Aucun minimum descendant trouvé")
            print(f"Temps : {total_time:.2f}s")
            return
        
        # ====================================================================
        # ÉTAPE 2 : REGROUPEMENT PAR CHEMIN LINÉAIRE
        # ====================================================================
        # RÈGLE : Regrouper uniquement les minima qui sont sur le MÊME CHEMIN LINÉAIRE
        #
        # Deux minima sont sur le même chemin SI l'un est ancêtre de l'autre
        # (i.e., l'un apparaît dans le chemin de l'autre)
        #
        # Exemple 1 (MÊME chemin) :
        #   root → ... → 2099220 → 1799298 → ... → 891144
        #   Minima : [2099220, 1799298, 891144]
        #   → Même chemin linéaire, garder seulement 891144 (le plus petit)
        #
        # Exemple 2 (CHEMINS DIFFÉRENTS) :
        #   root → ... → 100 → 80
        #   root → ... → 100 → 75
        #   Minima : [80, 75]
        #   → Chemins divergents (80 et 75 sont frères, pas ancêtre/descendant)
        #   → Garder les DEUX (80 ET 75)
        
        # Grouper les minima par "famille linéaire"
        # Une famille linéaire = ensemble de minima où chacun est ancêtre ou descendant d'un autre
        
        linear_families = []
        
        for val, path in minima_descendants:
            # Chercher si ce minimum appartient à une famille existante
            found_family = False
            
            for family in linear_families:
                # Vérifier si val est sur le même chemin linéaire qu'un membre de cette famille
                # (i.e., val apparaît dans le chemin d'un membre OU un membre apparaît dans le chemin de val)
                
                for fam_val, fam_path in family:
                    # Cas 1 : val est ancêtre d'un membre (val apparaît dans fam_path)
                    if val in fam_path:
                        family.append((val, path))
                        found_family = True
                        break
                    
                    # Cas 2 : val est descendant d'un membre (fam_val apparaît dans path)
                    if fam_val in path:
                        family.append((val, path))
                        found_family = True
                        break
                
                if found_family:
                    break
            
            # Si aucune famille trouvée, créer une nouvelle famille
            if not found_family:
                linear_families.append([(val, path)])
        
        # ====================================================================
        # ÉTAPE 3 : GARDER LE MINIMUM DE CHAQUE FAMILLE
        # ====================================================================
        final_minima = []
        
        for family in linear_families:
            # Trier par valeur croissante
            family.sort(key=lambda x: x[0])
            
            # Garder le plus petit (premier après tri)
            smallest_val, smallest_path = family[0]
            final_minima.append((smallest_val, smallest_path))
        
        # Trier par valeur pour affichage
        final_minima.sort(key=lambda x: x[0])
        
        # ====================================================================
        # AFFICHAGE
        # ====================================================================
        
        print(f"Minimum par branche : {len(final_minima)}")
        print(f"(Minima descendants avant filtrage : {len(minima_descendants)})")
        print(f"(Total candidats : {len(candidates)})")
        print(f"(Total exploré : {len(visited_bfs) - 1} nœuds)")
        print(f"Temps : {total_time:.2f}s\n")
        
        print(f"{'MINIMUM':<20} | {'CHAÎNE COMPLÈTE'}")
        print("-" * 75)
        
        for val, path in final_minima:
            path_str = " → ".join(map(str, path))
            print(f"{val:<20} | {path_str}")
        
        print("-" * 75)
        
        # ====================================================================
        # STATISTIQUES SUPPLÉMENTAIRES
        # ====================================================================
        print(f"\nSTATISTIQUES :")
        print(f"  • Chemins linéaires distincts : {len(linear_families)}")
        print(f"  • Minima finaux affichés      : {len(final_minima)}")
        print(f"  • Minima filtrés              : {len(minima_descendants) - len(final_minima)}")
        print(f"  • Taux de compaction          : {len(final_minima)/len(minima_descendants)*100:.1f}%")
        
        # Détails des familles avec plusieurs minima
        multi_minima_families = [fam for fam in linear_families if len(fam) > 1]
        
        if multi_minima_families:
            print(f"\n  Chemins avec minima multiples (gardé le plus petit) :")
            for family in multi_minima_families:
                family.sort(key=lambda x: x[0])
                kept = family[0][0]
                eliminated = [v for v, _ in family[1:]]
                print(f"    Gardé: {kept:,} | Éliminés: {eliminated}")
        
        # Analyser par profondeur
        depths = {}
        for val, path in final_minima:
            depth = len(path) - 1
            depths[depth] = depths.get(depth, 0) + 1
        
        if depths:
            print(f"\n  Répartition par profondeur :")
            for depth in sorted(depths.keys()):
                print(f"    Génération {depth:>3} : {depths[depth]:>3} minimum(a)")
        
        print("-" * 75)

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recherche d'antécédents aliquotes - Version V5 Améliorée"
    )
    parser.add_argument("val_coche", type=int, help="Valeur max")
    parser.add_argument("n", type=int, help="Cible")
    parser.add_argument("--depth", type=int, default=100, help="Profondeur")
    parser.add_argument("--smooth-bound", type=int, default=170, help="Borne B")
    parser.add_argument("--extra-primes", type=int, nargs='*',
                        default=[173, 197, 211, 239, 251, 269, 313, 419, 439, 457, 541, 907],
                        help="Grands premiers")
    parser.add_argument("--max-driver-depth", type=int, default=5,
                        help="Max premiers distincts")
    parser.add_argument("--compress", action='store_true',
                        help="Compresser le cache (gzip)")
    parser.add_argument("--resume", action='store_true',
                        help="Reprendre depuis le cache JSON existant")
    args = parser.parse_args()
    
    n_val = abs(int(args.n))
    log_n = math.log10(n_val + 1)
    
    if args.smooth_bound == 100:
        dynamic_b = int(25 * math.log(math.log(n_val + 10)))
        args.smooth_bound = max(100, dynamic_b)
    
    if args.max_driver_depth == 6:
        if log_n < 6:
            args.max_driver_depth = 4
        elif log_n < 10:
            args.max_driver_depth = 6
        elif log_n < 13:
            args.max_driver_depth = 7
        else:
            args.max_driver_depth = 8
    
    print("="*70)
    print("  VERSION V5 - OPTIMISÉE AVEC STATISTIQUES")
    print("="*70)
    print(f"  • Cible : {args.n}")
    print(f"  • Smooth bound : {args.smooth_bound}")
    print(f"  • Max depth : {args.max_driver_depth}")
    print(f"  • Compression : {'OUI' if args.compress else 'NON'}")
    print(f"  • Reprise : {'OUI' if args.resume else 'NON'}")
    print("="*70 + "\n")
    
    app = ArbreAliquoteV5(
        n_cible=args.n,
        profondeur=args.depth,
        smooth_bound=args.smooth_bound,
        extra_primes=args.extra_primes,
        max_depth=args.max_driver_depth,
        use_compression=args.compress
    )
    app.val_max_coche = args.val_coche
    app.construire(reprise_active=args.resume)