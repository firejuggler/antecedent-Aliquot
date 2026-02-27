#!/usr/bin/env python3
"""
Recherche d'antécédents aliquotes - Version POMERANCE AMÉLIORÉE + NUMBA
========================================================================
Basé sur V5 Ultra-Optimisée + Algorithme de Pomerance H2 AMÉLIORÉ + Numba JIT

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
- ✅ **NOUVEAU** : Optimisation Numba JIT (5-15x plus rapide pour n < 10^15)
  • σ(n) optimisé avec @njit
  • factorize_fast optimisé avec @njit
  • Pollard-Rho optimisé avec @njit
  • Dispatch intelligent Numba/gmpy2 selon la taille des nombres

Note : Pomerance H1 (k=2^a×p) et H3 (k=4×p) SUPPRIMÉS car redondants
      avec méthode Direct (les drivers incluent déjà ces formes)

Dépendances:
  pip install gmpy2 sympy numba

Usage:
  python3 Arbre_multi_g_optimized.py MIN_ALIQUOT MAX_ALIQUOT [options]
  
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
import hashlib
import os
import math
import pickle
import gzip
from multiprocessing import Pool, cpu_count, Array as SharedArray
from sympy import primerange
from collections import defaultdict, OrderedDict, deque

# ============================================================================
# P2 : ECM IMPORT (sympy Elliptic Curve Method)
# ============================================================================
try:
    from sympy.ntheory import ecm as _sympy_ecm
    ECM_AVAILABLE = True
except ImportError:
    ECM_AVAILABLE = False
    print("[ECM] X sympy.ntheory.ecm non disponible")

# ============================================================================
# P4 : CYTHON IMPORT (boucles compilees en C)
# ============================================================================
try:
    import fast_loops as _cython_loops
    CYTHON_AVAILABLE = True
    print("[Cython] OK Module fast_loops charge (boucles C compilees)")
except ImportError:
    CYTHON_AVAILABLE = False
    # print("[Cython] X Module fast_loops non disponible (mode Python pur)")

# ============================================================================
# P2 : Seuils ECM
# ============================================================================
ECM_FALLBACK_THRESHOLD = 10**15

# ============================================================================
# NUMBA OPTIMIZATION
# ============================================================================
try:
    from numba import njit
    NUMBA_AVAILABLE = True
    # Seuil pour utiliser Numba vs gmpy2
    NUMBA_THRESHOLD = 10**12  # Numba pour n < 10^12, gmpy2 pour n >= 10^12
except ImportError:
    NUMBA_AVAILABLE = False
    NUMBA_THRESHOLD = 0
    print("[NUMBA] X Numba non disponible - Mode gmpy2 uniquement")
    print("        Pour installer: pip install numba")
    # Créer un décorateur factice (gère @njit et @njit(cache=True))
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================

SIGMA_CACHE_SIZE = 10000
DIVISORS_CACHE_SIZE = 5000
MAX_DIVISORS = 8000
MAX_TARGET_QUADRATIC = 10**18
QUADRATIC_MAX_ITERATIONS = 1_000_000  # Budget max d'itérations par driver pour Quadratic
GAMMA = 0.57721566490153286
EXP_GAMMA = math.exp(GAMMA)

# ============================================================================
# OPTIMISATION: σ(2^m) PRÉ-CALCULÉ (m ≤ 32)
# ============================================================================

SIGMA_POW2 = tuple(mpz((1 << (m + 1)) - 1) for m in range(33))

# Liste unique de petits premiers pour trial division (jusqu'à 541)
_SMALL_PRIMES_EXTENDED = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
    331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
    421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541
)


# ============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# ============================================================================

@njit(cache=True)
def gcd_numba(a, b):
    """GCD rapide pour Numba (équivalent à math.gcd)"""
    while b:
        a, b = b, a % b
    return a

@njit(cache=True)
def _mulmod(a, b, m):
    """Multiplication modulaire — rapide si m < 2^31, sinon bit-à-bit."""
    if m < (1 << 31):
        return (a * b) % m
    result = 0
    a = a % m
    b = b % m
    while b > 0:
        if b & 1:
            result = (result + a) % m
        a = (a + a) % m
        b >>= 1
    return result

@njit(cache=True)
def _powmod(base, exp, mod):
    """Exponentiation modulaire optimisée pour Numba."""
    result = 1
    base = base % mod
    while exp > 0:
        if exp & 1:
            result = _mulmod(result, base, mod)
        exp >>= 1
        base = _mulmod(base, base, mod)
    return result

@njit(cache=True)
def is_prime_numba(n):
    """
    Test de primalité Miller-Rabin DÉTERMINISTE optimisé pour Numba.
    Nombre de témoins adaptatif selon la taille de n :
      - n < 2 047            : 1 témoin   (2)
      - n < 1 373 653        : 2 témoins  (2, 3)
      - n < 3 215 031 751    : 4 témoins  (2, 3, 5, 7)
      - n < 3.317×10^24      : 12 témoins (2..37) — déterministe (Sorenson & Webster 2016)
    Gain ~2x pour n < 10^9.
    """
    if n < 2:
        return False
    
    # Petits premiers - test direct
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    if n < 41:  # Tous les premiers < 41 déjà couverts
        return False
    
    # Décomposition n-1 = d × 2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    
    # Nombre de témoins adaptatif selon la taille de n
    if n < 2047:
        num_witnesses = 1
    elif n < 1373653:
        num_witnesses = 2
    elif n < 3215031751:
        num_witnesses = 4
    else:
        num_witnesses = 12
    
    witness_idx = 0
    for a in small_primes:
        if witness_idx >= num_witnesses:
            break
        if a >= n:
            continue
        
        x = _powmod(a, d, n)
        
        if x == 1 or x == n - 1:
            witness_idx += 1
            continue
        
        composite = True
        for _ in range(r - 1):
            x = _mulmod(x, x, n)
            if x == n - 1:
                composite = False
                break
        
        if composite:
            return False
        witness_idx += 1
    
    return True

@njit(cache=True)
def pollard_rho_numba(n, max_iterations=100000):
    """
    Algorithme Pollard-Rho Brent optimisé avec Numba
    10-50x plus rapide que la version gmpy2 pour n < 10^15
    """
    if n == 1:
        return 1
    if n % 2 == 0:
        return 2
    if is_prime_numba(n):
        return n
    
    # Essayer plusieurs valeurs de c pour augmenter les chances de succès
    for c_val in range(1, 50):
        y = 2
        g = 1
        r = 1
        q = 1
        
        iterations = 0
        
        while g == 1 and iterations < max_iterations:
            x = y
            for _ in range(r):
                y = (y * y + c_val) % n
            
            k = 0
            while k < r and g == 1:
                batch_size = min(128, r - k)
                for _ in range(batch_size):
                    y = (y * y + c_val) % n
                    diff = x - y if x > y else y - x
                    q = (q * diff) % n
                    iterations += 1
                    if iterations >= max_iterations:
                        break
                
                g = gcd_numba(q, n)
                k += batch_size
            
            r *= 2
        
        if g != n and g != 1:
            return g
    
    return 0  # Échec

@njit(cache=True)
def sigma_numba(n):
    """
    Calcul de σ(n) optimisé avec Numba + roue mod 30.
    Après les petits premiers, saute les multiples de 2, 3, 5
    (8 candidats sur 30 au lieu de 15 sur 30 avec d += 2).
    """
    if n < 2:
        return 1 if n == 1 else 0
    
    total = 1
    temp_n = n
    
    # Facteurs de 2
    tz = 0
    while temp_n % 2 == 0:
        tz += 1
        temp_n //= 2
    
    if tz > 0:
        # σ(2^tz) = 2^(tz+1) - 1
        total = (1 << (tz + 1)) - 1
    
    # Petits premiers explicites (3..97)
    small_primes = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
    
    for p in small_primes:
        if p * p > temp_n:
            break
        if temp_n % p == 0:
            p_pow = p
            p_sum = 1 + p
            temp_n //= p
            while temp_n % p == 0:
                p_pow *= p
                p_sum += p_pow
                temp_n //= p
            total *= p_sum
    
    # Facteurs restants avec roue mod 30 (8 candidats / 30 nombres)
    # Les résidus copremiers à 30 sont : 1, 7, 11, 13, 17, 19, 23, 29
    wheel_increments = (4, 2, 4, 2, 4, 6, 2, 6)  # Écarts entre résidus successifs
    d = 101  # Premier candidat > 97
    wi = 1   # Index dans wheel_increments (101 = 3×30 + 11, résidu 11 → index 2, mais on commence à 1 après +4)
    while d * d <= temp_n:
        if temp_n % d == 0:
            p_pow = d
            p_sum = 1 + d
            temp_n //= d
            while temp_n % d == 0:
                p_pow *= d
                p_sum += p_pow
                temp_n //= d
            total *= p_sum
        d += wheel_increments[wi]
        wi = (wi + 1) & 7  # Équivalent à (wi + 1) % 8 mais plus rapide
    
    # Si reste premier
    if temp_n > 1:
        total *= (1 + temp_n)
    
    return total


# ============================================================================
# HEURISTIQUE DE POMERANCE AMÉLIORÉE
# ============================================================================

class ImprovedPomeranceH2:
    """Heuristique de Pomerance H2 améliorée - VERSION OPTIMISÉE."""
    
    def __init__(self):
        # ✅ OPTIMISÉ: Ratios réduits
        self.standard_ratios = [
            (129, 100), (77, 100), (13, 10), (7, 10), (3, 2),
            (3, 5), (5, 8), (8, 13), (13, 21),
            (4, 5), (5, 6), (6, 7), (7, 8),
        ]
        self.extended_ratios = [
            (71, 55), (148, 151), (655, 731),
            (7, 11), (11, 13), (13, 17),
            (55, 89), (89, 144),
            (2, 5), (5, 12),
        ]
        self.multipliers_small = (2, 3, 5)
        self.multipliers_medium = (2, 3, 5, 7)
        self.multipliers_large = (2, 3, 5, 7, 11)
        
        # ✅ Cache Robin LRU
        self.robin_cache = OrderedDict()
        self.max_robin_cache = 1000
        
        # ✅ Cache log(log(x))
        self.loglog_cache = OrderedDict()
        self.max_loglog_cache = 500
        
        self.stats = {
            'std': 0, 'ext': 0, 'pow2': 0, 'filtered': 0,
            'total_generated': 0, 'cache_hits': 0
        }
        
        self.max_candidates_per_node = 200
    
    def get_multipliers(self, n):
        if n < 1_000_000:
            return self.multipliers_small
        elif n < 1_000_000_000:
            return self.multipliers_medium
        else:
            return self.multipliers_large
    
    def get_loglog_value(self, candidate):
        cache_key = int(candidate // 1000)
        if cache_key in self.loglog_cache:
            self.stats['cache_hits'] += 1
            self.loglog_cache.move_to_end(cache_key)
            return self.loglog_cache[cache_key]
        try:
            value = math.log(math.log(candidate))
        except Exception:
            return None
        self.loglog_cache[cache_key] = value
        if len(self.loglog_cache) > self.max_loglog_cache:
            self.loglog_cache.popitem(last=False)
        return value
    
    def passes_robin_filter(self, n, candidate):
        if n <= 5040 or candidate <= 5040:
            return True
        cache_key = (n, candidate)
        if cache_key in self.robin_cache:
            self.stats['cache_hits'] += 1
            self.robin_cache.move_to_end(cache_key)
            return self.robin_cache[cache_key]
        loglog_val = self.get_loglog_value(candidate)
        if loglog_val is None:
            return True
        try:
            max_ratio = EXP_GAMMA * loglog_val
            required_ratio = n / candidate + 1.0
            result = required_ratio <= max_ratio * 1.5
        except Exception:
            return True
        self.robin_cache[cache_key] = result
        if len(self.robin_cache) > self.max_robin_cache:
            self.robin_cache.popitem(last=False)
        if not result:
            self.stats['filtered'] += 1
        return result
    
    def generate_candidates_h2(self, node_int):
        candidates = {}
        multipliers = self.get_multipliers(node_int)
        
        def add_candidate(cand, typ):
            if len(candidates) >= self.max_candidates_per_node:
                return False
            if 2 <= cand <= node_int * 3 and cand not in candidates:
                if self.passes_robin_filter(node_int, cand):
                    candidates[cand] = typ
                    self.stats[typ.lower().replace('pom', '')] += 1
            return True
        
        for r_num, r_den in self.standard_ratios:
            for k in multipliers:
                cand = (node_int * r_den * k) // r_num
                if not add_candidate(cand, 'PomStd'):
                    break
                cand2 = (node_int * r_num) // (r_den * k)
                if cand2 >= 2:
                    if not add_candidate(cand2, 'PomStd'):
                        break
            if len(candidates) >= self.max_candidates_per_node:
                break
        
        if len(candidates) < self.max_candidates_per_node:
            for r_num, r_den in self.extended_ratios:
                for k in multipliers[:3]:
                    cand = (node_int * r_den * k) // r_num
                    if not add_candidate(cand, 'PomExt'):
                        break
                    cand2 = (node_int * r_num) // (r_den * k)
                    if cand2 >= 2:
                        if not add_candidate(cand2, 'PomExt'):
                            break
                if len(candidates) >= self.max_candidates_per_node:
                    break
        
        if len(candidates) < self.max_candidates_per_node:
            node_mpz = mpz(node_int)
            tz = gmpy2.bit_scan1(node_mpz)
            if tz > 0:
                odd_part = node_mpz >> tz
                for extra_twos in range(1, min(4, 15 - tz)):
                    for small_mult in [1, 3]:
                        cand = int((mpz(1) << (tz + extra_twos)) * odd_part * small_mult)
                        if not add_candidate(cand, 'PomPow2'):
                            break
                    if len(candidates) >= self.max_candidates_per_node:
                        break
        
        self.stats['total_generated'] += len(candidates)
        return candidates

_improved_pomerance = ImprovedPomeranceH2()

# ============================================================================
# CACHE GLOBAL UNIFIÉ
# ============================================================================

class GlobalAntecedenteCache:
    def __init__(self, cache_dir=".", use_compression=False):
        self.cache_dir = cache_dir
        self.use_compression = use_compression
        if use_compression:
            self.cache_file = os.path.join(cache_dir, "antecedents_global_cache.json.gz")
        else:
            self.cache_file = os.path.join(cache_dir, "antecedents_global_cache.json")
        self.incremental_file = os.path.join(cache_dir, "antecedents_incremental.jsonl")
        self.cache = {}
        self.stats = {
            'total_entries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'new_entries': 0,
        }
        self._load_cache()
    
    def _load_cache(self):
        print(f"[Cache Global] Chargement depuis {self.cache_file}...")
        if os.path.exists(self.cache_file):
            try:
                if self.use_compression:
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
                                if antecedents:
                                    self.cache[aliquot_str].update(antecedents)
                                count += 1
                        except Exception:
                            continue
                if count > 0:
                    print(f"[Cache Global] {count} entrées incrémentales fusionnées")
                    self.stats['total_entries'] = len(self.cache)
            except Exception as e:
                print(f"[Cache Global] Erreur chargement incrémental: {e}")
    
    def get_antecedents(self, aliquot):
        aliquot_str = str(aliquot)
        if aliquot_str in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[aliquot_str]
        else:
            self.stats['cache_misses'] += 1
            return None
    
    def add_antecedents(self, aliquot, antecedents_dict):
        aliquot_str = str(aliquot)
        if aliquot_str not in self.cache:
            self.cache[aliquot_str] = {}
            self.stats['new_entries'] += 1
        # ✅ CORRECTIF CACHE: Toujours mettre à jour, même si dict vide
        # Cela marque le nœud comme "complètement exploré"
        self.cache[aliquot_str].update(antecedents_dict or {})
        self._save_incremental(aliquot_str, antecedents_dict or {})
    
    def _save_incremental(self, aliquot_str, antecedents_dict):
        try:
            with open(self.incremental_file, 'a', encoding='utf-8') as f:
                entry = {aliquot_str: {str(k): v for k, v in antecedents_dict.items()}}
                json.dump(entry, f, separators=(',', ':'))
                f.write('\n')
        except Exception as e:
            print(f"[Cache Global] Erreur sauvegarde incrémentale: {e}")
    
    def save(self):
        print(f"[Cache Global] Sauvegarde de {len(self.cache)} entrées...")
        try:
            cache_to_save = {}
            for aliquot_str, antecedents in self.cache.items():
                cache_to_save[aliquot_str] = {str(k): v for k, v in antecedents.items()}
            if self.use_compression:
                with gzip.open(self.cache_file, 'wt', encoding='utf-8') as f:
                    json.dump(cache_to_save, f, separators=(',', ':'))
            else:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_to_save, f, indent=2)
            if os.path.exists(self.incremental_file):
                os.remove(self.incremental_file)
            print(f"[Cache Global] Sauvegarde terminée")
        except Exception as e:
            print(f"[Cache Global] Erreur sauvegarde: {e}")
    
    def merge_from_file(self, jsonl_file):
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
                    except Exception:
                        continue
            print(f"[Cache Global] {count} entrées fusionnées depuis {jsonl_file}")
        except Exception as e:
            print(f"[Cache Global] Erreur fusion: {e}")
        return count
    
    def get_stats(self):
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

# ============================================================================
# STATISTIQUES GLOBALES
# ============================================================================

class PerformanceStats:
    def __init__(self):
        self.reset()
        self._report_printed = False
    
    def reset(self):
        self.driver_generation_time = 0
        self.total_nodes_processed = 0
        self.total_solutions_found = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.generation_times = []
        self.solutions_per_type = defaultdict(int)
        self.start_time = time.time()
    
    def add_solution(self, solution_type):
        self.solutions_per_type[solution_type] += 1
        self.total_solutions_found += 1
    
    def add_generation_time(self, gen_time):
        self.generation_times.append(gen_time)
    
    def report(self, force=False):
        if self._report_printed and not force:
            return
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
        
        # Statistiques V6 (P1/P2/P3/P4)
        ns = _numba_stats
        total_sigma = ns['sigma_numba'] + ns['sigma_gmpy2'] + ns.get('sigma_from_factors', 0)
        total_factor = ns['factorize_numba'] + ns['factorize_gmpy2']
        
        if total_sigma > 0 or total_factor > 0:
            print(f"\n--- Optimisations V6 TURBO ---")
            if total_sigma > 0:
                pct_fused = (ns.get('sigma_from_factors', 0) / total_sigma * 100) if total_sigma > 0 else 0
                print(f"  P1 sigma via facteurs  : {ns.get('sigma_from_factors', 0):>6} ({pct_fused:5.1f}%)")
                print(f"     sigma Numba direct  : {ns['sigma_numba']:>6}")
                print(f"     sigma gmpy2 direct  : {ns['sigma_gmpy2']:>6}")
            if total_factor > 0:
                print(f"  P2 factorize Numba    : {ns['factorize_numba']:>6}")
                print(f"     factorize gmpy2    : {ns['factorize_gmpy2']:>6}")
                print(f"     factorize ECM used : {ns.get('factorize_ecm', 0):>6}")
                print(f"     ECM appels/succes  : {ns.get('ecm_calls', 0)}/{ns.get('ecm_success', 0)}")
            print(f"  P3 Crible semi-direct : {_SEMI_DIRECT_P_MAX:,} ({len(_SEMI_DIRECT_PRIMES):,} premiers)")
            if CYTHON_AVAILABLE:
                print(f"  P4 Cython+            : ACTIF (semi_direct, divisors, quadratic_scan, sieve, drivers)")
            else:
                print(f"  P4 Cython             : INACTIF (Python pur)")
        
        # Statistiques Filtrage Drivers
        fs = _filter_stats
        if fs['drivers_tested'] > 0:
            total_filtered = fs['filtered_bisect'] + fs['filtered_pmin'] + fs['filtered_qmax'] + fs['filtered_pmax_lt_pmin'] + fs['filtered_tq_prime']
            pct_filtered = (total_filtered / fs['drivers_tested'] * 100) if fs['drivers_tested'] > 0 else 0
            total_quadratic = fs['entered_quadratic'] + fs['factorized_quadratic'] + fs['fallback_quadratic']
            print(f"\nFiltrage pré-drivers :")
            print(f"  • Drivers examinés          : {fs['drivers_tested']:>10,}")
            print(f"  • Éliminés (bisect D>node)  : {fs['filtered_bisect']:>10,}")
            print(f"  • Éliminés (p_min)          : {fs['filtered_pmin']:>10,}")
            print(f"  • Éliminés (q_max ≤ p_min)  : {fs['filtered_qmax']:>10,}")
            print(f"  • Éliminés (isqrt arrondi)  : {fs['filtered_pmax_lt_pmin']:>10,}")
            print(f"  • Éliminés (target_q premier): {fs['filtered_tq_prime']:>10,}")
            print(f"  • TOTAL FILTRÉS             : {total_filtered:>10,} ({pct_filtered:.1f}%)")
            print(f"  • Entrés Semi-direct        : {fs['entered_semi_direct']:>10,}")
            print(f"  • Quadratic (scan ≤300)     : {fs['entered_quadratic']:>10,}")
            print(f"  • Quadratic (factorisé)     : {fs['factorized_quadratic']:>10,}")
            print(f"  • Quadratic (fallback)      : {fs['fallback_quadratic']:>10,}")
            print(f"  • TOTAL Quadratic           : {total_quadratic:>10,}")
        
        print(f"{'='*70}\n")
        self._report_printed = True

_stats = PerformanceStats()

# ============================================================================
# MOTEUR ARITHMÉTIQUE
# ============================================================================

_sigma_cache = OrderedDict()
_divisors_cache = OrderedDict()
_factors_cache = OrderedDict()  # P1: cache de factorisation
_FACTORS_CACHE_SIZE = 10000
# _SMALL_PRIMES supprimé : redondant avec _SMALL_PRIMES_EXTENDED[1:25]
_SMALL_PRIMES = _SMALL_PRIMES_EXTENDED[1:25]  # (3, 5, ..., 97) — vue sur le tuple existant

# Statistiques d'utilisation Numba + P1/P2
_numba_stats = {
    'sigma_numba': 0, 'sigma_gmpy2': 0, 'sigma_from_factors': 0,
    'factorize_numba': 0, 'factorize_gmpy2': 0, 'factorize_ecm': 0,
    'ecm_calls': 0, 'ecm_success': 0,
}


# ============================================================================
# P1 : sigma depuis factorisation — O(nb facteurs) au lieu de O(sqrt(n))
# ============================================================================

def sigma_from_factors(factors_dict):
    total = mpz(1)
    for p, e in factors_dict.items():
        total *= (mpz(p) ** (e + 1) - 1) // (p - 1)
    return total


# ============================================================================
# P2 : ECM fallback pour factorisation
# ============================================================================

def _ecm_factor(n):
    if not ECM_AVAILABLE:
        return None
    try:
        n_int = int(n)
        if n_int < 7:
            return None
        factors_set = _sympy_ecm(n_int)
        _numba_stats['ecm_calls'] += 1
        if factors_set and len(factors_set) > 1:
            _numba_stats['ecm_success'] += 1
            return min(factors_set)
        if factors_set and len(factors_set) == 1:
            f = next(iter(factors_set))
            if f != n_int:
                _numba_stats['ecm_success'] += 1
                return f
        return None
    except Exception:
        return None

# ============================================================================
# FILTRES PRÉ-DRIVERS
# ============================================================================

# Statistiques de filtrage (accumulées par worker, non thread-safe mais informatif)
_filter_stats = {
    'drivers_tested': 0,        # Total drivers examines
    'filtered_bisect': 0,       # Elimines par recherche binaire (D > node)
    'filtered_pmin': 0,         # Elimines par p_min (SD*(1+p_min) > node)
    'filtered_qmax': 0,         # Elimines par q(p_min) <= p_min (sans isqrt)
    'filtered_pmax_lt_pmin': 0, # Elimines car p_max < p_min (arrondi isqrt)
    'filtered_tq_prime': 0,     # Elimines car target_q est premier
    'entered_semi_direct': 0,   # Entres dans la boucle Semi-direct
    'entered_quadratic': 0,     # Entres dans Quadratic (scan lineaire <=300 iters)
    'factorized_quadratic': 0,  # Entres dans Quadratic (approche factorisee)
    'fallback_quadratic': 0,    # Fallback scan lineaire (factorisation echouee)
}

# Petits premiers pour calcul de p_min (3..53 = _SMALL_PRIMES_EXTENDED[1:16])
_P_MIN_CANDIDATES = _SMALL_PRIMES_EXTENDED[1:16]  # (3, 5, 7, ..., 53)

def _smallest_coprime_prime(D):
    """Plus petit premier impair ne divisant pas D (D est toujours pair)."""
    for p in _P_MIN_CANDIDATES:
        if D % p != 0:
            return p
    return 59  # Fallback: D divisible par tous les premiers ≤ 53 (extrêmement rare)

# ============================================================================
# WARMUP NUMBA (compile au 1er lancement, charge du cache ensuite)
# ============================================================================
if NUMBA_AVAILABLE:
    _warmup_start = time.time()
    _ = _mulmod(123456789, 987654321, 1000000007)
    _ = _powmod(2, 100, 1000000007)
    _ = sigma_numba(12345)
    _ = is_prime_numba(104729)
    _ = pollard_rho_numba(1000003 * 1000033)
    _warmup_elapsed = time.time() - _warmup_start
    print(f"[Numba] Warmup JIT terminé en {_warmup_elapsed:.1f}s")

def sigma_optimized(n):
    """
    P1: Dispatch intelligent avec fusion factorize+sigma pour n >= 10^12.
    Pour n < 10^12 : Numba direct.
    Pour n >= 10^12 : factorize puis sigma_from_factors (~200x plus rapide).
    """
    if n < 2:
        return mpz(1) if n == 1 else mpz(0)
    
    n_int = int(n)
    
    # Cache check (LRU)
    if n_int in _sigma_cache:
        _sigma_cache.move_to_end(n_int)
        return _sigma_cache[n_int]
    
    # P1: Pour n >= NUMBA_THRESHOLD, utiliser factorize + sigma_from_factors
    if n_int >= NUMBA_THRESHOLD:
        _numba_stats['sigma_from_factors'] += 1
        factors = factorize_fast(n_int)
        result = sigma_from_factors(factors)
    elif NUMBA_AVAILABLE:
        result = mpz(sigma_numba(n_int))
        _numba_stats['sigma_numba'] += 1
    else:
        # Utiliser gmpy2 pour grands nombres
        _numba_stats['sigma_gmpy2'] += 1
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
            # Wheel mod 30: test 8/30 candidates instead of 15/30 with d += 2
            # Residues coprime to 30: 1, 7, 11, 13, 17, 19, 23, 29
            # Increments starting from residue 7: 4, 2, 4, 2, 4, 6, 2, 6
            _wheel30 = (4, 2, 4, 2, 4, 6, 2, 6)
            d = mpz(101)
            wi = 1  # d=101 (residue 11) -> increment +2 to get to 103 (residue 13)
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
                d += _wheel30[wi]
                wi = (wi + 1) & 7
            if temp_n > 1:
                total *= (mpz(1) + temp_n)
        result = total
    
    # Cache LRU
    _sigma_cache[n_int] = result
    if len(_sigma_cache) > SIGMA_CACHE_SIZE:
        _sigma_cache.popitem(last=False)
    
    return result

def factorize_fast(n):
    """
    P1+P2: Factorisation unifiee avec cache + ECM fallback.
    Trial division -> Pollard-Rho -> ECM (pour composites recalcitrants).
    """
    n_int = int(n)
    
    # P1: Cache de factorisation
    if n_int in _factors_cache:
        _factors_cache.move_to_end(n_int)
        return _factors_cache[n_int].copy()
    
    # Dispatch : Numba vs gmpy2
    if NUMBA_AVAILABLE and n_int < NUMBA_THRESHOLD:
        # Utiliser Numba pour performance maximale
        _numba_stats['factorize_numba'] += 1
        
        if n_int <= 1:
            return {}
        if is_prime_numba(n_int):
            return {n_int: 1}
        
        factors = {}
        
        temp_n = n_int
        for p in _SMALL_PRIMES_EXTENDED:
            if temp_n % p == 0:
                exp = 0
                while temp_n % p == 0:
                    exp += 1
                    temp_n //= p
                factors[p] = exp
                if temp_n == 1:
                    break
        
        if temp_n > 1:
            if is_prime_numba(temp_n):
                factors[temp_n] = 1
            else:
                # Utiliser Pollard-Rho Numba
                factor = pollard_rho_numba(temp_n)
                
                if factor and factor > 1 and factor != temp_n:
                    # Décomposer récursivement
                    sub_factors1 = factorize_fast(factor)
                    sub_factors2 = factorize_fast(temp_n // factor)
                    
                    for p, e in sub_factors1.items():
                        factors[p] = factors.get(p, 0) + e
                    for p, e in sub_factors2.items():
                        factors[p] = factors.get(p, 0) + e
                else:
                    # Échec Pollard-Rho : fallback trial division étendue
                    d = 547
                    while d * d <= temp_n:
                        if temp_n % d == 0:
                            exp = 0
                            while temp_n % d == 0:
                                exp += 1
                                temp_n //= d
                            factors[d] = factors.get(d, 0) + exp
                            if temp_n == 1:
                                break
                            if is_prime_numba(temp_n):
                                factors[temp_n] = factors.get(temp_n, 0) + 1
                                break
                        d += 2
                    else:
                        # Reste (premier ou composite irréductible)
                        if temp_n > 1:
                            factors[temp_n] = factors.get(temp_n, 0) + 1
        
        # P1: cache result (était manquant dans le path Numba)
        _factors_cache[n_int] = factors.copy()
        if len(_factors_cache) > _FACTORS_CACHE_SIZE:
            _factors_cache.popitem(last=False)
        return factors
    
    else:
        # Utiliser gmpy2 pour très grands nombres (>= 10^15)
        _numba_stats['factorize_gmpy2'] += 1
        
        n = mpz(n)
        if n <= 1:
            return {}
        if gmpy2.is_prime(n):
            return {int(n): 1}
        
        factors = {}
        temp_n = n
        
        for p in _SMALL_PRIMES_EXTENDED:
            if temp_n % p == 0:
                exp = 0
                while temp_n % p == 0:
                    exp += 1
                    temp_n //= p
                factors[p] = exp
                if temp_n == 1:
                    return factors
        
        if temp_n == 1:
            return factors
        
        if gmpy2.is_prime(temp_n):
            factors[int(temp_n)] = 1
            return factors
        
        # Pollard-Rho Brent optimisé (gmpy2)
        def pollard_brent_optimized(m):
            if m == 1:
                return 1
            if gmpy2.is_prime(m):
                return m
            if m % 2 == 0:
                return 2
            
            # 50 tentatives avec différentes valeurs de c
            for c_val in range(1, 50):
                c = mpz(c_val)
                y = mpz(2)
                g = mpz(1)
                r = 1
                q = mpz(1)
                
                iterations = 0
                max_iter = 300000
                
                while g == 1:
                    x = y
                    for _ in range(r):
                        y = (y * y + c) % m
                    
                    k = 0
                    while k < r and g == 1:
                        batch_size = min(128, r - k)
                        for _ in range(batch_size):
                            y = (y * y + c) % m
                            q = (q * abs(x - y)) % m
                            iterations += 1
                        g = gmpy2.gcd(q, m)
                        k += batch_size
                    
                    r *= 2
                    
                    if iterations > max_iter:
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
            
            f = pollard_brent_optimized(m)
            
            # P2: ECM fallback si Pollard echoue sur grand composite
            if (f is None or f == m) and int(m) > ECM_FALLBACK_THRESHOLD:
                _numba_stats['factorize_ecm'] += 1
                ecm_f = _ecm_factor(m)
                if ecm_f is not None and ecm_f > 1 and ecm_f != int(m):
                    f = mpz(ecm_f)
            
            if f is None or f == m:
                # Fallback : trial division etendu jusqu'a 10^7
                limit = min(10_000_000, int(gmpy2.isqrt(m)) + 1)
                for p in range(547, limit, 2):
                    if m % p == 0:
                        f = p
                        break
                
                if f is None or f == m or f == 1:
                    factors[int(m)] = factors.get(int(m), 0) + 1
                    return
            
            decompose(f)
            decompose(m // f)
        
        if temp_n > 1:
            decompose(temp_n)
        
        # P1: cache result
        _factors_cache[n_int] = factors.copy()
        if len(_factors_cache) > _FACTORS_CACHE_SIZE:
            _factors_cache.popitem(last=False)
        return factors

def get_divisors_fast(n):
    n = int(n)
    if n in _divisors_cache:
        _divisors_cache.move_to_end(n)
        return _divisors_cache[n]
    f_dict = factorize_fast(n)
    # P4: Cython pour la construction des diviseurs (2-3x plus rapide)
    if CYTHON_AVAILABLE:
        divs = _cython_loops.get_divisors_fast_c(f_dict, MAX_DIVISORS)
    else:
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
    _divisors_cache[n] = divs
    if len(_divisors_cache) > DIVISORS_CACHE_SIZE:
        _divisors_cache.popitem(last=False)
    return divs

# ============================================================================
# P6 : DIVISEURS DANS UN INTERVALLE (pruning DFS)
# ============================================================================

def get_divisors_in_range(n, lo, hi):
    """
    P6: Genere les diviseurs de n dans [lo, hi] uniquement.
    Utilise Cython DFS avec pruning si disponible, sinon DFS Python.
    """
    n_int = int(n)
    f_dict = factorize_fast(n_int)
    num_divs = 1
    for exp in f_dict.values():
        num_divs *= (exp + 1)
        if num_divs > MAX_DIVISORS * 10:
            return []
    if CYTHON_AVAILABLE:
        return _cython_loops.divisors_in_range(f_dict, lo, hi)
    else:
        prime_list = sorted(f_dict.items())
        result = []
        _py_divisors_dfs(prime_list, 0, 1, lo, hi, result)
        result.sort()
        return result


def _py_divisors_dfs(prime_list, idx, current, lo, hi, result):
    """DFS Python avec pruning (fallback si pas de Cython)."""
    if current > hi:
        return
    if idx == len(prime_list):
        if current >= lo:
            result.append(current)
        return
    p, max_e = prime_list[idx]
    pe = 1
    for e in range(max_e + 1):
        if current * pe > hi:
            break
        _py_divisors_dfs(prime_list, idx + 1, current * pe, lo, hi, result)
        pe *= p


# ============================================================================
# GÉNÉRATION DES DRIVERS
# ============================================================================

def _generate_odd_drivers(all_primes, harpon_limit, max_depth):
    """Génère les drivers impairs (partie odd) comme dict {prod: sigma_prod}."""
    # P4: Cython DFS si disponible (2-4x plus rapide)
    if CYTHON_AVAILABLE:
        drivers_odd = _cython_loops.generate_odd_drivers_c(list(all_primes), harpon_limit, max_depth)
        return drivers_odd
    drivers_odd = {1: 1}
    def smooth_dfs(idx, prod, sigma_prod, depth):
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
    return drivers_odd

def _expand_to_even_flat(odd_dict, expansion_limit):
    """Convertit les drivers impairs en drivers pairs (D, SD, sD, mask_pmin) triés.
    V2: 4 champs par driver. mask_pmin pré-calculé via Cython (bitmask coprime + p_min).
    """
    temp_list = []
    seen_D = set()
    for d in sorted(odd_dict.keys()):
        sigma_d = odd_dict[d]
        D = d << 1
        for m in range(1, len(SIGMA_POW2)):
            if D > expansion_limit:
                break
            if D not in seen_D:
                seen_D.add(D)
                SD = int(SIGMA_POW2[m]) * sigma_d
                temp_list.append((D, SD, SD - D))
            D <<= 1
    temp_list.sort()
    # V2: calculer mask_pmin pour chaque driver (bitmask coprime + p_min)
    if CYTHON_AVAILABLE:
        flat_data = []
        for D, SD, sD in temp_list:
            flat_data.extend((D, SD, sD, _cython_loops.compute_mask_pmin(D)))
    else:
        # Fallback sans Cython: mask_pmin = 0 (pas de bitmask, p_min sera recalculé)
        flat_data = []
        _P_MIN_SMALL = [3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59]
        _PRIMES_24 = [3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
        for D, SD, sD in temp_list:
            mask = 0
            p_min_idx = 15
            for bit, p in enumerate(_PRIMES_24[:15]):
                if D % p == 0:
                    mask |= (1 << bit)
                elif p_min_idx == 15:
                    p_min_idx = bit
            for bit in range(15, 24):
                if D % _PRIMES_24[bit] == 0:
                    mask |= (1 << bit)
            flat_data.extend((D, SD, sD, mask | (p_min_idx << 24)))
    return flat_data, len(temp_list)

def generate_drivers_optimized(n_cible, val_max_coche=None, smooth_bound=311, extra_primes=None, max_depth=5):
    print(f"[Drivers] Génération DFS (B={smooth_bound}, depth={max_depth})...")
    start = time.time()
    n_cible_int = int(n_cible)
    ref_value = max(n_cible_int, val_max_coche) if val_max_coche else n_cible_int
    harpon_limit = n_cible_int - 1
    expansion_limit = ref_value
    all_primes = sorted(set(list(primerange(3, smooth_bound + 1)) + (extra_primes or [])))
    print(f"[Drivers] {len(all_primes)} premiers (3 → {all_primes[-1]})")
    drivers_odd = _generate_odd_drivers(all_primes, harpon_limit, max_depth)
    print(f"[Drivers] {len(drivers_odd) - 1} drivers impairs générés")
    flat_data, n_drivers = _expand_to_even_flat(drivers_odd, expansion_limit)
    del drivers_odd
    elapsed = time.time() - start
    _stats.driver_generation_time = elapsed
    ram_mb = n_drivers * 32 / 1024 / 1024
    print(f"[Drivers] OK {n_drivers} drivers en {elapsed:.2f}s ({ram_mb:.0f} MB)")
    return (flat_data, n_drivers)

def get_cached_drivers(n_cible, val_max_coche, smooth_bound, extra_primes, max_depth, use_compression=False):
    primes_key = hashlib.md5(str(sorted(extra_primes or [])).encode()).hexdigest()[:12]
    cache_base = f"drivers_v6_B{smooth_bound}_D{max_depth}_P{primes_key}"
    cache_name = f"{cache_base}.cache.gz" if use_compression else f"{cache_base}.cache"
    flat_data = None
    n_drivers = 0
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
            # V6: validate 4 fields per driver
            if len(flat_data) != n_drivers * 4:
                print(f"[Cache] X Format ancien ({len(flat_data)//n_drivers} champs), régénération...")
                flat_data = None
                _stats.cache_misses += 1
            else:
                print(f"[Cache] OK Chargé : {n_drivers} drivers ({size_mb:.1f} MB)")
                _stats.cache_hits += 1
        except Exception as e:
            print(f"[Cache] X Erreur : {e}, régénération...")
            flat_data = None
            _stats.cache_misses += 1
    else:
        _stats.cache_misses += 1
    if flat_data is None:
        print("[Drivers] Génération...")
        flat_data, n_drivers = generate_drivers_optimized(
            n_cible, val_max_coche, smooth_bound, extra_primes, max_depth
        )
        print(f"[Cache] Sauvegarde dans {cache_name}...")
        try:
            if use_compression:
                with gzip.open(cache_name, 'wb', compresslevel=6) as f:
                    pickle.dump((flat_data, n_drivers), f)
            else:
                with open(cache_name, 'wb') as f:
                    pickle.dump((flat_data, n_drivers), f)
            size_mb = os.path.getsize(cache_name) / 1024 / 1024
            print(f"[Cache] OK Sauvegardé ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[Cache] X Erreur de sauvegarde : {e}")
    print(f"[Drivers] Mise en mémoire partagée...")
    shared = SharedArray('q', n_drivers * 4, lock=False)
    shared[:] = flat_data
    del flat_data
    return (shared, n_drivers)

# ============================================================================
# WORKER DE RECHERCHE
# ============================================================================

_worker_drivers = None
_worker_n_drivers = 0

def _sieve_primes(limit):
    # P4: Cython si disponible (5-10x plus rapide)
    if CYTHON_AVAILABLE:
        return _cython_loops.sieve_primes(limit)
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return tuple(i for i in range(2, limit + 1) if is_p[i])

# P3: Crible etendu a 10^7 (664k premiers au lieu de 78k)
_sieve_start = time.time()
_SEMI_DIRECT_PRIMES = _sieve_primes(10_000_000)
_SEMI_DIRECT_P_MAX = _SEMI_DIRECT_PRIMES[-1]
_sieve_elapsed = time.time() - _sieve_start
_sieve_method = "Cython" if CYTHON_AVAILABLE else "Python"
print(f"[P3] OK {len(_SEMI_DIRECT_PRIMES):,} premiers cribles en {_sieve_elapsed:.2f}s (max={_SEMI_DIRECT_P_MAX:,}) [{_sieve_method}]")

def init_worker_with_drivers(drivers_tuple):
    global _worker_drivers, _worker_n_drivers, _worker_drv_np, _worker_sieve_np
    _worker_drivers, _worker_n_drivers = drivers_tuple
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # P7: Pré-calculer les numpy arrays pour Cython (zero-copy)
    if CYTHON_AVAILABLE:
        import numpy as np
        _worker_drv_np = np.frombuffer(_worker_drivers, dtype=np.int64)
        _worker_sieve_np = np.array(_SEMI_DIRECT_PRIMES, dtype=np.int64)
    else:
        _worker_drv_np = None
        _worker_sieve_np = None

def _get_D_factors_set(D_int):
    """Pre-factorise D pour test coprimality O(1)."""
    _D_factors = set()
    _tmp_D = D_int
    for _sp in _SMALL_PRIMES_EXTENDED:
        if _sp == 2:
            continue
        if _tmp_D % _sp == 0:
            _D_factors.add(_sp)
            while _tmp_D % _sp == 0:
                _tmp_D //= _sp
        if _tmp_D == 1:
            break
    if _tmp_D > 1:
        _D_factors.add(_tmp_D)
    return _D_factors


def worker_search_partial(args):
    """
    Recherche partielle : traite uniquement les drivers [drv_start, drv_end).
    Si do_pretests=True, exécute aussi les heuristiques rapides (Diff, Prim, Pomerance).
    Retourne (node_int, solutions_partielles, local_filter_stats).
    """
    global _worker_drivers, _worker_n_drivers
    node, drv_start, drv_end, do_pretests = args
    node_int = int(node)
    solutions = {}
    drv = _worker_drivers
    
    # Compteurs locaux (renvoyés au processus principal)
    lf = {'drivers_tested': 0, 'filtered_bisect': 0, 'filtered_pmin': 0,
          'filtered_qmax': 0, 'filtered_pmax_lt_pmin': 0, 'filtered_tq_prime': 0,
          'entered_semi_direct': 0, 'entered_quadratic': 0,
          'factorized_quadratic': 0, 'fallback_quadratic': 0}
    
    # Phase 1 : Pré-tests (seulement pour le premier chunk)
    if do_pretests:
        COMMON_DIFFS = [12, 56, 4, 8, 24, 40, 6, 20, 28, 44, 52, 60, 68, 76, 84, 92, 120, 992]
        for diff in COMMON_DIFFS:
            if diff >= node_int:
                continue
            k_candidate = node_int - diff
            if k_candidate <= 1:
                continue
            sig_k = sigma_optimized(k_candidate)
            if int(sig_k) - k_candidate == node_int:
                solutions[k_candidate] = f"Diff({diff})"
        
        SMALL_PRIMORIALS = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870]
        for primorial in SMALL_PRIMORIALS:
            offset = 2 * primorial
            if offset >= node_int:
                break
            k_candidate = node_int - offset
            if k_candidate <= 1:
                continue
            if k_candidate not in solutions:
                sig_k = sigma_optimized(k_candidate)
                if int(sig_k) - k_candidate == node_int:
                    solutions[k_candidate] = f"Prim({primorial})"
        
        h2_candidates = _improved_pomerance.generate_candidates_h2(node_int)
        for k_candidate, source_type in h2_candidates.items():
            if k_candidate not in solutions:
                sig_k = sigma_optimized(k_candidate)
                if int(sig_k) - k_candidate == node_int:
                    solutions[k_candidate] = source_type
    
    # Phase 2 : Drivers [drv_start, drv_end) — AVEC FILTRES
    _pretest_keys = set(solutions.keys())
    effective_end = drv_end  # Par défaut, sera réduit par bisect si nécessaire
    
    if drv is not None and drv_end > drv_start:
        
        # ============================================================
        # P4/P7: FAST PATH CYTHON — boucle principale fusionnée en C
        # Direct + filtres + Semi-direct en un seul appel C.
        # Les candidats retournés sont vérifiés is_prime en Python.
        # ============================================================
        if CYTHON_AVAILABLE:
            global _worker_drv_np, _worker_sieve_np
            
            direct_cands, semi_cands, c_stats = _cython_loops.driver_loop_direct_semi(
                node_int, _worker_drv_np, drv_start, drv_end,
                _worker_sieve_np, len(_SEMI_DIRECT_PRIMES), _SEMI_DIRECT_P_MAX
            )
            
            # Accumuler les stats Cython
            lf['drivers_tested'] += c_stats.get('drivers_tested', 0)
            lf['filtered_bisect'] += c_stats.get('filtered_bisect', 0)
            lf['filtered_pmin'] += c_stats.get('filtered_pmin', 0)
            lf['filtered_qmax'] += c_stats.get('filtered_qmax', 0)
            lf['filtered_pmax_lt_pmin'] += c_stats.get('filtered_pmax_lt_pmin', 0)
            lf['entered_semi_direct'] += c_stats.get('entered_semi_direct', 0)
            
            # Vérifier les candidats Direct
            for k_cand, D_cand, q_cand in direct_cands:
                if k_cand in _pretest_keys:
                    continue
                if gmpy2.is_prime(q_cand):
                    if int(sigma_optimized(k_cand)) - k_cand == node_int:
                        solutions[k_cand] = f"D({D_cand})"
                elif q_cand < 1_000_000 and math.gcd(D_cand, q_cand) == 1:
                    if int(sigma_optimized(k_cand)) - k_cand == node_int:
                        solutions[k_cand] = f"Multi({D_cand})"
            
            # Vérifier les candidats Semi-direct
            for k_cand, D_cand, p_cand, q_cand in semi_cands:
                if k_cand in _pretest_keys:
                    continue
                if gmpy2.is_prime(p_cand) and gmpy2.is_prime(q_cand):
                    if int(sigma_optimized(k_cand)) - k_cand == node_int:
                        solutions[k_cand] = f"S({D_cand})"
            
            # Quadratic: toujours en Python (rare, > 10^14)
            # Short-circuit: Quadratic ne s'active que quand p_max > SEMI_DIRECT_P_MAX
            # ce qui requiert target_q assez grand. Pour le plus petit driver D=2 (sD=1),
            # p_max ≈ sqrt(node) → p_max > 10^7 ssi node > 10^14.
            # On ne re-scanne les drivers que si node est assez grand.
            quadratic_threshold = _SEMI_DIRECT_P_MAX * _SEMI_DIRECT_P_MAX  # ~10^14
            if node_int > quadratic_threshold:
                effective_end_q = drv_end
                if drv[(drv_end - 1) * 4] > node_int:
                    lo, hi = drv_start, drv_end - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if drv[mid * 4] <= node_int:
                            lo = mid + 1
                        else:
                            hi = mid
                    effective_end_q = lo
            
                for idx in range(drv_start, effective_end_q):
                    off = idx * 4
                    D_int = drv[off]
                    SD_int = drv[off + 1]
                    if SD_int > node_int:
                        continue
                    sD = drv[off + 2]
                    if sD <= 0:
                        continue
                
                    p_min = _smallest_coprime_prime(D_int)
                    if SD_int * (1 + p_min) > node_int:
                        continue
                    target_q = sD * node_int + SD_int * D_int
                    if target_q <= 0 or target_q > MAX_TARGET_QUADRATIC:
                        continue
                    num_qmin = node_int - SD_int * (1 + p_min)
                    den_qmin = SD_int + p_min * sD
                    if num_qmin <= p_min * den_qmin:
                        continue
                    sqrt_target_int = int(gmpy2.isqrt(target_q))
                    p_max_needed = (sqrt_target_int - SD_int) // sD
                    if p_max_needed < p_min or p_max_needed <= _SEMI_DIRECT_P_MAX:
                        continue
                
                    # Quadratic zone: p > _SEMI_DIRECT_P_MAX
                    p_start = _SEMI_DIRECT_P_MAX + 1
                    div_min = p_start * sD + SD_int
                    if div_min > sqrt_target_int:
                        continue
                    n_quad_iters = (sqrt_target_int - div_min) // sD + 1
                    if n_quad_iters > QUADRATIC_MAX_ITERATIONS:
                        continue
                
                    if n_quad_iters > 300:
                        if gmpy2.is_prime(target_q):
                            lf['filtered_tq_prime'] += 1
                            continue
                        divisors = get_divisors_in_range(int(target_q), int(div_min), int(sqrt_target_int))
                        if divisors:
                            lf['factorized_quadratic'] += 1
                            for d_val in divisors:
                                diff = d_val - SD_int
                                if diff <= 0 or diff % sD != 0:
                                    continue
                                p_v = diff // sD
                                if p_v <= 1 or D_int % p_v == 0:
                                    continue
                                if not gmpy2.is_prime(p_v):
                                    continue
                                div_q = target_q // d_val
                                diff_q = div_q - SD_int
                                if diff_q <= 0 or diff_q % sD != 0:
                                    continue
                                q_v = diff_q // sD
                                if q_v <= p_v or D_int % q_v == 0:
                                    continue
                                if gmpy2.is_prime(q_v):
                                    k_quad = D_int * p_v * q_v
                                    if k_quad not in _pretest_keys:
                                        if int(sigma_optimized(k_quad)) - k_quad == node_int:
                                            solutions[k_quad] = f"Q({D_int})"
                        else:
                            lf['fallback_quadratic'] += 1
                            pq_candidates = _cython_loops.quadratic_scan_fallback(
                                target_q, div_min, sqrt_target_int, sD, SD_int, D_int
                            )
                            for p_v, q_v in pq_candidates:
                                if gmpy2.is_prime(p_v) and gmpy2.is_prime(q_v):
                                    k_quad = D_int * p_v * q_v
                                    if k_quad not in _pretest_keys:
                                        if int(sigma_optimized(k_quad)) - k_quad == node_int:
                                            solutions[k_quad] = f"Q({D_int})"
                    else:
                        lf['entered_quadratic'] += 1
                        pq_candidates = _cython_loops.quadratic_scan(
                            target_q, div_min, sqrt_target_int, sD, SD_int, D_int
                        )
                        for p_v, q_v in pq_candidates:
                            if gmpy2.is_prime(p_v) and gmpy2.is_prime(q_v):
                                k_quad = D_int * p_v * q_v
                                if k_quad not in _pretest_keys:
                                    if int(sigma_optimized(k_quad)) - k_quad == node_int:
                                        solutions[k_quad] = f"Q({D_int})"
        
            else:
                # ============================================================
                # FALLBACK PYTHON — boucle originale sans Cython
                # ============================================================
                # FILTRE 1 : Recherche binaire — éliminer D > node en O(log n)
                # Les drivers sont triés par D croissant.
                # ============================================================
                effective_end = drv_end
                if drv[(drv_end - 1) * 4] > node_int:
                    lo, hi = drv_start, drv_end - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if drv[mid * 4] <= node_int:
                            lo = mid + 1
                        else:
                            hi = mid
                    effective_end = lo
                    lf['filtered_bisect'] += (drv_end - effective_end)
        
                lf['drivers_tested'] += (drv_end - drv_start)
        
                for idx in range(drv_start, effective_end):
                    off = idx * 4
                    D_int = drv[off]
                    SD_int = drv[off + 1]
                    if SD_int > node_int:
                        continue
                    sD = drv[off + 2]
                    if sD <= 0:
                        continue
            
                    # Direct: k = D * q (test O(1) — toujours exécuté)
                    num_direct = node_int - SD_int
                    if num_direct > 0 and num_direct % sD == 0:
                        q_full = num_direct // sD
                        if q_full > 1 and D_int % q_full != 0:
                            if gmpy2.is_prime(q_full):
                                k = D_int * q_full
                                if k not in _pretest_keys:
                                    if int(sigma_optimized(k)) - k == node_int:
                                        solutions[k] = f"D({D_int})"
                            elif q_full < 1_000_000 and math.gcd(D_int, q_full) == 1:
                                k = D_int * q_full
                                if k not in _pretest_keys:
                                    if int(sigma_optimized(k)) - k == node_int:
                                        solutions[k] = f"Multi({D_int})"
            
                    # ============================================================
                    # FILTRE 2 : p_min — plus petit premier copremier à D
                    # Si σ(D)×(1+p_min) > node, aucun premier p ne donne une
                    # solution semi-directe ni quadratique → skip complet.
                    # ============================================================
                    p_min = _smallest_coprime_prime(D_int)
                    if SD_int * (1 + p_min) > node_int:
                        lf['filtered_pmin'] += 1
                        continue
            
                    # ============================================================
                    # FILTRE 3 (PRE-ISQRT) : q(p_min) ≤ p_min → skip
                    # q est décroissant en p. Si q(p_min) ≤ p_min, aucun
                    # couple (p,q) avec q > p n'existe.
                    # Équivalent à p_min ≥ p_max_needed mais SANS isqrt.
                    # Coût : 2 multiplications + 1 comparaison.
                    # ============================================================
                    target_q = sD * node_int + SD_int * D_int
                    if target_q <= 0:
                        continue
            
                    # q(p_min) = num_qmin / den_qmin ; skip si ≤ p_min
                    num_qmin = node_int - SD_int * (1 + p_min)  # > 0 (filtre 2 passé)
                    den_qmin = SD_int + p_min * sD
                    if num_qmin <= p_min * den_qmin:
                        lf['filtered_qmax'] += 1
                        continue
            
                    # ============================================================
                    # ISQRT + p_max (nécessaire pour borner les boucles)
                    # ============================================================
                    sqrt_target_int = int(gmpy2.isqrt(target_q))
                    p_max_needed = (sqrt_target_int - SD_int) // sD
            
                    # Sécurité: si arrondi d'isqrt donne p_max < p_min (rare)
                    if p_max_needed < p_min:
                        lf['filtered_pmax_lt_pmin'] += 1
                        continue
            
                    # ========================================================
                    # Semi-direct : k = D * p * q (sieve rapide, p ≤ 10^6)
                    # ========================================================
                    if p_max_needed <= _SEMI_DIRECT_P_MAX:
                        lf['entered_semi_direct'] += 1
                
                        # P3/P4: Pre-factoriser D + Cython si disponible
                        D_factors = _get_D_factors_set(D_int) if p_max_needed > 200 else set()
                
                        if CYTHON_AVAILABLE and p_max_needed > 500:
                            # P4: Cython pour la boucle semi-directe
                            candidates = _cython_loops.semi_direct_search(
                                node_int, D_int, SD_int, sD, p_max_needed,
                                _SEMI_DIRECT_PRIMES, D_factors
                            )
                            for p_v, q_v in candidates:
                                if gmpy2.is_prime(p_v) and gmpy2.is_prime(q_v):
                                    k_semi = D_int * p_v * q_v
                                    if k_semi not in _pretest_keys:
                                        if int(sigma_optimized(k_semi)) - k_semi == node_int:
                                            solutions[k_semi] = f"S({D_int})"
                        elif D_factors:
                            for p in _SEMI_DIRECT_PRIMES:
                                if p > p_max_needed:
                                    break
                                if p in D_factors:
                                    continue
                                SD_1p = SD_int * (1 + p)
                                if SD_1p > node_int:
                                    break
                                num_q = node_int - SD_1p
                                den = SD_int + p * sD
                                if num_q % den != 0:
                                    continue
                                q_v = num_q // den
                                if q_v <= p or D_int % q_v == 0 or not gmpy2.is_prime(q_v):
                                    continue
                                k_semi = D_int * p * q_v
                                if k_semi not in _pretest_keys:
                                    if int(sigma_optimized(k_semi)) - k_semi == node_int:
                                        solutions[k_semi] = f"S({D_int})"
                        else:
                            for p in _SEMI_DIRECT_PRIMES:
                                if p > p_max_needed:
                                    break
                                if D_int % p == 0:
                                    continue
                                SD_1p = SD_int * (1 + p)
                                if SD_1p > node_int:
                                    break
                                num_q = node_int - SD_1p
                                den = SD_int + p * sD
                                if num_q % den != 0:
                                    continue
                                q_v = num_q // den
                                if q_v <= p or D_int % q_v == 0 or not gmpy2.is_prime(q_v):
                                    continue
                                k_semi = D_int * p * q_v
                                if k_semi not in _pretest_keys:
                                    if int(sigma_optimized(k_semi)) - k_semi == node_int:
                                        solutions[k_semi] = f"S({D_int})"
            
                    # ========================================================
                    # Quadratic : recherche de paires (p,q) via diviseurs
                    # de target_q = (SD + sD·p)(SD + sD·q)
                    # ========================================================
                    if target_q > MAX_TARGET_QUADRATIC:
                        continue
            
                    # Borne inférieure : p_start évite de retester les p
                    # déjà couverts par Semi-direct
                    if p_max_needed <= _SEMI_DIRECT_P_MAX:
                        p_start = _SEMI_DIRECT_P_MAX + 1
                    else:
                        p_start = p_min
                    div_min = p_start * sD + SD_int
            
                    # Rien à chercher si div_min > √target_q
                    if div_min > sqrt_target_int:
                        continue
            
                    n_quad_iters = (sqrt_target_int - div_min) // sD + 1
            
                    if n_quad_iters > QUADRATIC_MAX_ITERATIONS:
                        continue
            
                    # ============================================================
                    # STRATÉGIE ADAPTATIVE :
                    # - Peu d'itérations (≤ 300) → scan linéaire direct.
                    #   Le scan est MOINS cher que is_prime(target_q) (~5µs)
                    #   car 300 × modulo < Miller-Rabin. On ne teste donc
                    #   PAS la primalité ici — si target_q est premier, le
                    #   scan ne trouvera rien en ~15µs (acceptable).
                    #
                    # - Beaucoup d'itérations (> 300) → tester is_prime d'abord
                    #   (évite une factorisation coûteuse pour ~5% des cas),
                    #   puis factoriser target_q et énumérer ses diviseurs.
                    # ============================================================
                    if n_quad_iters > 300:
                        # Test primalité UNIQUEMENT pour le path factorisé
                        if gmpy2.is_prime(target_q):
                            lf['filtered_tq_prime'] += 1
                            continue
                
                        # P6: APPROCHE FACTORISEE avec pruning DFS
                        divisors = get_divisors_in_range(int(target_q), int(div_min), int(sqrt_target_int))
                        if divisors:
                            lf['factorized_quadratic'] += 1
                            for d_val in divisors:
                                diff = d_val - SD_int
                                if diff <= 0 or diff % sD != 0:
                                    continue
                                p_v = diff // sD
                                if p_v <= 1 or D_int % p_v == 0:
                                    continue
                                if not gmpy2.is_prime(p_v):
                                    continue
                                div_q = target_q // d_val
                                diff_q = div_q - SD_int
                                if diff_q <= 0 or diff_q % sD != 0:
                                    continue
                                q_v = diff_q // sD
                                if q_v <= p_v or D_int % q_v == 0:
                                    continue
                                if gmpy2.is_prime(q_v):
                                    k_quad = D_int * p_v * q_v
                                    if k_quad not in _pretest_keys:
                                        if int(sigma_optimized(k_quad)) - k_quad == node_int:
                                            solutions[k_quad] = f"Q({D_int})"
                        else:
                            # Factorisation échouée → fallback scan linéaire
                            lf['fallback_quadratic'] += 1
                            # P4: Cython pour le fallback scan (3-5x plus rapide)
                            if CYTHON_AVAILABLE:
                                pq_candidates = _cython_loops.quadratic_scan_fallback(
                                    target_q, div_min, sqrt_target_int, sD, SD_int, D_int
                                )
                                for p_v, q_v in pq_candidates:
                                    if gmpy2.is_prime(p_v) and gmpy2.is_prime(q_v):
                                        k_quad = D_int * p_v * q_v
                                        if k_quad not in _pretest_keys:
                                            if int(sigma_optimized(k_quad)) - k_quad == node_int:
                                                solutions[k_quad] = f"Q({D_int})"
                            else:
                                d = div_min
                                while d <= sqrt_target_int:
                                    if target_q % d == 0:
                                        diff = d - SD_int
                                        if diff > 0 and diff % sD == 0:
                                            p_v = diff // sD
                                            if p_v > 1 and D_int % p_v == 0:
                                                d += sD
                                                continue
                                            if gmpy2.is_prime(p_v):
                                                div_q = target_q // d
                                                diff_q = div_q - SD_int
                                                if diff_q > 0 and diff_q % sD == 0:
                                                    q_v = diff_q // sD
                                                    if q_v > p_v and D_int % q_v != 0:
                                                        if gmpy2.is_prime(q_v):
                                                            k_quad = D_int * p_v * q_v
                                                            if k_quad not in _pretest_keys:
                                                                if int(sigma_optimized(k_quad)) - k_quad == node_int:
                                                                    solutions[k_quad] = f"Q({D_int})"
                                    d += sD
                    else:
                        # --- SCAN LINÉAIRE DIRECT (≤ 300 itérations) ---
                        # Pas de test de primalité : le scan coûte ≤ 15µs,
                        # moins cher que is_prime (~5µs) dans 95% des cas.
                        lf['entered_quadratic'] += 1
                        # P4: Cython pour le scan linéaire (3-5x plus rapide)
                        if CYTHON_AVAILABLE:
                            pq_candidates = _cython_loops.quadratic_scan(
                                target_q, div_min, sqrt_target_int, sD, SD_int, D_int
                            )
                            for p_v, q_v in pq_candidates:
                                if gmpy2.is_prime(p_v) and gmpy2.is_prime(q_v):
                                    k_quad = D_int * p_v * q_v
                                    if k_quad not in _pretest_keys:
                                        if int(sigma_optimized(k_quad)) - k_quad == node_int:
                                            solutions[k_quad] = f"Q({D_int})"
                        else:
                            d = div_min
                            while d <= sqrt_target_int:
                                if target_q % d == 0:
                                    diff = d - SD_int
                                    if diff > 0 and diff % sD == 0:
                                        p_v = diff // sD
                                        if p_v > 1 and D_int % p_v == 0:
                                            d += sD
                                            continue
                                        if gmpy2.is_prime(p_v):
                                            div_q = target_q // d
                                            diff_q = div_q - SD_int
                                            if diff_q > 0 and diff_q % sD == 0:
                                                q_v = diff_q // sD
                                                if q_v > p_v and D_int % q_v != 0:
                                                    if gmpy2.is_prime(q_v):
                                                        k_quad = D_int * p_v * q_v
                                                        if k_quad not in _pretest_keys:
                                                            if int(sigma_optimized(k_quad)) - k_quad == node_int:
                                                                solutions[k_quad] = f"Q({D_int})"
                                d += sD
    
    return (node_int, solutions, lf)


def worker_search(node):
    """Recherche complète (tous les drivers) - compatibilité avec l'ancien mode."""
    global _worker_n_drivers
    return worker_search_partial((node, 0, _worker_n_drivers, True))



# ============================================================================
# CLASSE PRINCIPALE
# ============================================================================

class ArbreAliquoteV6:
    def __init__(self, n_cible, profondeur=100, smooth_bound=120, extra_primes=None, 
                 max_depth=6, use_compression=False, allow_empty_exploration=False):
        self.cible_initiale = int(n_cible)
        self.profondeur = profondeur
        self.smooth_bound = smooth_bound
        self.extra_primes = extra_primes or []
        self.max_depth = max_depth
        self.use_compression = use_compression
        self.allow_empty_exploration = allow_empty_exploration  # ✅ NOUVEAU
        self.max_workers = max(1, cpu_count() - 2)
        self.global_cache = GlobalAntecedenteCache(cache_dir=".", use_compression=use_compression)
        self.explored = set()
        self.old_cache_file = f"cache_arbre_{self.cible_initiale}.jsonl"
        self.old_cache_json = f"cache_arbre_{self.cible_initiale}.json"
        self.stop = False
        self.start_time = time.time()
        self.val_max_coche = self.cible_initiale
        self._load_explored_nodes()
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _load_explored_nodes(self):
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
            except Exception:
                pass
        
        # ✅ NOUVEAU: Comportement conditionnel selon allow_empty_exploration
        if self.allow_empty_exploration:
            # Mode ré-exploration des nœuds vides: ne marquer comme explorés que les nœuds avec antécédents
            empty_nodes_count = 0
            for aliquot_str, antecedents in self.global_cache.cache.items():
                if antecedents:  # Seulement si le nœud a des antécédents
                    self.explored.add(int(aliquot_str))
                else:
                    empty_nodes_count += 1
            if self.explored:
                print(f"[Cache Global] {len(self.explored)} sommes aliquotes avec antécédents (explorées)")
            if empty_nodes_count:
                print(f"[Cache Global] OK {empty_nodes_count} nœuds vides {{}} disponibles pour ré-exploration")
        else:
            # Mode standard: marquer tous les nœuds comme explorés
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
        _stats.report(force=True)
        self.afficher()
        sys.exit(0)
    
    def _save_node(self, node_val, solutions):
        try:
            self.global_cache.add_antecedents(node_val, solutions or {})
        except Exception as e:
            print(f"[Cache Global] Erreur sauvegarde: {e}")
    
    def construire(self, reprise_active=False):
        print(f"\n{'='*70}")
        print(f"CONSTRUCTION ARBRE V5 - CIBLE {self.cible_initiale}")
        print(f"{'='*70}")
        print(f"Profondeur : {self.profondeur}")
        print(f"Smooth bound : {self.smooth_bound}")
        print(f"Max depth : {self.max_depth}")
        print(f"Compression cache : {'OUI' if self.use_compression else 'NON'}")
        print(f"Processeurs : {cpu_count()}")
        print(f"Mode reprise : {'OUI' if reprise_active else 'NON'}")
        print(f"{'='*70}\n")
        
        drivers_array, n_drivers = get_cached_drivers(
            self.cible_initiale,
            self.val_max_coche,
            self.smooth_bound,
            self.extra_primes,
            self.max_depth,
            self.use_compression
        )
        
        if reprise_active:
            current_gen, start_gen = self._resume_from_cache()
            if not current_gen:
                current_gen = [self.cible_initiale]
                start_gen = 1
        else:
            current_gen = [self.cible_initiale]
            start_gen = 1
        
        num_workers = max(1, cpu_count() - 2)
        pool = Pool(
            num_workers,
            initializer=init_worker_with_drivers,
            initargs=((drivers_array, n_drivers),)
        )
        try:
            for gen in range(start_gen, self.profondeur + 1):
                if self.stop or not current_gen:
                    break
                print(f"\n--- GÉNÉRATION {gen} ({len(current_gen)} nœuds) ---")
                gen_start = time.time()
                
                if reprise_active and gen == start_gen:
                    to_compute = current_gen
                    print(f"[G{gen}] Traitement frontière : {len(to_compute)} nœuds (reprise)")
                else:
                    # ✅ OPTIMISÉ: Filtrage en une seule passe avec marquage immédiat des nœuds > 100x
                    limit_100x = self.val_max_coche * 100
                    to_compute = []
                    beyond_limit_count = 0
                    
                    for v in current_gen:
                        if v in self.explored:
                            continue  # Déjà traité
                        
                        if v > limit_100x:
                            # Nœud > 100x cible : marquer comme feuille sans calcul
                            self._save_node(v, {})  # Pas d'antécédents
                            self.explored.add(v)
                            beyond_limit_count += 1
                        else:
                            # Nœud à explorer normalement
                            to_compute.append(v)
                    
                    if beyond_limit_count > 0:
                        print(f"[G{gen}] {beyond_limit_count} nœuds > {limit_100x} (100x cible) marqués comme feuilles (non explorés)")
                    
                    if not to_compute:
                        print(f"[G{gen}] Tous les nœuds déjà explorés, navigation via cache...")
                        next_gen_from_cache = []
                        empty_nodes = []  # Nœuds vides à ré-explorer
                        for v in current_gen:
                            ants = self.global_cache.get_antecedents(v)
                            if ants is not None:  # Nœud existant dans le cache
                                if ants:  # Nœud avec antécédents
                                    next_gen_from_cache.extend(
                                        [int(k) for k in ants.keys() if int(k) <= self.val_max_coche * 100]
                                    )
                                elif self.allow_empty_exploration:  # Nœud vide {} - ré-exploration si activée
                                    empty_nodes.append(v)
                        
                        if empty_nodes and self.allow_empty_exploration:
                            print(f"[G{gen}] OK {len(empty_nodes)} nœuds vides détectés - ajoutés pour ré-exploration")
                            # Retirer ces nœuds de explored pour permettre leur retraitement
                            for node in empty_nodes:
                                if node in self.explored:
                                    self.explored.discard(node)
                            next_gen_from_cache.extend(empty_nodes)
                        
                        if not next_gen_from_cache:
                            print(" -> Fin de branche atteinte (feuilles).")
                            break
                        current_gen = list(set(next_gen_from_cache))
                        continue
                
                next_gen = set()
                processed = 0
                
                # ✅ MODE COOPÉRATIF : si moins de nœuds que de workers,
                # chaque nœud est découpé en tranches de drivers traitées en parallèle
                if len(to_compute) < num_workers and n_drivers > num_workers:
                    workers_per_node = num_workers // len(to_compute)
                    chunk_size = (n_drivers + workers_per_node - 1) // workers_per_node
                    
                    print(f"[G{gen}] MODE COOPÉRATIF : {len(to_compute)} nœuds × {workers_per_node} workers/nœud "
                          f"({chunk_size} drivers/chunk sur {n_drivers})")
                    
                    # Construire les tâches partielles
                    partial_tasks = []
                    for node_val in to_compute:
                        for w in range(workers_per_node):
                            drv_start = w * chunk_size
                            drv_end = min((w + 1) * chunk_size, n_drivers)
                            if drv_start >= n_drivers:
                                break
                            # Seul le premier chunk fait les pré-tests (Diff, Prim, Pomerance)
                            do_pretests = (w == 0)
                            partial_tasks.append((node_val, drv_start, drv_end, do_pretests))
                    
                    print(f"[G{gen}] {len(partial_tasks)} tâches partielles soumises")
                    
                    # Collecter les résultats partiels par nœud
                    partial_results = {}  # {node_int: {k: type, ...}}
                    for node_int, partial_solutions, local_fs in pool.imap_unordered(
                            worker_search_partial, partial_tasks, chunksize=1):
                        if self.stop:
                            break
                        if node_int not in partial_results:
                            partial_results[node_int] = {}
                        partial_results[node_int].update(partial_solutions)
                        for key in local_fs:
                            _filter_stats[key] += local_fs[key]
                    
                    # Fusionner et sauvegarder
                    for node_val in to_compute:
                        if self.stop:
                            break
                        solutions = partial_results.get(node_val, {})
                        self._save_node(node_val, solutions)
                        self.explored.add(node_val)
                        _stats.total_nodes_processed += 1
                        for sol_type in solutions.values():
                            sol_prefix = sol_type.split('(')[0]
                            _stats.add_solution(sol_prefix)
                        for k in solutions:
                            next_gen.add(int(k))
                        processed += 1
                        print(f"   -> {node_val}: {len(solutions)} antécédents trouvés (coopératif)")
                
                else:
                    # ✅ MODE STANDARD : 1 nœud = 1 worker (comme avant)
                    for node_val, solutions, local_fs in pool.imap_unordered(worker_search, to_compute, chunksize=1):
                        if self.stop:
                            break
                        for key in local_fs:
                            _filter_stats[key] += local_fs[key]
                        self._save_node(node_val, solutions)
                        self.explored.add(node_val)
                        _stats.total_nodes_processed += 1
                        for sol_type in solutions.values():
                            sol_prefix = sol_type.split('(')[0]
                            _stats.add_solution(sol_prefix)
                        for k in solutions:
                            next_gen.add(int(k))
                        processed += 1
                        if processed % 10 == 0:
                            print(f"\r   -> {processed}/{len(to_compute)}", end='')
                
                gen_time = time.time() - gen_start
                _stats.add_generation_time(gen_time)
                print(f"\n[G{gen}] OK {len(next_gen)} nouvelles branches ({gen_time:.2f}s)")
                
                # ✅ CORRECTIF CACHE: Filtrer APRÈS avoir tout sauvegardé
                # Tous les enfants sont dans le cache, mais on explore que les pertinents
                next_gen_filtered = {k for k in next_gen if k <= self.val_max_coche * 100}
                if len(next_gen_filtered) < len(next_gen):
                    print(f"[G{gen}] Filtre 100x: {len(next_gen_filtered)}/{len(next_gen)} nœuds gardés pour exploration")
                
                current_gen = list(next_gen_filtered)
                if reprise_active and gen == start_gen:
                    reprise_active = False
                
                # ✅ CORRECTIF CACHE: Sauvegarder CHAQUE génération (pas toutes les 2)
                self.global_cache.save()
        finally:
            pool.close()
            pool.join()
            _stats.report()
            self.afficher()
    
    def _resume_from_cache(self):
        raw_cache = self.global_cache.cache or {}
        
        # ✅ CORRECTIF: Charger d'abord TOUT le cache
        full_cache = {}
        for parent_str, enfants in raw_cache.items():
            try:
                parent = int(parent_str)
            except Exception:
                continue
            child_keys = []
            if isinstance(enfants, dict):
                for child_str in enfants.keys():
                    try:
                        child_keys.append(int(child_str))
                    except Exception:
                        continue
            full_cache[parent] = child_keys
        
        if not full_cache:
            print("[Reprise] Cache global vide, démarrage normal")
            return ([self.cible_initiale], 1)
        
        print(f"[Reprise] Cache global chargé : {len(full_cache)} nœuds totaux")
        
        # ✅ CORRECTIF: Filtrer UNIQUEMENT les nœuds de l'arbre de la cible actuelle
        root = self.cible_initiale
        
        if root not in full_cache:
            print(f"[Reprise] Cible {root} non trouvée dans le cache, démarrage normal")
            return ([root], 1)
        
        # BFS pour identifier UNIQUEMENT les nœuds de cet arbre
        queue = deque([(root, 0)])
        tree_nodes = {root}  # Nœuds appartenant à cet arbre
        cache = {}  # Cache filtré pour cet arbre seulement
        max_depth_found = 0
        
        print(f"[Reprise] Extraction de l'arbre depuis la racine {root}...")
        
        while queue:
            node, d = queue.popleft()
            max_depth_found = max(max_depth_found, d)
            
            if node in full_cache:
                cache[node] = full_cache[node]  # Ajouter au cache filtré
                for child in full_cache[node]:
                    if child not in tree_nodes:
                        tree_nodes.add(child)
                        queue.append((child, d + 1))
        
        print(f"[Reprise] Arbre de {root} : {len(cache)} nœuds explorés (sur {len(full_cache)} totaux)")
        print(f"[Reprise] Nœuds de cet arbre : {len(tree_nodes)}")
        
        # Marquer SEULEMENT les nœuds de cet arbre comme explorés
        for parent in cache.keys():
            self.explored.add(parent)
        print(f"[Reprise] {len(self.explored)} nœuds marqués comme explorés")
        
        # Identifier la frontière (enfants dans tree_nodes mais pas dans cache)
        all_children = set()
        for enfants in cache.values():
            for e in enfants:
                all_children.add(e)
        
        frontier = {e for e in all_children if e not in cache}
        print(f"[Reprise] Frontière brute identifiée : {len(frontier)} nœuds")
        
        if not frontier:
            print("[Reprise] ⚠️  Frontière vide (arbre complet ?)")
            return ([], 1)
        
        depth = max_depth_found + 1
        print(f"[Reprise] Profondeur estimée : {depth}")
        
        frontier_filtered = [f for f in frontier if f <= self.val_max_coche * 100]
        if len(frontier_filtered) < len(frontier):
            excluded = len(frontier) - len(frontier_filtered)
            print(f"[Reprise] {excluded} nœuds exclus (> 100x val_coche)")
        
        frontier_list = sorted(frontier_filtered)
        print(f"[Reprise] Frontière finale : {len(frontier_list)} nœuds actifs")
        return (frontier_list, depth)
    
    def afficher(self):
        total_time = time.time() - self.start_time
        root = int(self.cible_initiale)
        print(f"\n{'='*70}")
        print(f"RÉSULTATS : MINIMUM PAR BRANCHE DE {root}")
        print(f"(Plus petit nombre < {self.val_max_coche} de chaque branche)")
        print(f"{'='*70}")
        raw_cache = self.global_cache.cache or {}
        numeric_cache = {}
        for parent_str, enfants in raw_cache.items():
            try:
                parent = int(parent_str)
            except Exception:
                continue
            child_map = {}
            if isinstance(enfants, dict):
                for child_str, typ in enfants.items():
                    try:
                        child_map[int(child_str)] = typ
                    except Exception:
                        continue
            numeric_cache[parent] = child_map
        if root not in numeric_cache and not any(root in v for v in numeric_cache.values()):
            print("Aucune donnée dans le cache global pour cette cible.")
            print(f"Temps : {total_time:.2f}s")
            return
        nodes_of_interest = set()
        parents_map = {}
        queue = deque([root])
        visited_bfs = {root}
        while queue:
            current_node = queue.popleft()
            children_map = numeric_cache.get(current_node, {})
            for child in children_map.keys():
                if child not in visited_bfs:
                    visited_bfs.add(child)
                    parents_map[child] = current_node
                    queue.append(child)
                    nodes_of_interest.add(child)
        candidates = sorted([n for n in nodes_of_interest if n < self.val_max_coche])
        if not candidates:
            print(f"Aucun antécédent < {self.val_max_coche}")
            print(f"Temps : {total_time:.2f}s")
            return
        minima_descendants = []
        for val in candidates:
            path = []
            curr = val
            safety = 0
            while curr in parents_map and safety < 1000:
                path.append(curr)
                curr = parents_map[curr]
                safety += 1
            path.append(root)
            path.reverse()
            # Check ancestors (all nodes between root and val, exclusive of both)
            ancestors_on_path = path[1:-1]  # Exclude root (path[0]) and val (path[-1])
            has_smaller_ancestor = any(a < val for a in ancestors_on_path)
            if not has_smaller_ancestor:
                minima_descendants.append((val, path))
        if not minima_descendants:
            print(f"Aucun minimum descendant trouvé")
            print(f"Temps : {total_time:.2f}s")
            return
        linear_families = []
        for val, path in minima_descendants:
            found_family = False
            for family in linear_families:
                for fam_val, fam_path in family:
                    if val in fam_path or fam_val in path:
                        family.append((val, path))
                        found_family = True
                        break
                if found_family:
                    break
            if not found_family:
                linear_families.append([(val, path)])
        final_minima = []
        for family in linear_families:
            family.sort(key=lambda x: x[0])
            smallest_val, smallest_path = family[0]
            final_minima.append((smallest_val, smallest_path))
        final_minima.sort(key=lambda x: x[0])
        print(f"Minimum par branche : {len(final_minima)}")
        print(f"(Minima descendants avant filtrage : {len(minima_descendants)})")
        print(f"(Total candidats : {len(candidates)})")
        print(f"(Total exploré (BFS) : {len(visited_bfs) - 1} nœuds)")
        print(f"Temps : {total_time:.2f}s\n")
        print(f"{'MINIMUM':<20} | {'CHAÎNE COMPLÈTE'}")
        print("-" * 75)
        for val, path in final_minima:
            path_str = " → ".join(map(str, path))
            print(f"{val:<20} | {path_str}")
        print("-" * 75)
        print(f"\nSTATISTIQUES :")
        print(f"  • Chemins linéaires distincts : {len(linear_families)}")
        print(f"  • Minima finaux affichés      : {len(final_minima)}")
        print(f"  • Minima filtrés              : {len(minima_descendants) - len(final_minima)}")
        print(f"  • Taux de compaction          : {len(final_minima)/len(minima_descendants)*100:.1f}%")
        multi_minima_families = [fam for fam in linear_families if len(fam) > 1]
        if multi_minima_families:
            print(f"\n  Chemins avec minima multiples (gardé le plus petit) :")
            for family in multi_minima_families:
                family.sort(key=lambda x: x[0])
                kept = family[0][0]
                eliminated = [v for v, _ in family[1:]]
                print(f"    Gardé: {kept:,} | Éliminés: {eliminated}")
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
        description="Recherche d'antécédents aliquotes - Version V5 Optimisée"
    )
    parser.add_argument("val_coche", type=int, help="Valeur max")
    parser.add_argument("n", type=int, help="Cible")
    parser.add_argument("--depth", type=int, default=100, help="Profondeur")
    parser.add_argument("--smooth-bound", type=int, default=311, help="Borne B")
    parser.add_argument("--extra-primes", type=int, nargs='*',
                        default=[313, 419, 439, 457, 541, 907],
                        help="Grands premiers")
    parser.add_argument("--max-driver-depth", type=int, default=4,
                        help="Max premiers distincts")
    parser.add_argument("--compress", action='store_true',
                        help="Compresser le cache (gzip)")
    parser.add_argument("--resume", action='store_true',
                        help="Reprendre depuis le cache global")
    parser.add_argument("--allow-empty-node-exploration", action='store_true',
                        help="Permettre la ré-exploration des nœuds vides {{}} (défaut: désactivé)")
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
    print("  VERSION V6 TURBO - P1:Fused-sigma P2:ECM P3:10M-sieve P4:Cython+")
    print("="*70)
    print(f"  • Cible : {args.n}")
    print(f"  • Smooth bound : {args.smooth_bound}")
    print(f"  • Max depth : {args.max_driver_depth}")
    print(f"  • Compression : {'OUI' if args.compress else 'NON'}")
    print(f"  • Reprise : {'OUI' if args.resume else 'NON'}")
    print(f"  • Ré-exploration nœuds vides : {'OUI' if args.allow_empty_node_exploration else 'NON'}")
    if NUMBA_AVAILABLE:
        print(f"  • Numba : ACTIVÉ (seuil: {NUMBA_THRESHOLD:,})")
    else:
        print(f"  • Numba : NON DISPONIBLE (pip install numba)")
    print("="*70 + "\n")
    
    app = ArbreAliquoteV6(
        n_cible=args.n,
        profondeur=args.depth,
        smooth_bound=args.smooth_bound,
        extra_primes=args.extra_primes,
        max_depth=args.max_driver_depth,
        use_compression=args.compress,
        allow_empty_exploration=args.allow_empty_node_exploration
    )
    app.val_max_coche = args.val_coche
    app.construire(reprise_active=args.resume)
    app.construire(reprise_active=args.resume)