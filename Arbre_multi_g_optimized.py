#!/usr/bin/env python3
"""
Recherche d'antécédents aliquotes - Version CORRIGÉE
=====================================================
- Numba pour sigma/is_prime/pollard_rho (gains 5-15x)
- sigma_pollard_numba pour n >= 10^7 (O(n^{1/4}) vs O(n^{1/2}))
- Pré-filtrage arithmétique des candidats Pomerance avant σ
- Recherche quadratique hybride (diviseurs Pollard + fallback linéaire)
"""

import gmpy2
from gmpy2 import mpz
import sys
import bisect
import time
import signal
import argparse
import json
import os
import math
import pickle
import gzip
from multiprocessing import Pool, cpu_count, Array as SharedArray
from sympy import primerange
from collections import defaultdict, OrderedDict, deque

# ============================================================================
# NUMBA OPTIMIZATION
# ============================================================================
try:
    from numba import njit
    NUMBA_AVAILABLE = True
    NUMBA_THRESHOLD = 10**15
except ImportError:
    NUMBA_AVAILABLE = False
    NUMBA_THRESHOLD = 0
    print("[NUMBA] X Numba non disponible - Mode gmpy2 uniquement")
    print("        Pour installer: pip install numba")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================
SIGMA_CACHE_SIZE = 100000
DIVISORS_CACHE_SIZE = 25000
MAX_DIVISORS = 8000
MAX_TARGET_QUADRATIC = 10**18
MAX_POLLARD_ITERATIONS = 30000
QUADRATIC_MAX_ITERATIONS = 1_000_000
FACTORIZE_THRESHOLD = 10_000_000
GAMMA = 0.57721566490153286
EXP_GAMMA = math.exp(GAMMA)
SIGMA_POW2 = tuple(mpz((1 << (m + 1)) - 1) for m in range(33))

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
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    if n < 41:
        return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small_primes:
        if a >= n:
            continue
        x = _powmod(a, d, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(r - 1):
            x = _mulmod(x, x, n)
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True

@njit(cache=True)
def pollard_rho_numba(n, max_iterations=100000):
    if n == 1:
        return 1
    if n % 2 == 0:
        return 2
    if is_prime_numba(n):
        return n
    for c_val in range(1, 50):
        y, g, r, q = 2, 1, 1, 1
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
    return 0

@njit(cache=True)
def sigma_numba(n):
    """σ(n) par trial division pure (3..97 + roue mod 30). Optimal pour n < 10^7."""
    if n < 2:
        return 1 if n == 1 else 0
    total = 1
    temp_n = n
    tz = 0
    while temp_n % 2 == 0:
        tz += 1
        temp_n //= 2
    if tz > 0:
        total = (1 << (tz + 1)) - 1
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
    wheel_increments = (4, 2, 4, 2, 4, 6, 2, 6)
    d = 101
    wi = 1
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
        wi = (wi + 1) & 7
    if temp_n > 1:
        total *= (1 + temp_n)
    return total

@njit(cache=True)
def _sigma_of_prime_power(p, e):
    """σ(p^e) = 1 + p + p² + ... + p^e."""
    s = 1
    pk = 1
    for _ in range(e):
        pk *= p
        s += pk
    return s

@njit(cache=True)
def _pollard_find_factor(n):
    """Trouve un facteur non-trivial de n (composite). Retourne 0 si échec."""
    for c_val in range(1, 20):
        y, g, r, q = 2, 1, 1, 1
        iterations = 0
        while g == 1 and iterations < 100000:
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
                    if iterations >= 100000:
                        break
                g = gcd_numba(q, n)
                k += batch_size
            r *= 2
        if g != n and g != 1:
            return g
    return 0

@njit(cache=True)
def sigma_pollard_numba(n):
    """σ(n) intégré: trial division (3..541) + Pollard-ρ. O(n^{1/4})."""
    if n < 2:
        return 1 if n == 1 else 0

    found_primes = [0] * 64
    found_exps = [0] * 64
    n_found = 0
    temp_n = n

    # --- Facteur 2 ---
    tz = 0
    while temp_n % 2 == 0:
        tz += 1
        temp_n //= 2
    if tz > 0:
        found_primes[0] = 2
        found_exps[0] = tz
        n_found = 1

    # --- Tous les premiers de 3 à 541 ---
    small_primes = (
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
        109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
        173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
        233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283,
        293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359,
        367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431,
        433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491,
        499, 503, 509, 521, 523, 541
    )
    for p in small_primes:
        if p * p > temp_n:
            break
        if temp_n % p == 0:
            e = 0
            while temp_n % p == 0:
                e += 1
                temp_n //= p
            found_primes[n_found] = p
            found_exps[n_found] = e
            n_found += 1

    # --- Pollard-ρ pour le cofacteur résiduel ---
    if temp_n > 1:
        if is_prime_numba(temp_n):
            found_primes[n_found] = temp_n
            found_exps[n_found] = 1
            n_found += 1
        else:
            stack = [0] * 64
            stack[0] = temp_n
            stack_top = 1

            while stack_top > 0:
                stack_top -= 1
                current = stack[stack_top]

                if current <= 1:
                    continue

                if is_prime_numba(current):
                    found_idx = -1
                    for i in range(n_found):
                        if found_primes[i] == current:
                            found_idx = i
                            break
                    if found_idx >= 0:
                        found_exps[found_idx] += 1
                    else:
                        found_primes[n_found] = current
                        found_exps[n_found] = 1
                        n_found += 1
                    continue

                f = _pollard_find_factor(current)

                if f == 0 or f == current or f <= 1:
                    # Fallback trial division
                    d2 = 547
                    while d2 * d2 <= current:
                        if current % d2 == 0:
                            e = 0
                            while current % d2 == 0:
                                e += 1
                                current //= d2
                            found_idx = -1
                            for i in range(n_found):
                                if found_primes[i] == d2:
                                    found_idx = i
                                    break
                            if found_idx >= 0:
                                found_exps[found_idx] += e
                            else:
                                found_primes[n_found] = d2
                                found_exps[n_found] = e
                                n_found += 1
                        d2 += 2
                    if current > 1:
                        found_idx = -1
                        for i in range(n_found):
                            if found_primes[i] == current:
                                found_idx = i
                                break
                        if found_idx >= 0:
                            found_exps[found_idx] += 1
                        else:
                            found_primes[n_found] = current
                            found_exps[n_found] = 1
                            n_found += 1
                    continue

                # f est un facteur non-trivial — empiler les deux moitiés
                if stack_top < 62:
                    stack[stack_top] = f
                    stack_top += 1
                    stack[stack_top] = current // f
                    stack_top += 1

    # --- Calcul final de σ ---
    total = 1
    for i in range(n_found):
        total *= _sigma_of_prime_power(found_primes[i], found_exps[i])
    return total


# ============================================================================
# HEURISTIQUE DE POMERANCE AMÉLIORÉE
# ============================================================================
class ImprovedPomeranceH2:
    def __init__(self):
        self.standard_ratios = [
            (129, 100), (77, 100), (13, 10), (7, 10), (3, 2),
            (3, 5), (5, 8), (8, 13), (13, 21), (4, 5), (5, 6), (6, 7), (7, 8),
        ]
        self.extended_ratios = [
            (71, 55), (148, 151), (655, 731), (7, 11), (11, 13), (13, 17),
            (55, 89), (89, 144), (2, 5), (5, 12),
        ]
        self.multipliers_small = (2, 3, 5)
        self.multipliers_medium = (2, 3, 5, 7)
        self.multipliers_large = (2, 3, 5, 7, 11)
        self.robin_cache = OrderedDict()
        self.max_robin_cache = 1000
        self.loglog_cache = OrderedDict()
        self.max_loglog_cache = 500
        self.stats = {'std': 0, 'ext': 0, 'pow2': 0, 'filtered': 0, 'total_generated': 0, 'cache_hits': 0}
        self.max_candidates_per_node = 200
        self.enable_pomerance = True

    def get_multipliers(self, n):
        if n < 1_000_000:
            return self.multipliers_small
        elif n < 1_000_000_000:
            return self.multipliers_medium
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
        if not self.enable_pomerance:
            return {}
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
                if cand2 >= 2 and not add_candidate(cand2, 'PomStd'):
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
                    if cand2 >= 2 and not add_candidate(cand2, 'PomExt'):
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

    def print_stats(self):
        print("\n" + "="*70)
        print("STATISTIQUES POMERANCE H2")
        print("="*70)
        print(f"  Total candidats générés : {self.stats['total_generated']:,}")
        print(f"  - Standard (PomStd)     : {self.stats['std']:,}")
        print(f"  - Étendu (PomExt)       : {self.stats['ext']:,}")
        print(f"  - Puissance 2 (PomPow2) : {self.stats['pow2']:,}")
        print(f"  Filtrés par Robin       : {self.stats['filtered']:,}")
        print("="*70)

_improved_pomerance = ImprovedPomeranceH2()

# ============================================================================
# CACHE GLOBAL UNIFIÉ
# ============================================================================
class GlobalAntecedenteCache:
    def __init__(self, cache_dir=".", use_compression=False):
        self.cache_dir = cache_dir
        self.use_compression = use_compression
        self.cache_file = os.path.join(cache_dir, "antecedents_global_cache.json.gz" if use_compression else "antecedents_global_cache.json")
        self.incremental_file = os.path.join(cache_dir, "antecedents_incremental.jsonl")
        self.cache = {}
        self.stats = {'total_entries': 0, 'cache_hits': 0, 'cache_misses': 0, 'new_entries': 0}
        self._load_cache()

    def _load_cache(self):
        print(f"[Cache Global] Chargement depuis {self.cache_file}...")
        if os.path.exists(self.cache_file):
            try:
                opener = gzip.open if self.use_compression else open
                mode = 'rt' if self.use_compression else 'r'
                with opener(self.cache_file, mode, encoding='utf-8') as f:
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
        self.stats['cache_misses'] += 1
        return None

    def add_antecedents(self, aliquot, antecedents_dict):
        aliquot_str = str(aliquot)
        if aliquot_str not in self.cache:
            self.cache[aliquot_str] = {}
            self.stats['new_entries'] += 1
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
            cache_to_save = {k: {str(kk): vv for kk, vv in v.items()} for k, v in self.cache.items()}
            opener = gzip.open if self.use_compression else open
            mode = 'wt' if self.use_compression else 'w'
            with opener(self.cache_file, mode, encoding='utf-8') as f:
                json.dump(cache_to_save, f, separators=(',', ':') if self.use_compression else None, indent=None if self.use_compression else 2)
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
        stats = {**self.stats, 'total_aliquots': len(self.cache), 'total_antecedents': total_antecedents}
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            stats['hit_rate'] = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) * 100
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
        pom_stats = _improved_pomerance.stats
        if sum(pom_stats.values()) > 0:
            print(f"\nHeuristique Pomerance H2 :")
            print(f"  • Candidats standard  : {pom_stats['std']}")
            print(f"  • Candidats étendus   : {pom_stats['ext']}")
            print(f"  • Candidats power2    : {pom_stats['pow2']}")
            print(f"  • Filtrés (Robin)     : {pom_stats['filtered']}")
        if NUMBA_AVAILABLE:
            total_sigma = _numba_stats['sigma_numba'] + _numba_stats['sigma_gmpy2']
            if total_sigma > 0:
                pct = _numba_stats['sigma_numba'] / total_sigma * 100
                print(f"\nNumba σ(n): {_numba_stats['sigma_numba']:,} ({pct:.1f}%)")
        print(f"{'='*70}\n")
        self._report_printed = True

_stats = PerformanceStats()
_numba_stats = {'sigma_numba': 0, 'sigma_gmpy2': 0, 'factorize_numba': 0, 'factorize_gmpy2': 0}
_sigma_cache = OrderedDict()
_divisors_cache = OrderedDict()
_SMALL_PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)

# ============================================================================
# WARMUP NUMBA
# ============================================================================
if NUMBA_AVAILABLE:
    _warmup_start = time.time()
    _ = _mulmod(123456789, 987654321, 1000000007)
    _ = _powmod(2, 100, 1000000007)
    _ = sigma_numba(12345)
    _ = sigma_pollard_numba(12345678901)
    _ = is_prime_numba(104729)
    _ = pollard_rho_numba(1000003 * 1000033)
    _warmup_elapsed = time.time() - _warmup_start
    print(f"[Numba] Warmup JIT terminé en {_warmup_elapsed:.1f}s")


# ============================================================================
# MOTEUR ARITHMÉTIQUE
# ============================================================================

def _sigma_from_factors(factors):
    """σ(n) depuis la factorisation : produit de (p^(e+1)-1)/(p-1)."""
    result = 1
    for p, e in factors.items():
        if e == 1:
            result *= (1 + p)
        else:
            result *= (p ** (e + 1) - 1) // (p - 1)
    return result

def sigma_optimized(n):
    """
    σ(n) avec dispatch par taille:
    - n < FACTORIZE_THRESHOLD (10^7): sigma_numba (trial division)
    - n ≥ FACTORIZE_THRESHOLD:        sigma_pollard_numba (Pollard-ρ intégré)
    - sans Numba:                      trial division gmpy2
    """
    if n < 2:
        return mpz(1) if n == 1 else mpz(0)
    n_int = int(n)

    if n_int in _sigma_cache:
        _sigma_cache.move_to_end(n_int)
        return _sigma_cache[n_int]

    if NUMBA_AVAILABLE:
        if n_int < FACTORIZE_THRESHOLD:
            result = mpz(sigma_numba(n_int))
        else:
            result = mpz(sigma_pollard_numba(n_int))
        _numba_stats['sigma_numba'] += 1
    else:
        _numba_stats['sigma_gmpy2'] += 1
        result = _sigma_trial_division_gmpy2(n_int)

    _sigma_cache[n_int] = result
    if len(_sigma_cache) > SIGMA_CACHE_SIZE:
        _sigma_cache.popitem(last=False)
    return result

def _sigma_trial_division_gmpy2(n_int):
    """Trial division gmpy2 — fallback sans Numba."""
    n = mpz(n_int)
    total = mpz(1)
    temp_n = n
    tz = gmpy2.bit_scan1(temp_n)
    if tz:
        total = SIGMA_POW2[tz] if tz <= 32 else (mpz(1) << (tz + 1)) - 1
        temp_n >>= tz
    for p in _SMALL_PRIMES:
        if p * p > temp_n:
            break
        if temp_n % p == 0:
            p_mpz, p_pow = mpz(p), mpz(p)
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
                p_pow, p_sum = d, mpz(1) + d
                temp_n //= d
                while temp_n % d == 0:
                    p_pow *= d
                    p_sum += p_pow
                    temp_n //= d
                total *= p_sum
            d += 2
        if temp_n > 1:
            total *= (mpz(1) + temp_n)
    return total


def factorize_fast(n):
    n_int = int(n)
    if NUMBA_AVAILABLE and n_int < NUMBA_THRESHOLD:
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
                    return factors
        if temp_n == 1:
            return factors
        if is_prime_numba(temp_n):
            factors[temp_n] = 1
            return factors
        factor = pollard_rho_numba(temp_n)
        if factor and factor > 1 and factor != temp_n:
            for p, e in factorize_fast(factor).items():
                factors[p] = factors.get(p, 0) + e
            for p, e in factorize_fast(temp_n // factor).items():
                factors[p] = factors.get(p, 0) + e
        else:
            d = 547
            while d * d <= temp_n:
                if temp_n % d == 0:
                    exp = 0
                    while temp_n % d == 0:
                        exp += 1
                        temp_n //= d
                    factors[d] = factors.get(d, 0) + exp
                    if temp_n == 1:
                        return factors
                    if is_prime_numba(temp_n):
                        factors[temp_n] = factors.get(temp_n, 0) + 1
                        return factors
                d += 2
            if temp_n > 1:
                factors[temp_n] = factors.get(temp_n, 0) + 1
        return factors
    else:
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
        def pollard_brent(m):
            if m == 1 or gmpy2.is_prime(m):
                return m
            if m % 2 == 0:
                return 2
            for c_val in range(1, 50):
                c, y, g, r, q = mpz(c_val), mpz(2), mpz(1), 1, mpz(1)
                iterations = 0
                while g == 1 and iterations < 300000:
                    x = y
                    for _ in range(r):
                        y = (y * y + c) % m
                    k = 0
                    while k < r and g == 1:
                        for _ in range(min(128, r - k)):
                            y = (y * y + c) % m
                            q = (q * abs(x - y)) % m
                            iterations += 1
                        g = gmpy2.gcd(q, m)
                        k += 128
                    r *= 2
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
                factors[int(m)] = factors.get(int(m), 0) + 1
                return
            decompose(f)
            decompose(m // f)
        if temp_n > 1:
            decompose(temp_n)
        return factors

def get_divisors_fast(n):
    n = int(n)
    if n in _divisors_cache:
        _divisors_cache.move_to_end(n)
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
    _divisors_cache[n] = divs
    if len(_divisors_cache) > DIVISORS_CACHE_SIZE:
        _divisors_cache.popitem(last=False)
    return divs

# ============================================================================
# GÉNÉRATION DES DRIVERS
# ============================================================================
def generate_drivers_optimized(n_cible, val_max_coche=None, smooth_bound=170, extra_primes=None, max_depth=6):
    print(f"[Drivers] Génération DFS (B={smooth_bound}, depth={max_depth})...")
    start = time.time()
    n_cible_int = int(n_cible)
    ref_value = max(n_cible_int, val_max_coche) if val_max_coche else n_cible_int
    harpon_limit = n_cible_int - 1
    expansion_limit = ref_value
    _SIGMA_POW2_LIST = [pow(2, m + 1) - 1 for m in range(33)]
    all_primes = sorted(set(list(primerange(3, smooth_bound + 1)) + (extra_primes or [])))
    print(f"[Drivers] {len(all_primes)} premiers (3 → {all_primes[-1]})")
    drivers_odd = {1: 1}
    def smooth_dfs(idx, prod, sigma_prod, depth):
        if depth >= max_depth:
            return
        for i in range(idx, len(all_primes)):
            p = all_primes[i]
            pp, sp = p, 1 + p
            while prod * pp <= harpon_limit:
                drivers_odd[prod * pp] = sigma_prod * sp
                smooth_dfs(i + 1, prod * pp, sigma_prod * sp, depth + 1)
                pp *= p
                sp += pp
    smooth_dfs(0, 1, 1, 0)
    print(f"[Drivers] {len(drivers_odd) - 1} drivers impairs générés")
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
    temp_list.sort()
    n_drivers = len(temp_list)
    flat_data = []
    for D, SD, sD in temp_list:
        flat_data.extend((D, SD, sD))
    elapsed = time.time() - start
    _stats.driver_generation_time = elapsed
    print(f"[Drivers] OK {n_drivers} drivers en {elapsed:.2f}s")
    return (flat_data, n_drivers)

def get_cached_drivers(n_cible, val_max_coche, smooth_bound, extra_primes, max_depth, use_compression=False):
    primes_sig = sum(extra_primes) if extra_primes else 0
    cache_name = f"drivers_v5_B{smooth_bound}_D{max_depth}_P{primes_sig}.cache" + (".gz" if use_compression else "")
    flat_data, n_drivers = None, 0
    if os.path.exists(cache_name):
        print(f"[Cache] Chargement depuis {cache_name}...")
        try:
            opener = gzip.open if use_compression else open
            with opener(cache_name, 'rb') as f:
                flat_data, n_drivers = pickle.load(f)
            print(f"[Cache] OK Chargé : {n_drivers} drivers")
            _stats.cache_hits += 1
        except Exception as e:
            print(f"[Cache] X Erreur : {e}")
            flat_data = None
            _stats.cache_misses += 1
    else:
        _stats.cache_misses += 1
    if flat_data is None:
        flat_data, n_drivers = generate_drivers_optimized(n_cible, val_max_coche, smooth_bound, extra_primes, max_depth)
        try:
            opener = gzip.open if use_compression else open
            with opener(cache_name, 'wb') as f:
                pickle.dump((flat_data, n_drivers), f)
            print(f"[Cache] OK Sauvegardé")
        except Exception as e:
            print(f"[Cache] X Erreur sauvegarde : {e}")
    shared = SharedArray('q', n_drivers * 3, lock=False)
    shared[:] = flat_data
    return (shared, n_drivers)

# ============================================================================
# WORKER DE RECHERCHE
# ============================================================================
_worker_drivers = None
_worker_n_drivers = 0

def _sieve_primes(limit):
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return tuple(i for i in range(2, limit + 1) if is_p[i])

_SEMI_DIRECT_PRIMES = _sieve_primes(1_000_000)
_SEMI_DIRECT_PRIMES_LIST = list(_SEMI_DIRECT_PRIMES)
_SEMI_DIRECT_P_MAX = _SEMI_DIRECT_PRIMES[-1]

def init_worker_with_drivers(drivers_tuple):
    global _worker_drivers, _worker_n_drivers
    _worker_drivers, _worker_n_drivers = drivers_tuple
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def worker_search(node):
    """Recherche d'antécédents pour un nœud donné."""
    global _worker_drivers, _worker_n_drivers
    node_int = int(node)
    solutions = {}
    drv = _worker_drivers
    n_drv = _worker_n_drivers

    # ---- Différences courantes ----
    COMMON_DIFFS = (12, 56, 4, 8, 24, 40, 6, 20, 28, 44, 52, 60, 68, 76, 84, 92, 120, 992)
    for diff in COMMON_DIFFS:
        if diff >= node_int:
            continue
        k_candidate = node_int - diff
        if k_candidate <= 1:
            continue
        sig_k = sigma_optimized(k_candidate)
        if int(sig_k) - k_candidate == node_int:
            solutions[k_candidate] = f"Diff({diff})"

    # ---- Primoriaux ----
    SMALL_PRIMORIALS = (2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870)
    for primorial in SMALL_PRIMORIALS:
        offset = 2 * primorial
        if offset >= node_int:
            break
        k_candidate = node_int - offset
        if k_candidate <= 1 or k_candidate in solutions:
            continue
        sig_k = sigma_optimized(k_candidate)
        if int(sig_k) - k_candidate == node_int:
            solutions[k_candidate] = f"Prim({primorial})"

    # ---- Pomerance H2 avec pré-filtrage arithmétique ----
    h2_candidates = _improved_pomerance.generate_candidates_h2(node_int)
    for k_candidate, source_type in h2_candidates.items():
        if k_candidate in solutions:
            continue

        sigma_needed = node_int + k_candidate

        # Filtre 1: borne inférieure — σ(k) >= k + 1
        if sigma_needed < k_candidate + 1:
            continue

        # Filtre 2: parité de σ(k)
        k_is_odd_sigma = gmpy2.is_square(mpz(k_candidate))
        if not k_is_odd_sigma and k_candidate % 2 == 0:
            k_is_odd_sigma = gmpy2.is_square(mpz(k_candidate >> 1))
        if k_is_odd_sigma:
            if sigma_needed % 2 == 0:
                continue
        else:
            if sigma_needed % 2 == 1:
                continue

        # Filtre 3: divisibilité par (2^(v+1) - 1) si k = 2^v * m, m impair, m > 1
        if k_candidate > 1 and k_candidate % 2 == 0:
            k_tmp = k_candidate
            v2 = 0
            while k_tmp % 2 == 0:
                k_tmp //= 2
                v2 += 1
            if k_tmp > 1:
                mersenne_factor = (1 << (v2 + 1)) - 1
                if sigma_needed % mersenne_factor != 0:
                    continue

        # Filtre 4: divisibilité par 4 si 3 || k
        if k_candidate % 3 == 0 and k_candidate % 9 != 0:
            if sigma_needed % 4 != 0:
                continue

        # Filtre 5: borne supérieure de Robin
        if k_candidate > 5040:
            try:
                loglog_k = math.log(math.log(k_candidate))
                upper_bound = k_candidate * (EXP_GAMMA * loglog_k + 1.0)
                if sigma_needed > upper_bound:
                    continue
            except (ValueError, ZeroDivisionError):
                pass

        sig_k = sigma_optimized(k_candidate)
        if int(sig_k) - k_candidate == node_int:
            solutions[k_candidate] = source_type

    _pretest_keys = set(solutions.keys())

    # ---- Drivers ----
    if drv is not None and n_drv > 0:
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

            # -- Direct: k = D * q --
            num_direct = node_int - SD_int
            if num_direct > 0 and num_direct % sD == 0:
                q_full = num_direct // sD
                if q_full > 1 and D_int % q_full != 0:
                    if gmpy2.is_prime(q_full):
                        k = D_int * q_full
                        if k not in _pretest_keys:
                            solutions[k] = f"D({D_int})"
                    elif q_full < 1_000_000 and math.gcd(D_int, q_full) == 1:
                        k = D_int * q_full
                        if k not in _pretest_keys and int(sigma_optimized(k)) - k == node_int:
                            solutions[k] = f"Multi({D_int})"

            # -- Semi-direct: k = D * p * q --
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
                    k_semi = D_int * p * q_v
                    if k_semi not in _pretest_keys:
                        solutions[k_semi] = f"S({D_int})"

            # -- Quadratic: hybride diviseurs + fallback linéaire --
            if target_q <= MAX_TARGET_QUADRATIC:
                sqrt_target_tmp = gmpy2.isqrt(target_q)
                if sqrt_target_tmp // sD > QUADRATIC_MAX_ITERATIONS:
                    continue
                if node_int % 2 == 1 and gmpy2.is_prime(target_q):
                    continue

                divisors = get_divisors_fast(target_q)

                if divisors:
                    # Méthode rapide : itérer sur les vrais diviseurs
                    for d in divisors:
                        if d * d > target_q:
                            break
                        diff = d - SD_int
                        if diff <= 0 or diff % sD != 0:
                            continue
                        p_v = diff // sD
                        if p_v <= 1 or D_int % p_v == 0:
                            continue
                        if not gmpy2.is_prime(p_v):
                            continue
                        div_q = target_q // d
                        diff_q = div_q - SD_int
                        if diff_q <= 0 or diff_q % sD != 0:
                            continue
                        q_v = diff_q // sD
                        if q_v > p_v and D_int % q_v != 0 and gmpy2.is_prime(q_v):
                            k_quad = D_int * p_v * q_v
                            if k_quad not in _pretest_keys:
                                solutions[k_quad] = f"Q({D_int})"
                else:
                    # Fallback : balayage linéaire
                    if p_max_needed <= _SEMI_DIRECT_P_MAX:
                        div_min = (_SEMI_DIRECT_P_MAX + 1) * sD + SD_int
                    else:
                        div_min = 2 * sD + SD_int
                    d = div_min
                    while d <= sqrt_target:
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
                                                    solutions[k_quad] = f"Q({D_int})"
                        d += sD

    return (node_int, solutions)


# ============================================================================
# CLASSE PRINCIPALE
# ============================================================================
class ArbreAliquoteV5:
    def __init__(self, n_cible, profondeur=100, smooth_bound=120, extra_primes=None,
                 max_depth=6, use_compression=False, allow_empty_exploration=False):
        self.cible_initiale = int(n_cible)
        self.profondeur = profondeur
        self.smooth_bound = smooth_bound
        self.extra_primes = extra_primes or []
        self.max_depth = max_depth
        self.use_compression = use_compression
        self.allow_empty_exploration = allow_empty_exploration
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
            print(f"[Migration] Fusion de {self.old_cache_file}...")
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
        if self.allow_empty_exploration:
            empty_nodes_count = 0
            for aliquot_str, antecedents in self.global_cache.cache.items():
                if antecedents:
                    self.explored.add(int(aliquot_str))
                else:
                    empty_nodes_count += 1
            if self.explored:
                print(f"[Cache Global] {len(self.explored)} nœuds avec antécédents")
            if empty_nodes_count:
                print(f"[Cache Global] {empty_nodes_count} nœuds vides pour ré-exploration")
        else:
            for aliquot_str in self.global_cache.cache.keys():
                self.explored.add(int(aliquot_str))
            if self.explored:
                print(f"[Cache Global] {len(self.explored)} nœuds déjà explorés")

    def _signal_handler(self, sig, frame):
        print("\n[!] Interruption...")
        self.stop = True
        self.global_cache.save()
        self.global_cache.print_stats()
        _stats.report(force=True)
        self.afficher()
        sys.exit(0)

    def _save_node(self, node_val, solutions):
        self.global_cache.add_antecedents(node_val, solutions or {})

    def afficher(self):
        """Affiche un résumé de l'arbre construit."""
        raw_cache = self.global_cache.cache or {}
        total_nodes = len(raw_cache)
        total_antecedents = sum(len(v) for v in raw_cache.values())
        print(f"\n{'='*70}")
        print(f"RÉSUMÉ ARBRE - CIBLE {self.cible_initiale}")
        print(f"{'='*70}")
        print(f"Nœuds explorés     : {total_nodes:,}")
        print(f"Antécédents totaux : {total_antecedents:,}")
        print(f"Temps écoulé       : {time.time() - self.start_time:.1f}s")
        print(f"{'='*70}\n")

    def construire(self, reprise_active=False):
        print(f"\n{'='*70}")
        print(f"CONSTRUCTION ARBRE V5 - CIBLE {self.cible_initiale}")
        print(f"{'='*70}")
        print(f"Profondeur : {self.profondeur}, Smooth bound : {self.smooth_bound}")
        print(f"Max depth : {self.max_depth}, Compression : {'OUI' if self.use_compression else 'NON'}")
        print(f"Processeurs : {cpu_count()}, Reprise : {'OUI' if reprise_active else 'NON'}")
        print(f"{'='*70}\n")

        drivers_array, n_drivers = get_cached_drivers(
            self.cible_initiale, self.val_max_coche, self.smooth_bound,
            self.extra_primes, self.max_depth, self.use_compression
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
        pool = Pool(num_workers, initializer=init_worker_with_drivers, initargs=((drivers_array, n_drivers),))

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
                    limit_100x = self.val_max_coche * 100
                    to_compute = []
                    beyond_limit_count = 0
                    for v in current_gen:
                        if v in self.explored:
                            continue
                        if v > limit_100x:
                            self._save_node(v, {})
                            self.explored.add(v)
                            beyond_limit_count += 1
                        else:
                            to_compute.append(v)
                    if beyond_limit_count > 0:
                        print(f"[G{gen}] {beyond_limit_count} nœuds > 100x cible marqués feuilles")

                    if not to_compute:
                        print(f"[G{gen}] Tous explorés, navigation cache...")
                        next_gen_from_cache = []
                        empty_nodes = []
                        for v in current_gen:
                            ants = self.global_cache.get_antecedents(v)
                            if ants is not None:
                                if ants:
                                    next_gen_from_cache.extend([int(k) for k in ants.keys() if int(k) <= self.val_max_coche * 100])
                                elif self.allow_empty_exploration:
                                    empty_nodes.append(v)
                        if empty_nodes and self.allow_empty_exploration:
                            print(f"[G{gen}] {len(empty_nodes)} nœuds vides pour ré-exploration")
                            for node in empty_nodes:
                                self.explored.discard(node)
                            next_gen_from_cache.extend(empty_nodes)
                        if not next_gen_from_cache:
                            print(" -> Fin de branche (feuilles).")
                            break
                        current_gen = list(set(next_gen_from_cache))
                        continue

                next_gen = set()
                processed = 0
                for node_val, solutions in pool.imap_unordered(worker_search, to_compute, chunksize=1):
                    if self.stop:
                        break
                    self._save_node(node_val, solutions)
                    self.explored.add(node_val)
                    _stats.total_nodes_processed += 1
                    for sol_type in solutions.values():
                        sol_prefix = sol_type.split('(')[0]
                        _stats.add_solution(sol_prefix)
                    for k in solutions:
                        k_val = int(k)
                        next_gen.add(k_val)
                    processed += 1
                    if processed % 10 == 0:
                        print(f"\r   -> {processed}/{len(to_compute)}", end='')

                gen_time = time.time() - gen_start
                _stats.add_generation_time(gen_time)
                print(f"\n[G{gen}] OK {len(next_gen)} nouvelles branches ({gen_time:.2f}s)")

                next_gen_filtered = {k for k in next_gen if k <= self.val_max_coche * 100}
                if len(next_gen_filtered) < len(next_gen):
                    print(f"[G{gen}] Filtre 100x: {len(next_gen_filtered)}/{len(next_gen)} nœuds gardés")

                current_gen = list(next_gen_filtered)
                if reprise_active and gen == start_gen:
                    reprise_active = False

                self.global_cache.save()
        finally:
            pool.close()
            pool.join()
            _stats.report()
            self.afficher()

    def _resume_from_cache(self):
        raw_cache = self.global_cache.cache or {}
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
        root = self.cible_initiale

        if root not in full_cache:
            print(f"[Reprise] Cible {root} non trouvée dans le cache, démarrage normal")
            return ([root], 1)

        queue = deque([(root, 0)])
        tree_nodes = {root}
        cache = {}
        max_depth_found = 0

        print(f"[Reprise] Extraction de l'arbre depuis la racine {root}...")

        while queue:
            node, d = queue.popleft()
            max_depth_found = max(max_depth_found, d)
            if node in full_cache:
                cache[node] = full_cache[node]
                for child in full_cache[node]:
                    if child not in tree_nodes:
                        tree_nodes.add(child)
                        queue.append((child, d + 1))

        print(f"[Reprise] Arbre de {root} : {len(cache)} nœuds explorés")
        print(f"[Reprise] Nœuds de cet arbre : {len(tree_nodes)}")

        for parent in cache.keys():
            self.explored.add(parent)
        print(f"[Reprise] {len(self.explored)} nœuds marqués comme explorés")

        all_children = set()
        for enfants in cache.values():
            for e in enfants:
                all_children.add(e)

        frontier = {e for e in all_children if e not in cache}
        print(f"[Reprise] Frontière brute identifiée : {len(frontier)} nœuds")

        if not frontier:
            print("[Reprise] Frontière vide (arbre complet ?)")
            return ([], 1)

        frontier_filtered = [n for n in frontier if n <= self.val_max_coche * 100]
        print(f"[Reprise] Frontière filtrée : {len(frontier_filtered)} nœuds")

        estimated_gen = max_depth_found + 1
        print(f"[Reprise] Profondeur max trouvée : {max_depth_found}")
        print(f"[Reprise] Reprise à la génération : {estimated_gen}")

        return (frontier_filtered, estimated_gen)


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arbre aliquote V5")
    parser.add_argument("cible", type=int, help="Nombre cible")
    parser.add_argument("-p", "--profondeur", type=int, default=100, help="Profondeur max")
    parser.add_argument("-B", "--smooth-bound", type=int, default=120, help="Borne smooth")
    parser.add_argument("-D", "--max-depth", type=int, default=6, help="Profondeur DFS drivers")
    parser.add_argument("--extra-primes", type=int, nargs="*", default=[], help="Premiers supplémentaires")
    parser.add_argument("--compress", action="store_true", help="Compression gzip")
    parser.add_argument("--reprise", action="store_true", help="Reprendre depuis le cache")
    parser.add_argument("--allow-empty", action="store_true", help="Ré-explorer les nœuds vides")

    args = parser.parse_args()

    arbre = ArbreAliquoteV5(
        n_cible=args.cible,
        profondeur=args.profondeur,
        smooth_bound=args.smooth_bound,
        extra_primes=args.extra_primes if args.extra_primes else None,
        max_depth=args.max_depth,
        use_compression=args.compress,
        allow_empty_exploration=args.allow_empty,
    )
    arbre.construire(reprise_active=args.reprise)