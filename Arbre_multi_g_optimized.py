#!/usr/bin/env python3
"""
Recherche d'antécédents aliquotes - Version Ultra-Optimisée
============================================================
Optimisations majeures:
- Sauvegarde batch (pas à chaque nœud)
- Drivers initialisés une seule fois par worker
- Cache LRU local optimisé
- Pollard-Rho Brent optimisé
- Filtrage précoce agressif
- imap_unordered pour meilleur parallélisme
"""

import gmpy2
from gmpy2 import mpz
import sys
import time
import signal
import argparse
import json
import os
from multiprocessing import Pool, cpu_count
from sympy import primerange
from itertools import combinations

# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================

SIGMA_CACHE_SIZE = 500000
DIVISORS_CACHE_SIZE = 100000
MAX_DIVISORS = 8000
MAX_TARGET_QUADRATIC = 10**14
MAX_POLLARD_ITERATIONS = 30000

# ============================================================================
# MOTEUR ARITHMÉTIQUE OPTIMISÉ
# ============================================================================

_sigma_cache = {}
_divisors_cache = {}

def sigma_optimized(n):
    """Calcul optimisé de sigma(n) avec cache local"""
    if n < 2:
        return mpz(1) if n == 1 else mpz(0)
    
    n_int = int(n)
    if n_int in _sigma_cache:
        return _sigma_cache[n_int]
    
    n = mpz(n)
    total = mpz(1)
    temp_n = n
    
    # Facteur 2 optimisé
    tz = gmpy2.bit_scan1(temp_n)
    if tz:
        total = (mpz(1) << (tz + 1)) - 1
        temp_n >>= tz
    
    # Petits premiers en dur
    small_primes = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
    
    for p in small_primes:
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
    
    # Grands facteurs
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
            d = gmpy2.next_prime(d)
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
    
    # Trial division étendue
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
        """Pollard-Rho avec optimisation de Brent"""
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
        p_pow = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * p_pow)
            p_pow *= p
        divs = new_divs
    
    divs.sort()
    
    if len(_divisors_cache) < DIVISORS_CACHE_SIZE:
        _divisors_cache[n] = divs
    
    return divs


# ============================================================================
# GÉNÉRATION DES DRIVERS
# ============================================================================

def generate_drivers_optimized(n_cible):
    """Génère des drivers optimisés avec pré-calcul de sigma"""
    print(f"[*] Génération des drivers optimisés...")
    start = time.time()
    
    n_cible = mpz(n_cible)
    limite_d = min(int(gmpy2.isqrt(n_cible * 20)), 10**7)
    
    drivers_set = {1}
    
    # Primoriaux
    p_prod = 1
    for p in primerange(3, 50):
        p_prod *= p
        if p_prod > limite_d:
            break
        drivers_set.add(p_prod)
    
    # Puissances de premiers
    for p in primerange(3, 200):
        for exp in range(1, 5):
            val = p ** exp
            if val > limite_d:
                break
            drivers_set.add(val)
    
    # Premiers individuels
    limite_premiers = min(int(gmpy2.isqrt(n_cible)) + 2000, 100000)
    for p in primerange(3, limite_premiers):
        if p < limite_d:
            drivers_set.add(p)
    
    # Combinaisons
    petits = list(primerange(3, 50))
    puissances = [p**2 for p in primerange(3, 30)] + [p**3 for p in primerange(3, 15)]
    composants = sorted(set(petits + puissances))[:40]
    
    for r in [2, 3]:
        for combo in combinations(composants, r):
            prod = 1
            for x in combo:
                prod *= x
                if prod > limite_d:
                    break
            if prod <= limite_d:
                drivers_set.add(prod)
    
    drivers_list = sorted(drivers_set)
    print(f"[*] Pré-calcul de sigma pour {len(drivers_list)} drivers...")
    
    drivers_with_sigma = []
    for d in drivers_list:
        sd = sigma_optimized(d)
        drivers_with_sigma.append((d, int(sd)))
    
    elapsed = time.time() - start
    print(f"[*] {len(drivers_with_sigma)} drivers générés en {elapsed:.2f}s")
    
    return drivers_with_sigma


# ============================================================================
# WORKER OPTIMISÉ
# ============================================================================

_worker_drivers = None

def init_worker_with_drivers(drivers):
    global _worker_drivers
    _worker_drivers = drivers
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def worker_search(node):
    """Worker optimisé pour la recherche d'antécédents"""
    global _worker_drivers
    
    node = mpz(node)
    solutions = {}
    
    max_D = node
    max_D_div16 = max_D // 16
    
    # Pré-calcul des puissances de 2
    pow2 = [1 << m for m in range(21)]
    pow2_minus1 = [(1 << (m + 1)) - 1 for m in range(21)]
    
    for d, sd_int in _worker_drivers:
        if d > max_D_div16:
            break
        
        sd = mpz(sd_int)
        
        for m in range(20):
            D = pow2[m] * d
            if D > max_D:
                break
            
            SD = pow2_minus1[m] * sd
            
            # CAS DIRECT
            den_direct = SD - D
            if den_direct > 0:
                num_direct = node - SD
                if num_direct > 0:
                    q_cand, rem = divmod(num_direct, den_direct)
                    if rem == 0 and q_cand > 1:
                        if D % q_cand != 0 and gmpy2.is_prime(q_cand):
                            k = D * q_cand
                            if sigma_optimized(k) - k == node:
                                solutions[int(k)] = f"D({D})"
            
            # CAS QUADRATIQUE
            A = 2 * D - SD
            if A == 0:
                continue
            
            target = A * (node + SD) + SD * SD
            
            if target <= 0 or target > MAX_TARGET_QUADRATIC:
                continue
            
            target_int = int(target)
            divs = get_divisors_fast(target_int)
            
            if not divs:
                continue
            
            A_int = int(A)
            SD_int = int(SD)
            D_int = int(D)
            
            sqrt_target = int(gmpy2.isqrt(target_int))
            
            for div in divs:
                if div > sqrt_target:
                    break
                
                diff = div - SD_int
                if diff % A_int != 0:
                    continue
                
                p_v = diff // A_int
                if p_v <= 1 or D_int % p_v == 0:
                    continue
                
                if not gmpy2.is_prime(p_v):
                    continue
                
                div_q = target_int // div
                diff_q = div_q - SD_int
                
                if diff_q % A_int != 0:
                    continue
                
                q_v = diff_q // A_int
                if q_v <= p_v or D_int % q_v == 0:
                    continue
                
                if not gmpy2.is_prime(q_v):
                    continue
                
                k = D_int * p_v * q_v
                if sigma_optimized(k) - k == node:
                    solutions[int(k)] = f"Q({D_int})"
            
            # ========================================
            # CAS CUBIQUE (k = D * p * q * r)
            # ========================================
            # Formule: s(k) = sigma(D)*(1+p)(1+q)(1+r) - D*p*q*r = node
            # Pour p fixé: q = (node - A*(1+r)) / (A + C*r)
            # où A = sigma(D)*(1+p), C = A - D*p
            
            # Limiter aux petits D pour éviter explosion combinatoire
            if D > 100000:
                continue
            
            node_int = int(node)
            
            # Liste étendue de petits premiers pour p et r
            SMALL_PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499)
            
            # Itérer sur p (premier)
            for p in SMALL_PRIMES:
                if D_int % p == 0:
                    continue
                
                A = SD_int * (1 + p)
                C = A - D_int * p
                
                # Itérer sur r (autre premier)
                for r in SMALL_PRIMES:
                    if r == p or D_int % r == 0:
                        continue
                    
                    # q = (node - A*(1+r)) / (A + C*r)
                    den = A + C * r
                    if den <= 0:
                        continue
                    
                    num = node_int - A * (1 + r)
                    if num <= 0:
                        continue
                    
                    if num % den != 0:
                        continue
                    
                    q = num // den
                    
                    # Vérifications sur q
                    if q <= 1 or q == p or q == r or D_int % q == 0:
                        continue
                    
                    if not gmpy2.is_prime(q):
                        continue
                    
                    # Vérification finale
                    k = D_int * p * q * r
                    if sigma_optimized(k) - k == node:
                        solutions[int(k)] = f"C({D_int})"
    
    return (int(node), solutions)


# ============================================================================
# CLASSE PRINCIPALE
# ============================================================================

class ArbreAliquoteUltra:
    def __init__(self, n_cible, val_max_coche, profondeur):
        self.cible_initiale = mpz(n_cible)
        self.val_max_coche = int(val_max_coche)
        self.profondeur = profondeur
        self.cache_file = "aliquote_cache.json"
        self.cache = self._load_cache()
        self.drivers = generate_drivers_optimized(self.cible_initiale)
        self.pending_saves = {}
        self.save_interval = 50
        self.stop = False
        signal.signal(signal.SIGINT, self._handle_stop)
    
    def _handle_stop(self, s, f):
        print("\n[!] Arrêt demandé, sauvegarde en cours...")
        self.stop = True
    
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    return {int(k): {int(ek): ev for ek, ev in v.items()} for k, v in data.items()}
            except Exception as e:
                print(f"[!] Erreur chargement cache: {e}")
                return {}
        return {}
    
    def _save_cache(self, force=False):
        if not self.pending_saves and not force:
            return
        
        self.cache.update(self.pending_saves)
        self.pending_saves = {}
        
        tmp_file = self.cache_file + ".tmp"
        with open(tmp_file, 'w') as f:
            serialized = {str(k): {str(ek): ev for ek, ev in v.items()} for k, v in self.cache.items()}
            json.dump(serialized, f)
        os.replace(tmp_file, self.cache_file)
    
    def _add_result(self, node, solutions):
        self.pending_saves[int(node)] = solutions
        
        if len(self.pending_saves) >= self.save_interval:
            self._save_cache()
    
    def construire(self):
        nodes = [int(self.cible_initiale)]
        num_workers = max(1, cpu_count() - 1)
        
        print(f"[*] Démarrage avec {num_workers} workers\n")
        
        try:
            for gen in range(1, self.profondeur + 1):
                if self.stop or not nodes:
                    break
                
                start_gen = time.time()
                
                to_compute = []
                next_gen = []
                
                for node in nodes:
                    if node in self.cache:
                        for k in self.cache[node].keys():
                            next_gen.append(k)
                    elif node in self.pending_saves:
                        for k in self.pending_saves[node].keys():
                            next_gen.append(k)
                    else:
                        to_compute.append(node)
                
                total_gen = len(nodes)
                current_node_str = "-"
                
                if to_compute:
                    with Pool(num_workers, initializer=init_worker_with_drivers, initargs=(self.drivers,)) as pool:
                        processed = 0
                        
                        chunksize = max(1, len(to_compute) // (num_workers * 4))
                        
                        for node, solutions in pool.imap_unordered(worker_search, to_compute, chunksize=chunksize):
                            if self.stop:
                                break
                            
                            self._add_result(node, solutions)
                            for k in solutions.keys():
                                next_gen.append(k)
                            
                            processed += 1
                            current_node_str = str(node)
                            
                            # Affichage sur une seule ligne
                            print(f"\rG{gen} | {processed}/{total_gen} | {current_node_str}", end='', flush=True)
                
                self._save_cache(force=True)
                
                nodes = list(set(next_gen))
                
                # Effacer la ligne et afficher le résumé final de la génération
                print(f"\rG{gen} | {total_gen}/{total_gen} | terminé → {len(nodes)} suivants")
        
        finally:
            self._save_cache(force=True)
        
        print()
        self.afficher()
    
    def afficher(self):
        print(f"\n{'='*70}")
        print(f"RÉSULTATS FINAUX (<= {self.val_max_coche})")
        print(f"{'='*70}")
        
        tous = {int(self.cible_initiale)}
        for parent, enfants in self.cache.items():
            tous.add(int(parent))
            for child in enfants:
                tous.add(int(child))
        
        resultats = sorted([n for n in tous if n <= self.val_max_coche])
        print(f"Total: {len(resultats)} nombres trouvés\n")
        
        for v in resultats:
            print(f"  [✓] {v}")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recherche d'antécédents aliquotes ultra-optimisée")
    parser.add_argument("val_coche", type=int, help="Valeur maximale pour l'affichage")
    parser.add_argument("n", type=int, help="Nombre cible")
    parser.add_argument("--depth", type=int, default=100, help="Profondeur maximale (défaut: 100)")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"    RECHERCHE D'ANTÉCÉDENTS ALIQUOTES - VERSION ULTRA")
    print(f"{'='*70}")
    print(f"Cible            : {args.n}")
    print(f"Affichage        : nombres <= {args.val_coche}")
    print(f"Profondeur max   : {args.depth}")
    print(f"CPUs disponibles : {cpu_count()}")
    print(f"Workers          : {max(1, cpu_count() - 1)}")
    print(f"{'='*70}")
    
    start_total = time.time()
    ArbreAliquoteUltra(args.n, args.val_coche, args.depth).construire()
    elapsed_total = time.time() - start_total
    
    print(f"\n{'='*70}")
    print(f"TEMPS TOTAL: {elapsed_total:.2f}s")
    print(f"{'='*70}\n")