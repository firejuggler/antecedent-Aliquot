# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
fast_loops.pyx V2 — Boucle principale drivers + Semi-direct compilée en C.
Fusionne : itération drivers, Direct, filtres, Semi-direct en un seul passage.

V2 CHANGEMENTS:
  - flat_data: 4 champs par driver (D, SD, sD, mask_pmin) au lieu de 3.
  - mask_pmin pré-calculé UNE SEULE FOIS à la génération des drivers.
    bits 0-23  = bitmask coprime (bit i=1 si D divisible par le (i+1)-ème premier 3..97)
    bits 24-27 = index p_min dans P_MIN_VALS (0..15 → 3..59)
  - Gain: 30-60% sur la boucle semi-direct (supprime les idiv D%p pour p≤97).
  - Fix overflow: target_q calculé en float64.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# ============================================================================
# Constantes
# ============================================================================
cdef int N_SMALL = 24  # nombre de premiers couverts par le bitmask (3..97)

cdef long long P_MIN_VALS[16]
P_MIN_VALS[0] = 3;  P_MIN_VALS[1] = 5;  P_MIN_VALS[2] = 7
P_MIN_VALS[3] = 11; P_MIN_VALS[4] = 13; P_MIN_VALS[5] = 17
P_MIN_VALS[6] = 19; P_MIN_VALS[7] = 23; P_MIN_VALS[8] = 29
P_MIN_VALS[9] = 31; P_MIN_VALS[10] = 37; P_MIN_VALS[11] = 41
P_MIN_VALS[12] = 43; P_MIN_VALS[13] = 47; P_MIN_VALS[14] = 53
P_MIN_VALS[15] = 59


# ============================================================================
# Fonctions utilitaires
# ============================================================================
cdef inline long long isqrt_c(long long n) noexcept nogil:
    """Integer square root via Newton (exact pour n < 2^62)."""
    cdef long long x, x1
    if n <= 1:
        return n
    x = <long long>sqrt(<double>n)
    while True:
        x1 = (x + n // x) >> 1
        if x1 >= x:
            break
        x = x1
    while x * x > n:
        x -= 1
    while (x + 1) * (x + 1) <= n:
        x += 1
    return x


def compute_mask_pmin(long long D):
    """
    Calcule le champ mask_pmin pour un driver D.
    Appelé UNE SEULE FOIS par driver lors de la génération (dans _expand_to_even_flat).
    Retourne un int64 emballant mask (bits 0-23) et p_min_idx (bits 24-27).
    """
    cdef unsigned int mask = 0
    cdef int p_min_idx = 15  # default: p_min=59

    if D % 3 == 0:  mask |= (1 << 0)
    else:
        if p_min_idx == 15: p_min_idx = 0
    if D % 5 == 0:  mask |= (1 << 1)
    else:
        if p_min_idx == 15: p_min_idx = 1
    if D % 7 == 0:  mask |= (1 << 2)
    else:
        if p_min_idx == 15: p_min_idx = 2
    if D % 11 == 0: mask |= (1 << 3)
    else:
        if p_min_idx == 15: p_min_idx = 3
    if D % 13 == 0: mask |= (1 << 4)
    else:
        if p_min_idx == 15: p_min_idx = 4
    if D % 17 == 0: mask |= (1 << 5)
    else:
        if p_min_idx == 15: p_min_idx = 5
    if D % 19 == 0: mask |= (1 << 6)
    else:
        if p_min_idx == 15: p_min_idx = 6
    if D % 23 == 0: mask |= (1 << 7)
    else:
        if p_min_idx == 15: p_min_idx = 7
    if D % 29 == 0: mask |= (1 << 8)
    else:
        if p_min_idx == 15: p_min_idx = 8
    if D % 31 == 0: mask |= (1 << 9)
    else:
        if p_min_idx == 15: p_min_idx = 9
    if D % 37 == 0: mask |= (1 << 10)
    else:
        if p_min_idx == 15: p_min_idx = 10
    if D % 41 == 0: mask |= (1 << 11)
    else:
        if p_min_idx == 15: p_min_idx = 11
    if D % 43 == 0: mask |= (1 << 12)
    else:
        if p_min_idx == 15: p_min_idx = 12
    if D % 47 == 0: mask |= (1 << 13)
    else:
        if p_min_idx == 15: p_min_idx = 13
    if D % 53 == 0: mask |= (1 << 14)
    else:
        if p_min_idx == 15: p_min_idx = 14
    # Les suivants contribuent au mask mais pas à p_min
    if D % 59 == 0: mask |= (1 << 15)
    if D % 61 == 0: mask |= (1 << 16)
    if D % 67 == 0: mask |= (1 << 17)
    if D % 71 == 0: mask |= (1 << 18)
    if D % 73 == 0: mask |= (1 << 19)
    if D % 79 == 0: mask |= (1 << 20)
    if D % 83 == 0: mask |= (1 << 21)
    if D % 89 == 0: mask |= (1 << 22)
    if D % 97 == 0: mask |= (1 << 23)

    return <long long>mask | (<long long>p_min_idx << 24)


# ============================================================================
# Boucle principale
# ============================================================================
def driver_loop_direct_semi(
    long long node,
    np.ndarray[np.int64_t, ndim=1] drv_flat,
    int drv_start,
    int drv_end,
    np.ndarray[np.int64_t, ndim=1] sieve,
    int sieve_len,
    long long semi_direct_p_max
):
    """
    Boucle principale fusionnee en C: Direct + filtres + Semi-direct.
    flat_data V2: 4 champs par driver (D, SD, sD, mask_pmin).
    Le bitmask est lu depuis mask_pmin (pré-calculé, coût runtime = 0).
    """
    cdef long long *drv = <long long *>drv_flat.data
    cdef long long *sv = <long long *>sieve.data

    cdef int idx, off, pi
    cdef long long D, SD, sD, mask_pmin_val, p_min, num_d, q_full, k_val
    cdef long long num_qmin, den_qmin, p_max
    cdef double target_q_f, sq_f
    cdef long long p_v, SD_1p, num_q, den_q, q_v, k_semi
    cdef unsigned int div_mask
    cdef int p_min_idx

    # Stats
    cdef long long st_drivers_tested = 0
    cdef long long st_filtered_bisect = 0
    cdef long long st_filtered_pmin = 0
    cdef long long st_filtered_qmax = 0
    cdef long long st_filtered_pmax = 0
    cdef long long st_entered_semi = 0

    direct_cands = []
    semi_cands = []

    # Bisect: trouver effective_end (D <= node)
    cdef int effective_end = drv_end
    cdef int lo, hi, mid

    if drv[(drv_end - 1) * 4] > node:
        lo = drv_start
        hi = drv_end - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if drv[mid * 4] <= node:
                lo = mid + 1
            else:
                hi = mid
        effective_end = lo
        st_filtered_bisect = drv_end - effective_end

    st_drivers_tested = drv_end - drv_start

    for idx in range(drv_start, effective_end):
        off = idx * 4
        D = drv[off]
        SD = drv[off + 1]
        sD = drv[off + 2]
        mask_pmin_val = drv[off + 3]

        if SD > node or sD <= 0:
            continue

        # ---- Direct: k = D * q, q = (node - SD) / sD ----
        num_d = node - SD
        if num_d > 0 and num_d % sD == 0:
            q_full = num_d // sD
            if q_full > 1 and D % q_full != 0:
                k_val = D * q_full
                direct_cands.append((k_val, D, q_full))

        # ---- Extraire p_min et bitmask depuis mask_pmin ----
        div_mask = <unsigned int>(mask_pmin_val & 0xFFFFFF)
        p_min_idx = <int>((mask_pmin_val >> 24) & 0xF)
        p_min = P_MIN_VALS[p_min_idx]

        # ---- Filtre p_min ----
        if SD * (1 + p_min) > node:
            st_filtered_pmin += 1
            continue

        # ---- Filtre q_max (pre-isqrt) ----
        target_q_f = <double>sD * <double>node + <double>SD * <double>D
        if target_q_f <= 0.0:
            continue

        num_qmin = node - SD * (1 + p_min)
        den_qmin = SD + p_min * sD
        if num_qmin <= p_min * den_qmin:
            st_filtered_qmax += 1
            continue

        # ---- sqrt(target_q) via float64 + p_max ----
        sq_f = sqrt(target_q_f)
        p_max = <long long>((sq_f - <double>SD) / <double>sD) + 1

        if p_max < p_min:
            st_filtered_pmax += 1
            continue

        # ---- Semi-direct ----
        if p_max > semi_direct_p_max:
            p_max = semi_direct_p_max

        st_entered_semi += 1

        # Boucle sur le crible (skip p=2)
        for pi in range(1, sieve_len):
            p_v = sv[pi]
            if p_v > p_max:
                break

            # Bitmask: pi=1..24 → bit (pi-1), coût = 1 cycle au lieu de 22
            if pi <= N_SMALL:
                if (div_mask >> (pi - 1)) & 1:
                    continue
            else:
                if D % p_v == 0:
                    continue

            SD_1p = SD * (1 + p_v)
            if SD_1p > node:
                break

            num_q = node - SD_1p
            den_q = SD + p_v * sD
            if num_q % den_q != 0:
                continue

            q_v = num_q // den_q
            if q_v <= p_v or D % q_v == 0:
                continue

            k_semi = D * p_v * q_v
            semi_cands.append((k_semi, D, p_v, q_v))

    stats = {
        'drivers_tested': st_drivers_tested,
        'filtered_bisect': st_filtered_bisect,
        'filtered_pmin': st_filtered_pmin,
        'filtered_qmax': st_filtered_qmax,
        'filtered_pmax_lt_pmin': st_filtered_pmax,
        'entered_semi_direct': st_entered_semi,
    }

    return direct_cands, semi_cands, stats


# ============================================================================
# Utilitaires
# ============================================================================
def sieve_primes(long long limit):
    """Crible d'Eratosthene retournant un tuple de premiers <= limit."""
    cdef long long i, j
    cdef bytearray is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = 0
    is_p[1] = 0
    i = 2
    while i * i <= limit:
        if is_p[i]:
            j = i * i
            while j <= limit:
                is_p[j] = 0
                j += i
        i += 1
    return tuple(i for i in range(2, limit + 1) if is_p[i])


def quadratic_scan(
    long long target_q, long long div_min, long long sqrt_tq,
    long long sD, long long SD, long long D
):
    """Scan lineaire pour trouver des paires (p,q) via diviseurs de target_q."""
    cdef long long d, diff, p_v, div_q, diff_q, q_v
    result = []
    d = div_min
    while d <= sqrt_tq:
        if target_q % d == 0:
            diff = d - SD
            if diff > 0 and diff % sD == 0:
                p_v = diff // sD
                if p_v > 1 and D % p_v != 0:
                    div_q = target_q // d
                    diff_q = div_q - SD
                    if diff_q > 0 and diff_q % sD == 0:
                        q_v = diff_q // sD
                        if q_v > p_v and D % q_v != 0:
                            result.append((p_v, q_v))
        d += sD
    return result


def quadratic_scan_fallback(
    long long target_q, long long div_min, long long sqrt_tq,
    long long sD, long long SD, long long D
):
    """Alias pour quadratic_scan."""
    return quadratic_scan(target_q, div_min, sqrt_tq, sD, SD, D)


def generate_odd_drivers_c(list all_primes, long long harpon_limit, int max_depth):
    """Generate odd drivers via DFS."""
    cdef dict drivers_odd = {1: 1}
    cdef list primes_c = all_primes
    cdef int n_primes = len(primes_c)

    def smooth_dfs(int idx, long long prod, long long sigma_prod, int depth):
        if depth >= max_depth:
            return
        cdef int i
        cdef long long p, pp, sp, new_prod, new_sigma
        for i in range(idx, n_primes):
            p = primes_c[i]
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


def get_divisors_fast_c(dict f_dict, int max_divisors):
    """Construire la liste triee des diviseurs."""
    cdef int num_divs = 1
    for exp in f_dict.values():
        num_divs *= (exp + 1)
        if num_divs > max_divisors:
            return []
    cdef list divs = [1]
    cdef long long p_power
    cdef list new_divs
    for p, e in f_dict.items():
        new_divs = []
        p_power = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * p_power)
            p_power *= p
        divs = new_divs
    divs.sort()
    return divs


def divisors_in_range(dict f_dict, long long lo, long long hi):
    """Diviseurs de n dans [lo, hi] via DFS pruning."""
    cdef list prime_list = sorted(f_dict.items())
    cdef list result = []
    _dfs_range(prime_list, 0, 1, lo, hi, result)
    result.sort()
    return result


cdef void _dfs_range(list prime_list, int idx, long long current,
                     long long lo, long long hi, list result):
    if current > hi:
        return
    if idx == len(prime_list):
        if current >= lo:
            result.append(current)
        return
    cdef long long p = prime_list[idx][0]
    cdef int max_e = prime_list[idx][1]
    cdef long long pe = 1
    cdef int e
    for e in range(max_e + 1):
        if current * pe > hi:
            break
        _dfs_range(prime_list, idx + 1, current * pe, lo, hi, result)
        pe *= p


def semi_direct_search(
    long long node, long long D, long long SD, long long sD,
    long long p_max_needed, tuple sieve_primes_t, set D_factors
):
    """Semi-direct search (legacy path)."""
    cdef long long p_v, SD_1p, num_q, den, q_v
    result = []
    for p_v in sieve_primes_t:
        if p_v > p_max_needed: break
        if p_v in D_factors or D % p_v == 0: continue
        SD_1p = SD * (1 + p_v)
        if SD_1p > node: break
        num_q = node - SD_1p
        den = SD + p_v * sD
        if num_q % den != 0: continue
        q_v = num_q // den
        if q_v <= p_v or D % q_v == 0: continue
        result.append((p_v, q_v))
    return result
