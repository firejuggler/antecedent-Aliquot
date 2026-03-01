# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
fast_loops.pyx V3 — Boucle principale drivers + Semi-direct compilée en C.
Fusionne : itération drivers, Direct, filtres, Semi-direct en un seul passage.

V2 CHANGEMENTS:
  - flat_data: 4 champs par driver (D, SD, sD, mask_pmin) au lieu de 3.
  - mask_pmin pré-calculé UNE SEULE FOIS à la génération des drivers.
    bits 0-23  = bitmask coprime (bit i=1 si D divisible par le (i+1)-ème premier 3..97)
    bits 24-27 = index p_min dans P_MIN_VALS (0..15 → 3..59)
  - Gain: 30-60% sur la boucle semi-direct (supprime les idiv D%p pour p≤97).
  - Fix overflow: target_q calculé en float64.

V3 CHANGEMENTS:
  - Ajout mulmod128/powmod128 via __uint128_t (GCC) pour arithmétique 64 bits sans overflow.
  - Nouveau is_prime_c : Miller-Rabin déterministe (12 témoins, valide pour n < 3.317×10^24).
  - cubic_loop: test de primalité de r_v remplacé par is_prime_c (supprime les faux positifs).
  - cubic_loop: ajout des filtres q_kill mod 11 et mod 13 (réduction ~10-15% de q testés).
  - cubic_loop: correction de la borne q_max (formule exacte via résolution quadratique),
    l'ancienne approximation sqrt(node/SDp)-1 manquait des candidats valides.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# ============================================================================
# Arithmétique modulaire 128 bits (évite l'overflow pour Miller-Rabin 64 bits)
# ============================================================================
cdef extern from *:
    """
    static inline unsigned long long _mulmod128(unsigned long long a,
                                                unsigned long long b,
                                                unsigned long long m) {
        return ((__uint128_t)a * b) % m;
    }
    static inline unsigned long long _powmod128(unsigned long long base,
                                                unsigned long long exp,
                                                unsigned long long mod) {
        unsigned long long result = 1;
        base %= mod;
        while (exp > 0) {
            if (exp & 1) result = (__uint128_t)result * base % mod;
            exp >>= 1;
            base = (__uint128_t)base * base % mod;
        }
        return result;
    }
    """
    unsigned long long _mulmod128(unsigned long long, unsigned long long, unsigned long long) noexcept nogil
    unsigned long long _powmod128(unsigned long long, unsigned long long, unsigned long long) noexcept nogil

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


cdef bint is_prime_c(long long n) noexcept nogil:
    """
    Test de primalité Miller-Rabin déterministe.
    Valide pour n < 3.317×10^24 (12 témoins : Sorenson & Webster 2016).
    Nombre de témoins adaptatif selon la taille de n.
    """
    if n < 2:
        return False
    if n == 2 or n == 3 or n == 5 or n == 7:
        return True
    if (n & 1) == 0 or n % 3 == 0 or n % 5 == 0:
        return False
    if n < 49:
        return True

    cdef unsigned long long un = <unsigned long long>n
    cdef unsigned long long d = un - 1
    cdef int r = 0
    while (d & 1) == 0:
        r += 1
        d >>= 1

    cdef unsigned long long witnesses[12]
    witnesses[0] = 2;  witnesses[1] = 3;  witnesses[2] = 5;  witnesses[3] = 7
    witnesses[4] = 11; witnesses[5] = 13; witnesses[6] = 17; witnesses[7] = 19
    witnesses[8] = 23; witnesses[9] = 29; witnesses[10] = 31; witnesses[11] = 37

    cdef int num_witnesses
    cdef unsigned long long L3215031751 = 3215031751ULL
    if un < 2047:
        num_witnesses = 1
    elif un < 1373653:
        num_witnesses = 2
    elif un < L3215031751:
        num_witnesses = 4
    else:
        num_witnesses = 12

    cdef unsigned long long a, x
    cdef int i, j
    cdef bint composite

    for i in range(num_witnesses):
        a = witnesses[i]
        if a >= un:
            continue
        x = _powmod128(a, d, un)
        if x == 1 or x == un - 1:
            continue
        composite = True
        for j in range(r - 1):
            x = _mulmod128(x, x, un)
            if x == un - 1:
                composite = False
                break
        if composite:
            return False
    return True


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

    Filtres V4 (fusion V2+V3):
      - skip_all: m|SD ET m|sD ET m∤node → skip driver (m=3,4,7,11,13)
      - p_kill: p ≡ pk (mod m) → m|den_q ET m∤num_q → skip p (m=3,7,11,13)
      - mod8: 8|den_q ET 8∤num_q → skip p (après p_kill)
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
    cdef int node_is_odd = node & 1

    # Variables filtres modulaires
    cdef int SD3, sD3, SD7, sD7, SD4, sD4, SD11, sD11, SD13, sD13
    cdef int skip_all_3, skip_all_7, skip_all_4, skip_all_11, skip_all_13
    cdef int node3, node7, node4, node11, node13
    cdef int q_kill3, q_kill3_active, q_kill7, q_kill7_active
    cdef int q_kill11, q_kill11_active, q_kill13, q_kill13_active
    cdef int inv3, inv7, inv11, inv13, nA3, nA7, nA11, nA13
    cdef int SD8, sD8, den_q8

    # Stats
    cdef long long st_drivers_tested = 0
    cdef long long st_filtered_bisect = 0
    cdef long long st_filtered_pmin = 0
    cdef long long st_filtered_qmax = 0
    cdef long long st_filtered_pmax = 0
    cdef long long st_filtered_skipall = 0
    cdef long long st_filtered_pkill = 0
    cdef long long st_filtered_mod8 = 0
    cdef long long st_entered_semi = 0

    direct_cands = []
    semi_cands = []

    # Pré-calcul node mod petits premiers
    node3 = <int>(node % 3)
    node7 = <int>(node % 7)
    node4 = <int>(node % 4)
    node11 = <int>(node % 11)
    node13 = <int>(node % 13)

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

        # ---- Node impair → skip semi-direct ----
        if node_is_odd:
            continue

        # ---- skip_all mod4: 4|SD ET 4|sD ET 4∤node → skip ----
        SD4 = <int>(SD % 4)
        sD4 = <int>(sD % 4)
        if SD4 == 0 and sD4 == 0 and node4 != 0:
            st_filtered_skipall += 1
            continue

        # ---- Filtre p_min (quasi gratuit: 1 multiplication) ----
        if SD * (1 + p_min) > node:
            st_filtered_pmin += 1
            continue

        # ---- skip_all: m|SD ET m|sD ET m∤node → skip driver ----
        SD3 = <int>(SD % 3)
        sD3 = <int>(sD % 3)
        skip_all_3 = (SD3 == 0 and sD3 == 0 and node3 != 0)

        SD7 = <int>(SD % 7)
        sD7 = <int>(sD % 7)
        skip_all_7 = (SD7 == 0 and sD7 == 0 and node7 != 0)

        SD11 = <int>(SD % 11)
        sD11 = <int>(sD % 11)
        skip_all_11 = (SD11 == 0 and sD11 == 0 and node11 != 0)

        SD13 = <int>(SD % 13)
        sD13 = <int>(sD % 13)
        skip_all_13 = (SD13 == 0 and sD13 == 0 and node13 != 0)

        if skip_all_3 or skip_all_7 or skip_all_11 or skip_all_13:
            st_filtered_skipall += 1
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

        # ---- p_kill: pré-calculer résidus tueurs mod 3, 7, 11, 13 ----
        nA3 = <int>((node - SD) % 3)
        q_kill3_active = 0
        q_kill3 = 0
        if sD % 3 != 0:
            inv3 = <int>(sD % 3)
            q_kill3 = (3 - (SD % 3 * inv3) % 3) % 3
            if (nA3 - SD % 3 * q_kill3 % 3 + 9) % 3 != 0:
                q_kill3_active = 1

        nA7 = <int>((node - SD) % 7)
        q_kill7_active = 0
        q_kill7 = 0
        if sD % 7 != 0:
            inv7 = [0,1,4,5,2,3,6][<int>(sD % 7)]
            q_kill7 = (7 - (SD % 7 * inv7) % 7) % 7
            if (nA7 - SD % 7 * q_kill7 % 7 + 49) % 7 != 0:
                q_kill7_active = 1

        nA11 = <int>((node - SD) % 11)
        q_kill11_active = 0
        q_kill11 = 0
        if sD % 11 != 0:
            inv11 = [0,1,6,4,3,9,2,8,7,5,10][<int>(sD % 11)]
            q_kill11 = (11 - (SD % 11 * inv11) % 11) % 11
            if (nA11 - SD % 11 * q_kill11 % 11 + 121) % 11 != 0:
                q_kill11_active = 1

        nA13 = <int>((node - SD) % 13)
        q_kill13_active = 0
        q_kill13 = 0
        if sD % 13 != 0:
            inv13 = [0,1,7,9,10,8,11,2,5,3,4,6,12][<int>(sD % 13)]
            q_kill13 = (13 - (SD % 13 * inv13) % 13) % 13
            if (nA13 - SD % 13 * q_kill13 % 13 + 169) % 13 != 0:
                q_kill13_active = 1

        # Pré-calcul mod8 pour le driver
        SD8 = <int>(SD % 8)
        sD8 = <int>(sD % 8)

        # Boucle sur le crible (skip p=2)
        for pi in range(1, sieve_len):
            p_v = sv[pi]
            if p_v > p_max:
                break

            # Bitmask: pi=1..24 → bit (pi-1)
            if pi <= N_SMALL:
                if (div_mask >> (pi - 1)) & 1:
                    continue
            else:
                if D % p_v == 0:
                    continue

            # p_kill: résidus modulaires → skip avant calcul
            if q_kill3_active and p_v % 3 == q_kill3:
                st_filtered_pkill += 1
                continue
            if q_kill7_active and p_v % 7 == q_kill7:
                st_filtered_pkill += 1
                continue
            if q_kill11_active and p_v % 11 == q_kill11:
                st_filtered_pkill += 1
                continue
            if q_kill13_active and p_v % 13 == q_kill13:
                st_filtered_pkill += 1
                continue

            SD_1p = SD * (1 + p_v)
            if SD_1p > node:
                break

            num_q = node - SD_1p
            den_q = SD + p_v * sD

            # mod8 filter: 8|den_q ET 8∤num_q → skip
            den_q8 = <int>((SD8 + (p_v & 7) * sD8) & 7)
            if den_q8 == 0 and (num_q & 7) != 0:
                st_filtered_mod8 += 1
                continue

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
        'filtered_skipall': st_filtered_skipall,
        'filtered_pmin': st_filtered_pmin,
        'filtered_qmax': st_filtered_qmax,
        'filtered_pmax_lt_pmin': st_filtered_pmax,
        'filtered_pkill': st_filtered_pkill,
        'filtered_mod8': st_filtered_mod8,
        'entered_semi_direct': st_entered_semi,
    }

    return direct_cands, semi_cands, stats


# ============================================================================
# Méthode Cubic: k = D * p * q * r  (p < q < r premiers, copremiers à D)
# ============================================================================
def cubic_loop(
    long long node,
    np.ndarray[np.int64_t, ndim=1] drv_flat,
    int drv_start,
    int drv_end,
    np.ndarray[np.int64_t, ndim=1] sieve,
    int sieve_len,
):
    """
    Cubic search: k = D * p * q * r  (p < q < r premiers, copremiers à D).

    Bornes:
      p < cbrt(node / SD)
      q > p,  q < sqrt(node / (SD * (1+p)))
      r = (node - SDp*(1+q)) / (SDp + q*sDp),  doit être > q et premier

    Filtres (V2):
      - node impair → 0 solutions (den_r toujours pair, num_r impair)
      - Bitmask de D réutilisé pour p et q copremiers à D
      - skip_all mod m: si m|SDp ET m|sDp ET m∤(node-SDp) → skip paire entière
        (den_r ≡ 0 mod m pour tout q, mais num_r ≢ 0 → jamais divisible)
      - per-q residue: q ≡ q_kill (mod m) → m|den_r ET m∤num_r → skip
      - Appliqué pour m = 3 et m = 7

    flat_data V2: 4 champs par driver (D, SD, sD, mask_pmin).
    """
    cdef long long *drv = <long long *>drv_flat.data
    cdef long long *sv = <long long *>sieve.data

    cdef int idx, off, pi_p, pi_q
    cdef long long D, SD, sD, mask_pmin_val
    cdef long long p_v, q_v, r_v, k_cubic
    cdef long long Dp, SDp, sDp
    cdef long long SD1q, num_r, den_r
    cdef double cbrt_val, q_max_f
    cdef long long p_max_cubic, q_max_cubic
    cdef unsigned int div_mask
    cdef int qi_lo, qi_hi, qi_mid

    # Variables pour les filtres modulaires pré-calculés
    cdef int SDp3, sDp3, nA3, SDp7, sDp7, nA7
    cdef int SDp5, sDp5, nA5, SDp11, sDp11, nA11, SDp13, sDp13, nA13
    cdef int skip_all_3, skip_all_7, skip_all_5, skip_all_11, skip_all_13
    cdef int skip_all_v2
    cdef int SDp4, sDp4
    cdef int q_kill3, q_kill3_active  # résidu q mod 3 à skip (-1 = inactif)
    cdef int q_kill7, q_kill7_active
    cdef int q_kill5, q_kill5_active
    cdef int q_kill11, q_kill11_active
    cdef int q_kill13, q_kill13_active
    cdef int inv3, inv7, inv5, nat3, nat7, nat5
    cdef int inv11, inv13, nat11, nat13

    # Filtre v2: au niveau PAIRE — si v2(den_r) > v2(node) pour tout q impair → skip paire
    cdef long long v2_mask
    cdef long long node_tmp = node
    cdef int v2_node = 0

    # Stats
    cdef long long st_drivers = 0
    cdef long long st_p_tested = 0
    cdef long long st_pairs_survived = 0
    cdef long long st_q_tested = 0
    cdef long long st_skip_pair = 0
    cdef long long st_skip_mod = 0
    cdef long long st_skip_v2 = 0
    cdef long long st_skip_idiv = 0
    cdef long long st_hits = 0

    cubic_cands = []

    # Node impair → aucun hit cubic possible
    if node & 1:
        stats = {
            'drivers': 0, 'p_tested': 0, 'pairs_survived': 0,
            'q_tested': 0, 'skip_pair': 0, 'skip_mod': 0, 'skip_idiv': 0, 'hits': 0,
        }
        return cubic_cands, stats

    # Pré-calcul v2_mask: (den_r & v2_mask) == 0 → v2(den_r) > v2(node) → skip
    # Car v2(num_r) = v2(node) toujours (node pair, SDp*(1+q) ≡ 0 mod 4*node_v2)
    node_tmp = node
    v2_node = 0
    while (node_tmp & 1) == 0:
        v2_node += 1
        node_tmp >>= 1
    v2_mask = (1LL << (v2_node + 1)) - 1  # ex: v2=1 → mask=3, v2=2 → mask=7

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

    for idx in range(drv_start, effective_end):
        off = idx * 4
        D = drv[off]
        SD = drv[off + 1]
        sD = drv[off + 2]
        mask_pmin_val = drv[off + 3]

        if SD > node or sD <= 0:
            continue

        st_drivers += 1

        # Extraire bitmask
        div_mask = <unsigned int>(mask_pmin_val & 0xFFFFFF)

        # Borne cubique: p < cbrt(node / SD)
        cbrt_val = (<double>node / <double>SD)
        if cbrt_val <= 27.0:
            continue
        cbrt_val = cbrt_val ** (1.0 / 3.0)
        p_max_cubic = <long long>cbrt_val
        if p_max_cubic < 3:
            continue

        # Boucle sur p
        for pi_p in range(1, sieve_len):
            p_v = sv[pi_p]
            if p_v > p_max_cubic:
                break

            # p copremier à D
            if pi_p <= N_SMALL:
                if (div_mask >> (pi_p - 1)) & 1:
                    continue
            else:
                if D % p_v == 0:
                    continue

            st_p_tested += 1

            Dp = D * p_v
            if Dp > node:
                break

            SDp = SD * (1 + p_v)
            sDp = SDp - Dp

            if SDp > node or sDp <= 0:
                continue

            # Borne exacte sur q: résolution de SDp*(1+2q)+q²*sDp ≤ node
            # (condition r ≥ q), soit q_max = (-SDp + √(SDp·Dp + sDp·node)) / sDp
            q_max_f = (-<double>SDp + sqrt(<double>SDp * <double>Dp + <double>sDp * <double>node)) / <double>sDp
            q_max_cubic = <long long>q_max_f
            if q_max_cubic <= p_v:
                continue

            # ============================================================
            # FILTRES MODULAIRES PRÉ-CALCULÉS pour cette paire (D,p)
            # skip_all: m|SDp ET m|sDp ET m∤(node-SDp) → skip paire
            # q_kill: q ≡ qk (mod m) → m|den_r ET m∤num_r → skip q
            # ============================================================

            # --- mod 3 ---
            SDp3 = <int>(SDp % 3)
            sDp3 = <int>(sDp % 3)
            nA3 = <int>((node - SDp) % 3)
            skip_all_3 = 0
            q_kill3_active = 0
            q_kill3 = 0

            if sDp3 == 0:
                if SDp3 == 0 and nA3 != 0:
                    skip_all_3 = 1
            else:
                inv3 = sDp3
                q_kill3 = (3 - (SDp3 * inv3) % 3) % 3
                nat3 = (nA3 - SDp3 * q_kill3 % 3 + 9) % 3
                if nat3 != 0:
                    q_kill3_active = 1

            # --- mod 5 ---
            SDp5 = <int>(SDp % 5)
            sDp5 = <int>(sDp % 5)
            nA5 = <int>((node - SDp) % 5)
            skip_all_5 = 0
            q_kill5_active = 0
            q_kill5 = 0

            if sDp5 == 0:
                if SDp5 == 0 and nA5 != 0:
                    skip_all_5 = 1
            else:
                # inv(1,5)=1, inv(2,5)=3, inv(3,5)=2, inv(4,5)=4
                if sDp5 == 1: inv5 = 1
                elif sDp5 == 2: inv5 = 3
                elif sDp5 == 3: inv5 = 2
                else: inv5 = 4
                q_kill5 = (5 - (SDp5 * inv5) % 5) % 5
                nat5 = (nA5 - SDp5 * q_kill5 % 5 + 25) % 5
                if nat5 != 0:
                    q_kill5_active = 1

            # --- mod 7 ---
            SDp7 = <int>(SDp % 7)
            sDp7 = <int>(sDp % 7)
            nA7 = <int>((node - SDp) % 7)
            skip_all_7 = 0
            q_kill7_active = 0
            q_kill7 = 0

            if sDp7 == 0:
                if SDp7 == 0 and nA7 != 0:
                    skip_all_7 = 1
            else:
                if sDp7 == 1: inv7 = 1
                elif sDp7 == 2: inv7 = 4
                elif sDp7 == 3: inv7 = 5
                elif sDp7 == 4: inv7 = 2
                elif sDp7 == 5: inv7 = 3
                else: inv7 = 6
                q_kill7 = (7 - (SDp7 * inv7) % 7) % 7
                nat7 = (nA7 - SDp7 * q_kill7 % 7 + 49) % 7
                if nat7 != 0:
                    q_kill7_active = 1

            # --- mod 11 ---
            SDp11 = <int>(SDp % 11)
            sDp11 = <int>(sDp % 11)
            nA11 = <int>((node - SDp) % 11)
            skip_all_11 = (sDp11 == 0 and SDp11 == 0 and nA11 != 0)
            q_kill11_active = 0
            q_kill11 = 0
            if not skip_all_11 and sDp11 != 0:
                if sDp11 == 1:   inv11 = 1
                elif sDp11 == 2: inv11 = 6
                elif sDp11 == 3: inv11 = 4
                elif sDp11 == 4: inv11 = 3
                elif sDp11 == 5: inv11 = 9
                elif sDp11 == 6: inv11 = 2
                elif sDp11 == 7: inv11 = 8
                elif sDp11 == 8: inv11 = 7
                elif sDp11 == 9: inv11 = 5
                else:            inv11 = 10
                q_kill11 = (11 - (SDp11 * inv11) % 11) % 11
                nat11 = (nA11 - SDp11 * q_kill11 % 11 + 121) % 11
                if nat11 != 0:
                    q_kill11_active = 1

            # --- mod 13 ---
            SDp13 = <int>(SDp % 13)
            sDp13 = <int>(sDp % 13)
            nA13 = <int>((node - SDp) % 13)
            skip_all_13 = (sDp13 == 0 and SDp13 == 0 and nA13 != 0)
            q_kill13_active = 0
            q_kill13 = 0
            if not skip_all_13 and sDp13 != 0:
                if sDp13 == 1:    inv13 = 1
                elif sDp13 == 2:  inv13 = 7
                elif sDp13 == 3:  inv13 = 9
                elif sDp13 == 4:  inv13 = 10
                elif sDp13 == 5:  inv13 = 8
                elif sDp13 == 6:  inv13 = 11
                elif sDp13 == 7:  inv13 = 2
                elif sDp13 == 8:  inv13 = 5
                elif sDp13 == 9:  inv13 = 3
                elif sDp13 == 10: inv13 = 4
                elif sDp13 == 11: inv13 = 6
                else:             inv13 = 12
                q_kill13 = (13 - (SDp13 * inv13) % 13) % 13
                nat13 = (nA13 - SDp13 * q_kill13 % 13 + 169) % 13
                if nat13 != 0:
                    q_kill13_active = 1

            # Skip paire entière?
            if skip_all_3 or skip_all_5 or skip_all_7 or skip_all_11 or skip_all_13:
                st_skip_pair += 1
                continue

            # skip_all_v2: si v2(den_r) > v2(node) pour TOUT q impair → skip paire
            # Pour v2_node=1 (v2_mask=3): (SDp%4==0 et sDp%4==0) ou (SDp%4==2 et sDp%4==2)
            SDp4 = <int>(SDp & 3)
            sDp4 = <int>(sDp & 3)
            skip_all_v2 = (SDp4 == sDp4) and (SDp4 & 1) == 0
            if skip_all_v2:
                st_skip_pair += 1
                continue

            st_pairs_survived += 1

            # Bisect: premier indice dans sieve > p_v
            qi_lo = 1
            qi_hi = sieve_len
            while qi_lo < qi_hi:
                qi_mid = (qi_lo + qi_hi) // 2
                if sv[qi_mid] <= p_v:
                    qi_lo = qi_mid + 1
                else:
                    qi_hi = qi_mid

            # Boucle sur q > p
            for pi_q in range(qi_lo, sieve_len):
                q_v = sv[pi_q]
                if q_v > q_max_cubic:
                    break

                st_q_tested += 1

                # q copremier à D: bitmask pour petit q
                if pi_q <= N_SMALL:
                    if (div_mask >> (pi_q - 1)) & 1:
                        continue
                else:
                    if D % q_v == 0:
                        continue

                # Filtre per-q: test résidu mod 3, 5, 7, 11 et 13
                if q_kill3_active and q_v % 3 == q_kill3:
                    st_skip_mod += 1
                    continue
                if q_kill5_active and q_v % 5 == q_kill5:
                    st_skip_mod += 1
                    continue
                if q_kill7_active and q_v % 7 == q_kill7:
                    st_skip_mod += 1
                    continue
                if q_kill11_active and q_v % 11 == q_kill11:
                    st_skip_mod += 1
                    continue
                if q_kill13_active and q_v % 13 == q_kill13:
                    st_skip_mod += 1
                    continue

                # SD' * (1+q) > node → break
                SD1q = SDp * (1 + q_v)
                if SD1q > node:
                    break

                num_r = node - SD1q
                den_r = SDp + q_v * sDp

                # Gros idiv
                if num_r % den_r != 0:
                    st_skip_idiv += 1
                    continue

                r_v = num_r // den_r
                if r_v <= q_v:
                    continue
                if r_v == p_v or D % r_v == 0:
                    continue

                # r premier? Miller-Rabin déterministe (élimine les faux positifs)
                if not is_prime_c(r_v):
                    continue

                st_hits += 1
                k_cubic = D * p_v * q_v * r_v
                cubic_cands.append((k_cubic, D, p_v, q_v, r_v))

    stats = {
        'drivers': st_drivers,
        'p_tested': st_p_tested,
        'pairs_survived': st_pairs_survived,
        'q_tested': st_q_tested,
        'skip_pair': st_skip_pair,
        'skip_mod': st_skip_mod,
        'skip_v2': st_skip_v2,
        'skip_idiv': st_skip_idiv,
        'hits': st_hits,
    }
    return cubic_cands, stats


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
