import json
import sys
from collections import defaultdict, deque
from pyvis.network import Network


def charger_json(chemin):
    with open(chemin, 'r') as f:
        return json.load(f)


def extraire_arbre_complet(data, noeud_origine):
    relations = []
    tous_les_noeuds = set([noeud_origine])
    file = deque([noeud_origine])
    visites = set([noeud_origine])
    while file:
        noeud = file.popleft()
        if noeud not in data:
            continue
        for ant in data[noeud]:
            tous_les_noeuds.add(ant)
            relations.append((ant, noeud))
            if ant not in visites:
                visites.add(ant)
                file.append(ant)
    # Construire l'index des successeurs aliquotes depuis le JSON global
    # data[K] contient N signifie s(N) = K
    successeur = {}
    for k, v in data.items():
        for n in v:
            if n not in successeur:
                successeur[n] = k

    # Depuis chaque noeud de l'arbre, suivre la chaîne des successeurs
    # pour détecter et inclure les cycles complets de toute longueur
    noeuds_a_explorer = set(tous_les_noeuds)
    cycles_detectes = []  # liste de frozensets pour éviter les doublons
    for depart in noeuds_a_explorer:
        noeud = depart
        chemin_succ = [noeud]
        visite_succ = {noeud: 0}
        for _ in range(200):
            succ = successeur.get(noeud)
            if succ is None:
                break
            if succ in visite_succ:
                idx = visite_succ[succ]
                membres_cycle = chemin_succ[idx:]
                cle = frozenset(membres_cycle)
                if cle not in cycles_detectes:
                    cycles_detectes.append(cle)
                    for i, m in enumerate(membres_cycle):
                        if m not in tous_les_noeuds:
                            tous_les_noeuds.add(m)
                        suivant = membres_cycle[(i + 1) % len(membres_cycle)]
                        relations.append((m, suivant))
                break
            visite_succ[succ] = len(chemin_succ)
            chemin_succ.append(succ)
            noeud = succ

    # Collecter tous les membres de cycles depuis les cycles déjà détectés
    membres_cycles_tous = set()
    for cle in cycles_detectes:
        for m in cle:
            membres_cycles_tous.add(m)

    return relations, tous_les_noeuds, membres_cycles_tous


def trouver_cycles(relations):
    """Détecte tous les noeuds appartenant à un cycle (toute longueur)."""
    g = defaultdict(set)
    for a, b in relations:
        g[a].add(b)
    en_cycle = set()
    # Détection des cycles directs A <-> B et cycles plus longs via DFS itératif
    # Cycles directs
    for n in list(g):
        for v in list(g[n]):
            if n in g[v]:
                en_cycle.add(n)
                en_cycle.add(v)
    # Cycles plus longs : DFS itératif (Tarjan simplifié)
    tous_noeuds = set(x for p in relations for x in p)
    visites_globales = set()
    for depart in tous_noeuds:
        if depart in visites_globales:
            continue
        pile = [(depart, [depart], {depart})]
        while pile:
            noeud, chemin, en_cours = pile.pop()
            visites_globales.add(noeud)
            for voisin in list(g[noeud]):
                if voisin in en_cours:
                    idx = chemin.index(voisin)
                    for n in chemin[idx:]:
                        en_cycle.add(n)
                elif voisin not in visites_globales:
                    pile.append((voisin, chemin + [voisin], en_cours | {voisin}))
    return en_cycle


def chemin_bfs(racine, cible, graphe_desc):
    """BFS depuis une racine pour trouver le chemin vers la cible."""
    file = deque([(racine, [racine])])
    visites = {racine}
    while file:
        noeud, chemin = file.popleft()
        if noeud == cible:
            return chemin
        for enfant in graphe_desc.get(noeud, set()):
            if enfant not in visites:
                visites.add(enfant)
                file.append((enfant, chemin + [enfant]))
    return None


def trouver_chemin_principal(racines_vraies, noeud_origine, graphe_descendant):
    """Trouve le plus long chemin d'une racine jusqu'au noeud origine."""
    meilleur = []
    for racine in racines_vraies:
        ch = chemin_bfs(racine, noeud_origine, graphe_descendant)
        if ch and len(ch) > len(meilleur):
            meilleur = ch
    return meilleur


def trouver_branches(racines_vraies, noeud_origine, chemin_set, graphe_descendant, membres_cycles=None, tous_les_noeuds=None):
    """
    Pour chaque racine inférieure au minimum du cycle (ou au noeud d'origine),
    trouve son chemin complet jusqu'au premier noeud du chemin principal OU
    du cycle (jonction). Retourne une liste de (jonction, chemin_lateral).
    """
    # Référence = min du cycle si l'origine est dans un cycle, sinon noeud_origine
    if membres_cycles:
        valeur_ref = min(int(m) for m in membres_cycles)
    else:
        valeur_ref = int(noeud_origine)

    # Points d'attache valides = chemin principal + tous membres du cycle
    points_attache = chemin_set | (membres_cycles if membres_cycles else set())

    branches = []
    noeuds_deja_en_branche = set()

    def bfs_vers_attache(depart, points_attache, graphe_desc):
        """BFS depuis depart vers le premier point d'attache rencontré (en profondeur)."""
        file = deque([(depart, [depart])])
        visites = {depart}
        while file:
            noeud, chemin = file.popleft()
            if noeud in points_attache:
                return chemin[:-1], noeud  # chemin sans la jonction, jonction
            for enfant in graphe_desc.get(noeud, set()):
                if enfant not in visites:
                    visites.add(enfant)
                    file.append((enfant, chemin + [enfant]))
        return None, None

    # Chercher depuis tous les noeuds < valeur_ref, triés du plus petit au plus grand
    # pour s'assurer que les noeuds les plus bas sont traités en premier
    base = tous_les_noeuds if tous_les_noeuds else racines_vraies
    candidats = sorted(
        [n for n in base if int(n) < valeur_ref and n not in points_attache],
        key=lambda x: int(x)
    )
    racines_de_branches = set()  # noeuds déjà utilisés comme point de départ de branche
    for racine in candidats:
        if racine in racines_de_branches:
            continue
        # Trouver la jonction la plus proche dans points_attache
        chemin_lateral, jonction = bfs_vers_attache(racine, points_attache, graphe_descendant)
        if chemin_lateral is None:
            continue
        # Trouver le premier noeud < valeur_ref dans le chemin (la vraie racine affichable)
        # et garder uniquement les noeuds consécutifs < valeur_ref à partir de là
        idx_debut = None
        for j, m in enumerate(chemin_lateral):
            if int(m) < valeur_ref:
                idx_debut = j
                break
        if idx_debut is None:
            continue  # aucun noeud < valeur_ref dans cette branche
        # Couper à partir du premier noeud >= valeur_ref après idx_debut
        chemin_valide = []
        for m in chemin_lateral[idx_debut:]:
            if int(m) < valeur_ref:
                chemin_valide.append(m)
            else:
                break  # noeud trop grand, arrêt
        chemin_lateral = chemin_valide
        # Rejeter si la racine effective de la branche est > jonction :
        # la jonction est déjà une meilleure origine, la branche est superflue
        if chemin_lateral and int(chemin_lateral[0]) < int(jonction):
            branches.append((jonction, chemin_lateral))
            racines_de_branches.add(chemin_lateral[0])
    return branches


def generer_graphe(noeud_origine, chemin_json, fichier_sortie=None):
    print(f"Chargement du JSON...")
    data = charger_json(chemin_json)

    noeud_origine = str(noeud_origine)
    print(f"Extraction de l'arbre depuis {noeud_origine}...")
    relations, tous_les_noeuds, membres_cycles = extraire_arbre_complet(data, noeud_origine)
    print(f"  → {len(tous_les_noeuds)} noeuds, {len(relations)} relations")

    graphe_montant = defaultdict(set)
    graphe_descendant = defaultdict(set)
    for p, e in relations:
        graphe_montant[e].add(p)
        graphe_descendant[p].add(e)

    est_un_enfant = set(e for _, e in relations)
    racines_vraies = tous_les_noeuds - est_un_enfant
    nombres_en_cycle = trouver_cycles(relations)
    print(f"  → {len(racines_vraies)} racines, {len(nombres_en_cycle)} noeud(s) en cycle")

    print(f"Calcul du chemin principal...")
    meilleur_chemin = trouver_chemin_principal(racines_vraies, noeud_origine, graphe_descendant)
    chemin_set = set(meilleur_chemin)
    print(f"  → {len(meilleur_chemin)} noeuds : {meilleur_chemin[0]} → {meilleur_chemin[-1]}")

    # Calcul de la valeur de référence pour le filtre des branches
    if membres_cycles:
        valeur_ref = min(int(m) for m in membres_cycles)
    else:
        valeur_ref = int(noeud_origine)

    # Tronquer le chemin principal : supprimer les noeuds en tête > valeur_ref
    if not membres_cycles:
        # Pas de cycle : couper les noeuds > noeud_origine en tête du chemin
        idx_coupe = 0
        for i, n in enumerate(meilleur_chemin):
            if int(n) > valeur_ref:
                idx_coupe = i + 1
            else:
                break
        if idx_coupe > 0:
            meilleur_chemin = meilleur_chemin[idx_coupe:]
            chemin_set = set(meilleur_chemin)
            print(f"  → Chemin tronqué à {len(meilleur_chemin)} noeuds (ref={valeur_ref})")
    elif membres_cycles and noeud_origine not in membres_cycles:
        # L'origine n'est pas dans le cycle : on cherche où le chemin rejoint le cycle
        for i, n in enumerate(meilleur_chemin):
            if n in membres_cycles:
                meilleur_chemin = meilleur_chemin[i:]
                chemin_set = set(meilleur_chemin)
                print(f"  → Chemin tronqué à {len(meilleur_chemin)} noeuds (entrée cycle: {n})")
                break
    elif membres_cycles:
        # L'origine est dans le cycle : tronquer les noeuds avant l'origine
        # qui sont supérieurs à valeur_ref
        idx_coupe = 0
        for i, n in enumerate(meilleur_chemin):
            if int(n) > valeur_ref and n not in membres_cycles:
                idx_coupe = i + 1
            else:
                break
        if idx_coupe > 0:
            meilleur_chemin = meilleur_chemin[idx_coupe:]
            chemin_set = set(meilleur_chemin)
            print(f"  → Chemin tronqué à {len(meilleur_chemin)} noeuds (ref={valeur_ref})")

    print(f"Calcul des branches latérales (racines < origine)...")
    branches = trouver_branches(racines_vraies, noeud_origine, chemin_set, graphe_descendant, membres_cycles, tous_les_noeuds)
    print(f"  → {len(branches)} branche(s) retenue(s)")
    for jonction, ch in branches:
        print(f"     racine={ch[0]}, jonction={jonction}, {len(ch)} noeuds")

    # Construire noeuds et arêtes des branches
    tous_noeuds_branches = set()
    toutes_rels_branches = set()
    for jonction, ch in branches:
        for n in ch:
            tous_noeuds_branches.add(n)
        for i in range(len(ch) - 1):
            toutes_rels_branches.add((ch[i], ch[i + 1]))
        toutes_rels_branches.add((ch[-1], jonction))

    noeuds_a_afficher = chemin_set | tous_noeuds_branches | membres_cycles

    # Construction du graphe
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)

    # Recalculer est_un_enfant uniquement sur les noeuds affichés
    enfants_affiches = set(e for p, e in relations if p in noeuds_a_afficher and e in noeuds_a_afficher)

    for n_str in noeuds_a_afficher:
        if n_str in nombres_en_cycle:
            net.add_node(n_str, label=n_str, size=35, color="#FF00FF", shape="diamond",
                         title="Fait partie d'un cycle A-B-A")
        elif n_str not in enfants_affiches:
            net.add_node(n_str, label=n_str, size=40, color="#FFD700", shape="dot",
                         title="Racine (sans parent)")
        else:
            net.add_node(n_str, label=n_str, size=20, color="#97C2FC", shape="dot")

    # Arêtes du chemin principal
    for i in range(len(meilleur_chemin) - 1):
        p, e = meilleur_chemin[i], meilleur_chemin[i + 1]
        if p in nombres_en_cycle and e in nombres_en_cycle:
            c, w = "#FF00FF", 5
        elif int(p) < int(e):
            c, w = "#2ECC40", 2
        else:
            c, w = "#FF4136", 2
        net.add_edge(p, e, color=c, width=w)

    # Arêtes des branches latérales
    for (p, e) in toutes_rels_branches:
        if p in nombres_en_cycle and e in nombres_en_cycle:
            c, w = "#FF00FF", 5
        elif int(p) < int(e):
            c, w = "#2ECC40", 2
        else:
            c, w = "#FF4136", 2
        net.add_edge(p, e, color=c, width=w)

    # Arêtes des cycles complets
    g_relations = defaultdict(set)
    for p, e in relations:
        g_relations[p].add(e)
    for m in membres_cycles:
        for v in list(g_relations[m]):
            if v in membres_cycles:
                if m in nombres_en_cycle and v in nombres_en_cycle:
                    c, w = "#FF00FF", 5
                elif int(m) < int(v):
                    c, w = "#2ECC40", 2
                else:
                    c, w = "#FF4136", 2
                net.add_edge(m, v, color=c, width=w)

    net.set_options("""var options = {"layout": {"hierarchical": {"enabled": false, "direction": "UD", "sortMethod": "directed", "nodeSpacing": 300}}, "physics": {"enabled": true}}""")

    if fichier_sortie is None:
        fichier_sortie = f"arbre_collatz_{noeud_origine}.html"

    net.show(fichier_sortie, notebook=False)
    print(f"\nGraphe généré : {fichier_sortie}")
    print(f"  {len(noeuds_a_afficher)} noeuds affichés")
    print(f"Légende :")
    print(f"  ◆ Fuchsia = noeuds en cycle")
    print(f"  ● Or      = racines (sans parent)")
    print(f"  ● Bleu    = noeuds intermédiaires")
    print(f"  → Vert    = antécédent < descendant")
    print(f"  → Rouge   = antécédent > descendant")


if __name__ == "__main__":
    CHEMIN_JSON = "antecedents_global_cache.json"

    if len(sys.argv) < 2:
        with open(CHEMIN_JSON) as f:
            d = json.load(f)
        noeud = list(d.keys())[0]
        print(f"Aucun noeud spécifié. Utilisation du premier noeud: {noeud}")
    else:
        noeud = sys.argv[1]

    sortie = f"arbre_collatz_{noeud}.html"
    generer_graphe(noeud, CHEMIN_JSON, fichier_sortie=sortie)