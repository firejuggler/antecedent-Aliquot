#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sqlite_cache.py - Cache SQLite pour antécédents aliquotes

À placer dans le même répertoire que Arbre_multi_g_optimized.py
"""

import sqlite3
import json
from pathlib import Path
from collections import OrderedDict


class SQLiteAntecedenteCache:
    """
    Cache d'antécédents avec SQLite.
    
    Avantages vs JSON:
    - Écritures O(1) au lieu de O(n)  
    - Pas de chargement complet en RAM
    - Transactions atomiques
    - Index B-tree pour lookups rapides
    """
    
    def __init__(self, db_path="antecedents_cache.db", cache_size_mb=100):
        self.db_path = Path(db_path)
        self.conn = None
        self.cache_size_mb = cache_size_mb
        self._write_buffer = OrderedDict()
        self._buffer_size = 1000
        
        self._init_db()
    
    def _init_db(self):
        """Initialise la base SQLite avec optimisations"""
        self.conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None
        )
        
        # Optimisations SQLite
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(f"PRAGMA cache_size=-{self.cache_size_mb * 1024}")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=268435456")
        
        # Créer tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS antecedents (
                node INTEGER PRIMARY KEY,
                antecedents_json TEXT NOT NULL
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        print(f"[Cache] SQLite: {self.db_path} (WAL, {self.cache_size_mb}MB cache)")
    
    def get(self, node):
        """Récupère les antécédents d'un nœud"""
        if node in self._write_buffer:
            return self._write_buffer[node]
        
        cursor = self.conn.execute(
            "SELECT antecedents_json FROM antecedents WHERE node = ?",
            (int(node),)
        )
        row = cursor.fetchone()
        
        if row:
            return json.loads(row[0])
        return None
    
    def set(self, node, antecedents_dict):
        """Enregistre les antécédents (avec buffering)"""
        self._write_buffer[int(node)] = antecedents_dict
        
        if len(self._write_buffer) >= self._buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffer vers SQLite (batch)"""
        if not self._write_buffer:
            return
        
        self.conn.execute("BEGIN TRANSACTION")
        
        try:
            for node, ants in self._write_buffer.items():
                json_str = json.dumps(ants, ensure_ascii=False)
                self.conn.execute(
                    "INSERT OR REPLACE INTO antecedents (node, antecedents_json) VALUES (?, ?)",
                    (node, json_str)
                )
            
            self.conn.execute("COMMIT")
            
        except Exception as e:
            self.conn.execute("ROLLBACK")
            raise e
        
        finally:
            self._write_buffer.clear()
    
    def exists(self, node):
        """Vérifie si un nœud existe"""
        if int(node) in self._write_buffer:
            return True
        
        cursor = self.conn.execute(
            "SELECT 1 FROM antecedents WHERE node = ? LIMIT 1",
            (int(node),)
        )
        return cursor.fetchone() is not None
    
    def get_empty_nodes(self, limit=None):
        """Récupère les nœuds sans antécédents"""
        query = "SELECT node FROM antecedents WHERE antecedents_json = '{}'"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.execute(query)
        return [row[0] for row in cursor.fetchall()]
    
    def count(self):
        """Nombre total de nœuds"""
        count_buffer = len(self._write_buffer)
        cursor = self.conn.execute("SELECT COUNT(*) FROM antecedents")
        count_db = cursor.fetchone()[0]
        return count_db + count_buffer
    
    def count_empty(self):
        """Nombre de nœuds vides"""
        empty_in_buffer = sum(1 for ants in self._write_buffer.values() if not ants)
        cursor = self.conn.execute("SELECT COUNT(*) FROM antecedents WHERE antecedents_json = '{}'")
        empty_in_db = cursor.fetchone()[0]
        return empty_in_db + empty_in_buffer
    
    def close(self):
        """Ferme proprement la connexion"""
        self._flush_buffer()
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Destructeur : flush automatique"""
        try:
            self.close()
        except:
            pass
    
    @classmethod
    def migrate_from_json(cls, json_path, db_path="antecedents_cache.db"):
        """
        Migre un cache JSON vers SQLite.
        
        Usage:
            SQLiteAntecedenteCache.migrate_from_json("antecedents_global_cache.json")
        """
        import time
        
        print(f"[Migration] JSON → SQLite")
        print(f"  Source: {json_path}")
        print(f"  Dest:   {db_path}")
        
        start = time.time()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  Chargé: {len(data):,} nœuds")
        
        cache = cls(db_path)
        cache.conn.execute("BEGIN TRANSACTION")
        
        for i, (node, ants) in enumerate(data.items(), 1):
            json_str = json.dumps(ants, ensure_ascii=False)
            cache.conn.execute(
                "INSERT OR REPLACE INTO antecedents (node, antecedents_json) VALUES (?, ?)",
                (int(node), json_str)
            )
            
            if i % 10000 == 0:
                cache.conn.execute("COMMIT")
                cache.conn.execute("BEGIN TRANSACTION")
                print(f"  Progrès: {i:,}/{len(data):,} ({i/len(data)*100:.1f}%)")
        
        cache.conn.execute("COMMIT")
        
        elapsed = time.time() - start
        print(f"[Migration] Terminée en {elapsed:.1f}s ({len(data)/elapsed:,.0f} nœuds/s)")
        
        cache.close()
        return db_path
