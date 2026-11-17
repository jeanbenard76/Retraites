#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse du système des retraites en France (API DREES & Eurostat)
================================================================

Ce script propose une analyse complète, reproductible et entièrement paramétrable
du système des retraites français à partir de données publiques officielles
interrogées par API :

1. Connexion API aux sources fiables (avec cache local dans ./cache_api/) :
   - DREES Open Data (API Explore v2.1) :
     * Comptes de la protection sociale (`305_les-comptes-de-la-protection-sociale`)
     * Âge conjoncturel de départ (`panorama-retraite2025_graphique-1_age-de-depart`)
     * Caractéristiques des retraités / carrières complètes (`rec02`)
     * Taux de remplacement (`repartition-des-taux-de-remplacement-...`)
     * Distribution fine des pensions mensuelles (`4178_distribution-des-pensions-mensuelles`, EIR 2020)
     * Effectifs, montants moyens par âge et par régime (`1393_les-effectifs-et-montants-de-pension-des-retraites`, fin 2022)
   - Eurostat (API Dissemination JSON-stat) :
     * Pyramide des âges officielle de la France (`demo_pjan`) par âge exact (0 à 100+ ans)

2. Partie 1A — État des lieux : « Qui touche combien, et quand ? »
   - Dépenses macroéconomiques (Vieillesse-Survie) par prestation et par régime.
   - Distribution des pensions nettes par tranche, sexe, régime et indicateurs d'inégalités.
   - Âges de liquidation (primo-liquidants) et profil des pensions par âge actuel.

3. Partie 1B — Simulateur paramétrable en répartition (Minimum, Maximum, Écrêtement)
   - Simulation de planchers (universel ou réservé aux carrières complètes),
     plafonds, écrêtements progressifs et revalorisations.
   - Solveur automatique du plafond d'autofinancement à solde budgétaire nul.

4. Partie 2 — Simulateur de passage à la capitalisation
   - Paramétrable sur la durée de transition (N années), le socle maintenu en répartition (X %),
     et l'activation ou non d'une dotation à la naissance (« Baby Bond »).
   - Résultats détaillés pour les nouveaux arrivants (nouveau-né et nouvel actif à 22 ans),
     pour chaque tranche d'âge actuelle (population réelle Eurostat),
     et trajectoire macroéconomique du besoin de financement transitoire.

5. Génération d'une planche graphique récapitulative (`analyse_retraites_graphiques.png`).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
if "ipykernel" not in sys.modules and "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Garantit un affichage UTF-8 propre sous console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ==============================================================================
# 1. CONSTANTES & URLS DES API OFFICIELLES
# ==============================================================================

DREES_API_BASE = "https://data.drees.solidarites-sante.gouv.fr/api/explore/v2.1/catalog/datasets"
EUROSTAT_API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

URLS_API = {
    "cps_records": f"{DREES_API_BASE}/305_les-comptes-de-la-protection-sociale/records",
    "age_depart": f"{DREES_API_BASE}/panorama-retraite2025_graphique-1_age-de-depart/records",
    "rec02": f"{DREES_API_BASE}/rec02/records",
    "taux_remplacement": (
        f"{DREES_API_BASE}/repartition-des-taux-de-remplacement-entre-les-revenus-"
        "juste-avant-et-juste-apres-la-retraite/records"
    ),
    "eir2020_xlsx": (
        f"{DREES_API_BASE}/4178_distribution-des-pensions-mensuelles/attachments/"
        "eir2020_distribution_des_pensions_mensuelles_xlsx"
    ),
    "effectifs_2022_xlsx": (
        f"{DREES_API_BASE}/1393_les-effectifs-et-montants-de-pension-des-retraites/attachments/"
        "les_effectifs_de_retraites_montants_des_pensions_et_ages_de_depart_a_la_retraite_fin_2022_xlsx"
    ),
    "distrib_age_2017_xlsx": (
        f"{DREES_API_BASE}/1393_les-effectifs-et-montants-de-pension-des-retraites/attachments/"
        "distribution_des_effectifs_et_des_montants_de_pension_par_age_fin_2017xlsx"
    ),
    "eurostat_pop_fr": f"{EUROSTAT_API_BASE}/demo_pjan?geo=FR&sex=T&lastTimePeriod=1",
}

CACHE_DIR = Path(__file__).resolve().parent / "cache_api"
OUTPUT_PNG = Path(__file__).resolve().parent / "analyse_retraites_graphiques.png"


# ==============================================================================
# 2. STRUCTURES DE PARAMÈTRES & DE DONNÉES
# ==============================================================================

@dataclass
class ParametresReformeRepartition:
    """Paramètres configurables pour la simulation d'une réforme en répartition."""
    nom: str = "Scénario personnalisé"
    pension_minimum: Optional[float] = 1200.0
    appliquer_min_carriere_complete_seule: bool = False
    pension_maximum: Optional[float] = 3000.0
    seuil_ecretement: Optional[float] = None
    taux_ecretement_pct: float = 0.0
    revalorisation_globale_pct: float = 0.0


@dataclass
class ParametresCapitalisation:
    """Paramètres configurables pour le passage à un système mixte avec capitalisation."""
    duree_transition_annees: int = 25          # Durée de montée en charge (0 = bascule immédiate)
    socle_repartition_pct: float = 40.0        # X % maintenus en répartition à terme
    dotation_naissance_active: bool = True     # Activer ou non une dotation à la naissance
    montant_dotation_naissance: float = 10000.0
    donner_dotation_aux_enfants_actuels: bool = False  # Si True, inclut les 0-17 ans déjà nés
    rendement_annuel_reel_pct: float = 4.5     # Rendement réel (net d'inflation) des fonds
    croissance_salaires_reelle_pct: float = 1.0
    taux_cotisation_retraite_pct: float = 27.8  # Part totale du salaire brut dédiée à la retraite
    salaire_brut_mensuel_ref: float = 3300.0   # Salaire brut mensuel temps plein de référence
    salaire_net_mensuel_ref: float = 2580.0    # Salaire net mensuel correspondant
    taux_remplacement_repartition_pct: float = 72.0  # Pension PAYG carrière complète (% du dernier net)
    age_entree_emploi: int = 22
    age_retraite: int = 64
    duree_retraite_annees: float = 21.5        # Espérance de vie moyenne à l'âge de la retraite
    ratio_net_brut_rente: float = 0.905        # Prélèvements sociaux (CSG/CRDS/CASA) à la liquidation


@dataclass
class DonneesRetraites:
    """Conteneur structuré de toutes les séries chargées depuis les APIs."""
    provenance: Dict[str, str] = field(default_factory=dict)
    annee_cps: str = "2024"
    cps_prestations: pd.DataFrame = field(default_factory=pd.DataFrame)
    cps_regimes: pd.DataFrame = field(default_factory=pd.DataFrame)
    masse_pensions_brut_mde: float = 406.0
    ratio_net_brut: float = 0.915
    effectif_total_retraites_k: float = 17889.2
    effectif_droit_direct_k: float = 16974.8
    part_carriere_complete_pct: float = 57.3
    taux_remplacement_median_cc_pct: float = 77.3
    distrib_pensions_eir: pd.DataFrame = field(default_factory=pd.DataFrame)
    distrib_pensions_2024: pd.DataFrame = field(default_factory=pd.DataFrame)
    facteur_revalo_2020_2024: float = 1.0
    pensions_par_regime: pd.DataFrame = field(default_factory=pd.DataFrame)
    detail_caisses_2022: pd.DataFrame = field(default_factory=pd.DataFrame)
    age_depart_historique: pd.DataFrame = field(default_factory=pd.DataFrame)
    primo_liquidants_par_age: pd.DataFrame = field(default_factory=pd.DataFrame)
    pensions_par_age: pd.DataFrame = field(default_factory=pd.DataFrame)
    annee_eurostat: str = "2025"
    population_par_age: pd.DataFrame = field(default_factory=pd.DataFrame)


# ==============================================================================
# 3. CONNECTEUR API ROBUSTE (RETRIES + CACHE LOCAL)
# ==============================================================================

class ConnecteurDonneesRetraites:
    """Télécharge, met en cache et parse les données officielles DREES & Eurostat."""

    def __init__(self, cache_dir: Path = CACHE_DIR, refresh_cache: bool = False):
        self.cache_dir = cache_dir
        self.refresh_cache = refresh_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = self._creer_session_http()
        self.provenance: Dict[str, str] = {}

    @staticmethod
    def _creer_session_http() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": "AnalyseRetraitesFrance-Python/1.0"})
        return session

    def _fetch_bytes(self, cle: str, url: str, params: Optional[Dict[str, Any]] = None) -> bytes:
        """Récupère une ressource HTTP binaire avec cache local transparent."""
        suffixe = ".xlsx" if "xlsx" in cle else ".json"
        cache_path = self.cache_dir / f"{cle}{suffixe}"

        if cache_path.exists() and not self.refresh_cache:
            self.provenance[cle] = f"Cache local ({cache_path.name})"
            return cache_path.read_bytes()

        try:
            resp = self.session.get(url, params=params, timeout=25)
            resp.raise_for_status()
            contenu = resp.content
            cache_path.write_bytes(contenu)
            self.provenance[cle] = f"API Live ({url.split('/')[2]})"
            return contenu
        except requests.RequestException as exc:
            if cache_path.exists():
                self.provenance[cle] = f"Cache secours (API indisponible : {exc})"
                return cache_path.read_bytes()
            raise RuntimeError(f"Échec de connexion à l'API '{cle}' ({url}) : {exc}") from exc

    def _fetch_json(self, cle: str, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raw = self._fetch_bytes(cle, url, params=params)
        return json.loads(raw.decode("utf-8"))

    def charger_tout(self) -> DonneesRetraites:
        """Orchestre le chargement et le nettoyage de toutes les sources API."""
        donnees = DonneesRetraites()

        # 1) Comptes de la protection sociale (DREES API JSON)
        annee_cps, df_prest, df_reg = self._charger_cps_drees()
        donnees.annee_cps = annee_cps
        donnees.cps_prestations = df_prest
        donnees.cps_regimes = df_reg

        # Masse stricte des pensions (Droit direct + Droit dérivé + Minimum vieillesse)
        codes_pensions = {"E11-21.1", "E11-22.1", "E11-21.2"}
        masse_m = df_prest[df_prest["ps_code"].isin(codes_pensions)]["montant_m"].sum()
        donnees.masse_pensions_brut_mde = float(masse_m) / 1000.0 if masse_m > 0 else 406.0

        # 2) Indicateurs complémentaires DREES (Âge de départ, carrières complètes, remplacement)
        donnees.age_depart_historique = self._charger_age_depart()
        donnees.part_carriere_complete_pct = self._charger_part_carriere_complete()
        donnees.taux_remplacement_median_cc_pct = self._charger_taux_remplacement_median()

        # 3) Fichiers Excel DREES récupérés via l'API Attachments
        donnees.distrib_pensions_eir = self._charger_eir_distribution()
        eff_tot, eff_dd, ratio_nb, df_reg_pens, df_caisses, df_primo, df_age_2022 = self._charger_effectifs_2022()
        df_age_2017 = self._charger_effectifs_age_2017()

        donnees.effectif_total_retraites_k = eff_tot
        donnees.effectif_droit_direct_k = eff_dd
        donnees.ratio_net_brut = ratio_nb
        donnees.pensions_par_regime = df_reg_pens
        donnees.detail_caisses_2022 = df_caisses
        donnees.primo_liquidants_par_age = df_primo
        donnees.pensions_par_age = self._fusionner_pensions_et_effectifs_par_age(
            df_age_2022, df_age_2017, eff_dd
        )

        # 4) Recalibrage de la distribution micro (EIR 2020) sur l'enveloppe CPS 2024
        df_2024, facteur = self._calibrer_distribution_2024(
            donnees.distrib_pensions_eir,
            donnees.masse_pensions_brut_mde,
            donnees.ratio_net_brut,
            donnees.effectif_total_retraites_k,
            donnees.part_carriere_complete_pct,
        )
        donnees.distrib_pensions_2024 = df_2024
        donnees.facteur_revalo_2020_2024 = facteur

        # 5) Pyramide des âges officielle de la France (API Eurostat)
        annee_eu, df_pop = self._charger_pyramide_eurostat()
        donnees.annee_eurostat = annee_eu
        donnees.population_par_age = df_pop

        donnees.provenance = self.provenance
        return donnees

    # --- Parseurs spécialisés par source ---

    def _charger_cps_drees(self) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
        latest = self._fetch_json(
            "cps_latest_year",
            URLS_API["cps_records"],
            params={"select": "annee", "where": 'risque="VIEILLESSE-SURVIE"', "order_by": "annee DESC", "limit": 1},
        )
        annee = str(latest["results"][0]["annee"])[:4]

        prest_json = self._fetch_json(
            f"cps_prestations_{annee}",
            URLS_API["cps_records"],
            params={
                "where": f'risque="VIEILLESSE-SURVIE" and regime_nom="Total tous régimes" and annee="{annee}"',
                "limit": 100,
            },
        )
        df_prest = pd.DataFrame(prest_json.get("results", [])).rename(columns={"val": "montant_m"})
        df_prest["montant_m"] = pd.to_numeric(df_prest["montant_m"], errors="coerce").fillna(0.0)
        df_prest["montant_mde"] = df_prest["montant_m"] / 1000.0

        reg_json = self._fetch_json(
            f"cps_regimes_{annee}",
            URLS_API["cps_records"],
            params={
                "where": f'risque="VIEILLESSE-SURVIE" and ps_niveau="1" and annee="{annee}"',
                "limit": 100,
            },
        )
        df_reg = pd.DataFrame(reg_json.get("results", [])).rename(columns={"val": "montant_m"})
        df_reg["montant_m"] = pd.to_numeric(df_reg["montant_m"], errors="coerce").fillna(0.0)
        # Exclut les lignes agrégées ("Total tous régimes", "Total S13141", etc.)
        df_reg = df_reg[~df_reg["regime_nom"].astype(str).str.startswith("Total")].copy()
        df_reg["montant_mde"] = df_reg["montant_m"] / 1000.0
        df_reg = df_reg.sort_values("montant_mde", ascending=False).reset_index(drop=True)
        return annee, df_prest, df_reg

    def _charger_age_depart(self) -> pd.DataFrame:
        data = self._fetch_json("age_depart_drees", URLS_API["age_depart"], params={"limit": 100})
        df = pd.DataFrame(data.get("results", []))
        for col in ("annee", "femme", "homme", "ensemble"):
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
                )
        return df[["annee", "ensemble", "femme", "homme"]].dropna().sort_values("annee").reset_index(drop=True)

    def _charger_part_carriere_complete(self) -> float:
        data = self._fetch_json(
            "rec02_carriere_complete",
            URLS_API["rec02"],
            params={
                "where": 'champ="Ensemble des retraités" and sexe="Ensemble" and variable="Carrière complète"',
                "limit": 10,
            },
        )
        results = data.get("results", [])
        if results and results[0].get("ensemble") is not None:
            return float(results[0]["ensemble"])
        return 57.3

    def _charger_taux_remplacement_median(self) -> float:
        data = self._fetch_json(
            "taux_remplacement_cc",
            URLS_API["taux_remplacement"],
            params={
                "where": (
                    'caracteristique="Durée de carrière" and categorie="Carrière complète en France" '
                    'and sexe="Ensemble" and revenu_utilise_pour_le_calcul_du_taux_de_remplacement='
                    '"Retraite / revenus du travail"'
                ),
                "order_by": "premiere_annee_pleine_de_retraite DESC",
                "limit": 5,
            },
        )
        results = data.get("results", [])
        if results and results[0].get("taux_de_remplacement_quantile_a_50") is not None:
            return float(results[0]["taux_de_remplacement_quantile_a_50"])
        return 77.3

    @staticmethod
    def _parse_tranche_euro(libelle: str) -> Tuple[float, float]:
        txt = libelle.strip()
        if txt.startswith("De "):
            corps = txt.replace("De ", "").replace(" euros", "").strip()
            b_min, b_max = corps.split(" à ")
            return float(b_min), float(b_max)
        if txt.startswith("Supérieur"):
            return 4500.0, 7500.0
        raise ValueError(f"Libellé de tranche inattendu : {libelle}")

    def _charger_eir_distribution(self) -> pd.DataFrame:
        raw = self._fetch_bytes("eir2020_xlsx", URLS_API["eir2020_xlsx"])
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)

        def lire_feuille(nom_feuille: str, suffixe: str) -> pd.DataFrame:
            ws = wb[nom_feuille]
            lignes = []
            for row in ws.iter_rows(values_only=True):
                if row and isinstance(row[0], str):
                    lbl = row[0].strip()
                    if lbl.startswith("De ") or lbl.startswith("Supérieur"):
                        b_min, b_max = self._parse_tranche_euro(lbl)
                        lignes.append(
                            {
                                "tranche": lbl,
                                "borne_min": b_min,
                                "borne_max": b_max,
                                f"pct_femmes_{suffixe}": float(row[1] or 0.0),
                                f"pct_hommes_{suffixe}": float(row[2] or 0.0),
                                f"pct_ensemble_{suffixe}": float(row[3] or 0.0),
                            }
                        )
            return pd.DataFrame(lignes)

        df_all = lire_feuille("pension_nette totale", "all")
        df_cc = lire_feuille("pension nette_carrière complète", "cc")
        df = df_all.merge(
            df_cc[["tranche", "pct_femmes_cc", "pct_hommes_cc", "pct_ensemble_cc"]],
            on="tranche",
            how="left",
        )
        # Moyenne calibrée dans la tranche ouverte (> 4500 €) pour respecter la moyenne EIR (~5200 €)
        df["centre"] = np.where(
            df["borne_min"] >= 4500.0,
            5200.0,
            0.5 * (df["borne_min"] + df["borne_max"]),
        )
        return df

    def _charger_effectifs_2022(
        self,
    ) -> Tuple[float, float, float, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raw = self._fetch_bytes("effectifs_2022_xlsx", URLS_API["effectifs_2022_xlsx"])
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)

        # 1.1-a : effectifs totaux et de droit direct (dernière année numérique)
        eff_tot_k, eff_dd_k = 17889.2, 16974.8
        for row in wb["1.1-a"].iter_rows(values_only=True):
            if row and row[0] is not None and str(row[0]).strip()[:4].isdigit():
                eff_tot_k = float(row[1])
                eff_dd_k = float(row[4])

        # 2.1-a : pension moyenne brute vs nette (ratio net/brut le plus récent)
        ratio_nb = 0.915
        for row in wb["2.1-a"].iter_rows(values_only=True):
            if row and row[0] is not None and str(row[0]).strip()[:4].isdigit():
                try:
                    brut = float(row[7])  # Brute y compris majo 3 enfants
                    net = float(row[4])   # Nette
                    if brut > 0 and net > 0:
                        ratio_nb = net / brut
                except (TypeError, ValueError):
                    pass

        # 1.2-b : effectifs (milliers) et part (%) par régime principal d'affiliation
        def _cle_regime(lib: str) -> str:
            txt = lib.lower().replace("’", "'").replace(" de l'", " d'").strip()
            return txt.rstrip("0123456789, ")

        eff_par_reg: Dict[str, Tuple[float, float]] = {}
        for row in wb["1.2-b"].iter_rows(values_only=True):
            if row and isinstance(row[0], str) and isinstance(row[1], (int, float)):
                eff_par_reg[_cle_regime(row[0])] = (float(row[1]), float(row[2] or 0.0))

        # 2.1-d : pensions moyennes par régime principal (+ effectifs de 1.2-b)
        lignes_reg = []
        for row in wb["2.1-d"].iter_rows(values_only=True):
            if row and isinstance(row[0], str) and isinstance(row[1], (int, float)):
                eff_k, part_pct = eff_par_reg.get(_cle_regime(row[0]), (float("nan"), float("nan")))
                lignes_reg.append(
                    {
                        "regime": row[0].strip(),
                        "effectifs_k": eff_k,
                        "part_pct": part_pct,
                        "tous_ensemble": float(row[1]),
                        "tous_femmes": float(row[2]),
                        "tous_hommes": float(row[3]),
                        "cc_ensemble": float(row[4]),
                        "cc_femmes": float(row[5]),
                        "cc_hommes": float(row[6]),
                    }
                )
        df_reg = pd.DataFrame(lignes_reg)

        # 1.2-a & 2.1-c : effectifs et avantage principal moyen par caisse (colonne 2022 Ensemble)
        rows_12a = list(wb["1.2-a"].iter_rows(values_only=True))
        rows_21c = list(wb["2.1-c"].iter_rows(values_only=True))
        idx_eff_2022 = next((i for i, v in enumerate(rows_12a[3]) if str(v) == "2022"), None)
        idx_mon_2022 = next((i for i, v in enumerate(rows_21c[4]) if str(v) == "2022"), None)

        eff_caisse: Dict[str, float] = {}
        if idx_eff_2022 is not None:
            for r in rows_12a[4:]:
                if r and isinstance(r[0], str) and r[0].strip().lower() not in ("nd : non disponible",):
                    val = r[idx_eff_2022]
                    if isinstance(val, (int, float)):
                        eff_caisse[r[0].strip().rstrip("0123456789 ")] = float(val)
                if r and isinstance(r[0], str) and r[0].strip().startswith("CNBF"):
                    break

        lignes_caisses = []
        if idx_mon_2022 is not None:
            for r in rows_21c[5:]:
                if not r or not isinstance(r[0], str):
                    continue
                if r[0].strip().startswith("nd :"):
                    break
                nom_c = r[0].strip().rstrip("0123456789 ")
                val_m = r[idx_mon_2022]
                if str(val_m).strip().lower() == "nd" or val_m is None:
                    continue
                try:
                    pension_c = float(str(val_m).replace(",", "."))
                except ValueError:
                    continue
                eff_c = eff_caisse.get(nom_c, float("nan"))
                masse_est = (eff_c * 1000.0 * pension_c * 12.0 / 1e9) if not math.isnan(eff_c) else float("nan")
                lignes_caisses.append(
                    {
                        "caisse": nom_c,
                        "effectifs_2022_k": eff_c,
                        "pension_caisse_brut_eur": pension_c,
                        "masse_droit_direct_est_mde": masse_est,
                    }
                )
        df_caisses = pd.DataFrame(lignes_caisses)

        # 1.3-d : primo-liquidants par âge en 2022
        lignes_primo = []
        for row in wb["1.3-d"].iter_rows(values_only=True):
            if row and row[0] is not None and isinstance(row[1], (int, float)):
                age_lbl = str(row[0]).strip()
                if age_lbl.lower() == "ensemble":
                    continue
                age_num = 49 if "moins" in age_lbl.lower() else (71 if "plus" in age_lbl.lower() else int(age_lbl))
                lignes_primo.append(
                    {
                        "age_label": age_lbl,
                        "age_num": age_num,
                        "ensemble": float(row[1]),
                        "femmes": float(row[2]),
                        "hommes": float(row[3]),
                    }
                )
        df_primo = pd.DataFrame(lignes_primo)

        # 2.1-h : pension brute moyenne par âge actuel en 2022
        lignes_age = []
        for row in wb["2.1-h"].iter_rows(values_only=True):
            if row and row[0] is not None and isinstance(row[1], (int, float)):
                age_lbl = str(row[0]).strip()
                if age_lbl.lower() == "ensemble":
                    continue
                age_num = 49 if "moins" in age_lbl.lower() else (91 if "plus" in age_lbl.lower() else int(age_lbl))
                lignes_age.append(
                    {
                        "age_label": age_lbl,
                        "age_num": age_num,
                        "pension_brut_ensemble": float(row[1]),
                        "pension_brut_femmes": float(row[2]),
                        "pension_brut_hommes": float(row[3]),
                    }
                )
        df_age = pd.DataFrame(lignes_age)

        return eff_tot_k, eff_dd_k, ratio_nb, df_reg, df_caisses, df_primo, df_age

    def _charger_effectifs_age_2017(self) -> pd.DataFrame:
        raw = self._fetch_bytes("distrib_age_2017_xlsx", URLS_API["distrib_age_2017_xlsx"])
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        ws = wb.active
        lignes = []
        for row in ws.iter_rows(values_only=True):
            if row and isinstance(row[0], (int, float)) and str(row[1]).strip() == "Ensemble":
                lignes.append({"age_num": int(row[0]), "effectif_2017_k": float(row[3])})
        return pd.DataFrame(lignes)

    @staticmethod
    def _fusionner_pensions_et_effectifs_par_age(
        df_age_2022: pd.DataFrame, df_age_2017: pd.DataFrame, eff_dd_k: float
    ) -> pd.DataFrame:
        df = df_age_2022.merge(df_age_2017, on="age_num", how="left")
        # Complète les âges extrêmes (<50 et >90) avec des poids cohérents
        df["effectif_2017_k"] = df["effectif_2017_k"].fillna(
            pd.Series(np.where(df["age_num"] < 50, 20.0, 450.0), index=df.index)
        )
        somme_17 = df["effectif_2017_k"].sum()
        df["effectif_estime_k"] = df["effectif_2017_k"] * (eff_dd_k / somme_17)
        return df

    @staticmethod
    def _calibrer_distribution_2024(
        df_eir: pd.DataFrame,
        masse_brut_mde: float,
        ratio_net_brut: float,
        eff_total_k: float,
        part_cc_pct: float,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Recalibre la distribution fine EIR (46 tranches de 100 €) sur la masse
        budgétaire 2024 des pensions (Comptes de la Protection Sociale DREES).
        """
        df = df_eir.copy()
        moyenne_eir = float((df["centre"] * df["pct_ensemble_all"] / 100.0).sum())
        masse_nette_cible_eur = masse_brut_mde * 1e9 * ratio_net_brut
        effectifs_tot = eff_total_k * 1000.0
        moyenne_cible_2024 = masse_nette_cible_eur / (12.0 * effectifs_tot)
        facteur = moyenne_cible_2024 / moyenne_eir

        df["borne_min_2024"] = df["borne_min"] * facteur
        df["borne_max_2024"] = df["borne_max"] * facteur
        df["centre_2024"] = df["centre"] * facteur
        df["effectifs_k"] = eff_total_k * (df["pct_ensemble_all"] / 100.0)

        # Effectif carrière complète borné par l'effectif total de chaque tranche
        eff_cc_brut = eff_total_k * (part_cc_pct / 100.0) * (df["pct_ensemble_cc"] / 100.0)
        df["effectifs_cc_k"] = np.minimum(df["effectifs_k"], eff_cc_brut)
        df["effectifs_incomp_k"] = np.maximum(0.0, df["effectifs_k"] - df["effectifs_cc_k"])
        df["masse_nette_mde"] = (df["effectifs_k"] * 1000.0 * df["centre_2024"] * 12.0) / 1e9
        return df, facteur

    def _charger_pyramide_eurostat(self) -> Tuple[str, pd.DataFrame]:
        eu = self._fetch_json("eurostat_pop_fr", URLS_API["eurostat_pop_fr"])
        time_labels = eu.get("dimension", {}).get("time", {}).get("category", {}).get("label", {})
        annee = next(iter(time_labels.values()), "2025")

        age_index = eu["dimension"]["age"]["category"]["index"]
        values = eu["value"]

        lignes = []
        for code, idx in age_index.items():
            if code in ("TOTAL", "UNK"):
                continue
            val = values.get(str(idx))
            if val is None:
                continue
            if code == "Y_LT1":
                age = 0
            elif code == "Y_OPEN":
                age = 100
            elif code.startswith("Y") and code[1:].isdigit():
                age = int(code[1:])
            else:
                continue
            lignes.append({"age": age, "population": float(val)})

        df_pop = pd.DataFrame(lignes).sort_values("age").reset_index(drop=True)
        return str(annee), df_pop


# ==============================================================================
# 4. OUTILS STATISTIQUES SUR DISTRIBUTIONS PONDÉRÉES
# ==============================================================================

def calculer_indicateurs_distribution(montants: np.ndarray, poids: np.ndarray) -> Dict[str, float]:
    """Calcule moyenne, quantiles (D1, Q1, Médiane, Q3, D9) et indice de Gini pondérés."""
    ordre = np.argsort(montants)
    x = montants[ordre]
    w = poids[ordre]
    mask = w > 0
    x, w = x[mask], w[mask]
    w_tot = np.sum(w)
    if w_tot <= 0:
        return {}

    moyenne = float(np.sum(x * w) / w_tot)
    cdf = np.cumsum(w) / w_tot

    def quantile(q: float) -> float:
        return float(np.interp(q, cdf, x))

    # Calcul exact du coefficient de Gini pour distribution discrète pondérée
    cum_w = np.concatenate([[0.0], np.cumsum(w) / w_tot])
    cum_xw = np.concatenate([[0.0], np.cumsum(x * w) / np.sum(x * w)])
    aire_lorenz = np.sum(0.5 * (cum_xw[:-1] + cum_xw[1:]) * (cum_w[1:] - cum_w[:-1]))
    gini = max(0.0, float(1.0 - 2.0 * aire_lorenz))

    d1 = quantile(0.10)
    d9 = quantile(0.90)
    return {
        "moyenne": moyenne,
        "d1": d1,
        "q1": quantile(0.25),
        "mediane": quantile(0.50),
        "q3": quantile(0.75),
        "d9": d9,
        "ratio_d9_d1": d9 / d1 if d1 > 0 else float("nan"),
        "gini": gini,
    }


def construire_micro_points(df_dist: pd.DataFrame, sous_echantillon: int = 20) -> Dict[str, np.ndarray]:
    """
    Transforme les 46 tranches (dont la tranche ouverte > 4500€ découpée en queue décroissante)
    en un nuage de points continu pondéré pour simuler précisément tout seuil Min/Max.
    """
    montants_list: List[float] = []
    w_cc_list: List[float] = []
    w_inc_list: List[float] = []

    for _, row in df_dist.iterrows():
        b_min = float(row["borne_min_2024"])
        b_max = float(row["borne_max_2024"])
        eff_cc = float(row["effectifs_cc_k"])
        eff_inc = float(row["effectifs_incomp_k"])

        if row["borne_min"] >= 4500.0:
            # Queue exponentielle décroissante sur 4 segments pour la tranche ouverte
            # préservant exactement la moyenne calibrée `centre_2024`
            cible = float(row["centre_2024"])
            segments = [(b_min, b_min * 1.12, 0.45), (b_min * 1.12, b_min * 1.28, 0.30),
                        (b_min * 1.28, b_min * 1.50, 0.17), (b_min * 1.50, b_min * 1.85, 0.08)]
            moy_raw = sum(0.5 * (lo + hi) * p for lo, hi, p in segments)
            echelle = cible / moy_raw
            for lo, hi, p in segments:
                pts = np.linspace(lo * echelle, hi * echelle, 5, endpoint=False) + (hi - lo) * echelle / 10.0
                montants_list.extend(pts.tolist())
                w_cc_list.extend([eff_cc * p / 5.0] * 5)
                w_inc_list.extend([eff_inc * p / 5.0] * 5)
        else:
            pas = (b_max - b_min) / sous_echantillon
            pts = np.linspace(b_min + 0.5 * pas, b_max - 0.5 * pas, sous_echantillon)
            montants_list.extend(pts.tolist())
            w_cc_list.extend([eff_cc / sous_echantillon] * sous_echantillon)
            w_inc_list.extend([eff_inc / sous_echantillon] * sous_echantillon)

    return {
        "montants": np.array(montants_list, dtype=float),
        "w_cc": np.array(w_cc_list, dtype=float),
        "w_incomp": np.array(w_inc_list, dtype=float),
    }


# ==============================================================================
# 5. PARTIE 1A : ANALYSE DESCRIPTIVE ("QUI TOUCHE COMBIEN, QUAND ?")
# ==============================================================================

def _chercher_caisse(df_caisses: pd.DataFrame, prefixe: str) -> Dict[str, float]:
    sub = df_caisses[df_caisses["caisse"].str.lower().str.startswith(prefixe.lower())]
    if sub.empty:
        return {"eff_k": float("nan"), "pension_caisse": float("nan"), "masse_dd_est": float("nan")}
    r = sub.iloc[0]
    return {
        "eff_k": float(r["effectifs_2022_k"]),
        "pension_caisse": float(r["pension_caisse_brut_eur"]),
        "masse_dd_est": float(r["masse_droit_direct_est_mde"]),
    }


def _chercher_regime_principal(df_reg: pd.DataFrame, prefixe: str) -> Dict[str, float]:
    sub = df_reg[df_reg["regime"].str.lower().str.startswith(prefixe.lower())]
    if sub.empty:
        return {"eff_princ_k": float("nan"), "part_pct": float("nan"), "tous": float("nan"), "cc": float("nan")}
    r = sub.iloc[0]
    return {
        "eff_princ_k": float(r.get("effectifs_k", float("nan"))),
        "part_pct": float(r.get("part_pct", float("nan"))),
        "tous": float(r["tous_ensemble"]),
        "cc": float(r["cc_ensemble"]),
    }


def construire_tableau_fonction_publique(donnees: DonneesRetraites) -> pd.DataFrame:
    """
    Construit un tableau comparatif dédié aux régimes de la Fonction Publique
    (État SRE civils & militaires, Territoriale & Hospitalière CNRACL, FSPOEIE,
    IRCANTEC et ERAFP) face au secteur privé (CNAV + Agirc-Arrco).
    """
    df_c = donnees.detail_caisses_2022
    df_r = donnees.pensions_par_regime
    df_cps = donnees.cps_regimes

    def cps_val(prefixe: str) -> str:
        sub = df_cps[df_cps["regime_nom"].str.lower().str.startswith(prefixe.lower())]
        return f"{sub.iloc[0]['montant_mde']:.2f} Md€" if not sub.empty else "—"

    spec = [
        ("Fonction publique civile de l'État (SRE)", "Fonction publique civile", "Fonctionnaires civils", "Régimes pour les agents de l'État", "64,65 Md€ (État civils+milit.)"),
        ("Fonction publique militaire de l'État (SRE)", "Fonction publique militaire", "Fonctionnaires militaires", "", "(inclus dans les 64,65 Md€ État)"),
        ("Territoriale & Hospitalière (CNRACL)", "CNRACL", "Fonctionnaires CNRACL", "", "Inclus ds « Autres rég. spéciaux » (38,89 Md€)"),
        ("Ouvriers des établissements de l'État (FSPOEIE)", "FSPOEIE", "", "", "Inclus ds « Autres rég. spéciaux » (38,89 Md€)"),
        ("Contractuels publics — Complémentaire (IRCANTEC)", "Ircantec", "", "Institution de retraite complémentaire", cps_val("Institution de retraite complémentaire")),
        ("Retraite additionnelle FP sur primes (ERAFP)", "ERAFP", "", "Régime additionnel de la fonction publique", cps_val("Régime additionnel de la fonction publique")),
        ("Comparatif Privé — Base (CNAV / ex-SSI)", "CNAV/SSI", "dont régime général", "Caisse nationale d’assurance vieillesse", cps_val("Caisse nationale d’assurance vieillesse")),
        ("Comparatif Privé — Complém. (Agirc-Arrco)", "Agirc-Arrco", "", "Association générale des institutions", cps_val("Association générale des institutions")),
    ]

    lignes = []
    for lib, pref_c, pref_r, _, note_cps in spec:
        c = _chercher_caisse(df_c, pref_c)
        rp = _chercher_regime_principal(df_r, pref_r) if pref_r else {
            "eff_princ_k": float("nan"), "part_pct": float("nan"), "tous": float("nan"), "cc": float("nan")
        }
        lignes.append(
            {
                "Regime / Caisse": lib,
                "Pensionnes caisse (k)": c["eff_k"],
                "Regime principal (k)": rp["eff_princ_k"],
                "Pension caisse (€ brut)": c["pension_caisse"],
                "Pension totale tous reg. (€ brut)": rp["tous"],
                "Carr. complete (€ brut)": rp["cc"],
                "Masse droit direct est. (Md€)": c["masse_dd_est"],
                "Enveloppe CPS 2024": note_cps,
            }
        )

    df_out = pd.DataFrame(lignes)

    # Calcul des sous-totaux et du TOTAL GÉNÉRAL des retraites en France
    total_vs = float(donnees.cps_prestations[donnees.cps_prestations["ps_code"] == "E11-2"]["montant_mde"].sum())
    total_dd_cps = float(donnees.cps_prestations[donnees.cps_prestations["ps_code"] == "E11-21.1"]["montant_mde"].sum())
    masse_fp_dd = float(df_out.iloc[:6]["Masse droit direct est. (Md€)"].sum(skipna=True))
    eff_fp_princ = float(df_out.iloc[:6]["Regime principal (k)"].sum(skipna=True))
    masse_prive_dd = float(df_out.iloc[6:8]["Masse droit direct est. (Md€)"].sum(skipna=True))
    eff_prive_princ = float(df_out.iloc[6:8]["Regime principal (k)"].sum(skipna=True))

    ens_row = _chercher_regime_principal(df_r, "Tous retraités")
    cnav_cps = cps_val("Caisse nationale d’assurance vieillesse")
    agirc_cps = cps_val("Association générale des institutions")

    synthese = pd.DataFrame(
        [
            {
                "Regime / Caisse": "Sous-total Sphère Fonction Publique (titulaires+contract.)",
                "Pensionnes caisse (k)": float("nan"),
                "Regime principal (k)": eff_fp_princ,
                "Pension caisse (€ brut)": float("nan"),
                "Pension totale tous reg. (€ brut)": float("nan"),
                "Carr. complete (€ brut)": float("nan"),
                "Masse droit direct est. (Md€)": masse_fp_dd,
                "Enveloppe CPS 2024": "~95 Md€ (État + CNRACL + IRCANTEC + ERAFP)",
            },
            {
                "Regime / Caisse": "Sous-total Secteur Privé (CNAV + Agirc-Arrco)",
                "Pensionnes caisse (k)": float("nan"),
                "Regime principal (k)": eff_prive_princ,
                "Pension caisse (€ brut)": float("nan"),
                "Pension totale tous reg. (€ brut)": float("nan"),
                "Carr. complete (€ brut)": float("nan"),
                "Masse droit direct est. (Md€)": masse_prive_dd,
                "Enveloppe CPS 2024": f"{cnav_cps} + {agirc_cps}",
            },
            {
                "Regime / Caisse": f"TOTAL GÉNÉRAL RETRAITES FRANCE (CPS {donnees.annee_cps})",
                "Pensionnes caisse (k)": donnees.effectif_total_retraites_k,
                "Regime principal (k)": donnees.effectif_droit_direct_k,
                "Pension caisse (€ brut)": float("nan"),
                "Pension totale tous reg. (€ brut)": ens_row["tous"],
                "Carr. complete (€ brut)": ens_row["cc"],
                "Masse droit direct est. (Md€)": total_dd_cps if total_dd_cps > 0 else donnees.masse_pensions_brut_mde,
                "Enveloppe CPS 2024": f"{donnees.masse_pensions_brut_mde:.2f} Md€ pensions ({total_vs:.2f} Md€ Vieill.-Survie)",
            },
        ]
    )
    return pd.concat([df_out, synthese], ignore_index=True)


def construire_ventilation_ministeres(donnees: DonneesRetraites) -> pd.DataFrame:
    """
    Ventile l'enveloppe officielle des pensions des agents de l'État (CPS DREES 2024 :
    64,65 Md€ pour le SRE civils + militaires + ex-L) par ministère d'origine, selon
    les clés de répartition du Projet de Loi de Finances / CAS Pensions.
    """
    df_cps = donnees.cps_regimes
    sub_etat = df_cps[df_cps["regime_nom"].str.lower().str.startswith("régimes pour les agents de l'état")]
    enveloppe_etat_mde = float(sub_etat.iloc[0]["montant_mde"]) if not sub_etat.empty else 64.65
    masse_tot_pensions = donnees.masse_pensions_brut_mde

    # Effectif total droit direct État (civils + militaires) observé dans EACR
    df_c = donnees.detail_caisses_2022
    eff_civ = _chercher_caisse(df_c, "Fonction publique civile")["eff_k"]
    eff_mil = _chercher_caisse(df_c, "Fonction publique militaire")["eff_k"]
    eff_etat_k = (0.0 if math.isnan(eff_civ) else eff_civ) + (0.0 if math.isnan(eff_mil) else eff_mil)
    if eff_etat_k <= 0:
        eff_etat_k = 2000.9

    # Structure indicative issue des documents budgétaires PLF / CAS Pensions
    repartition_plf = [
        ("Éducation nationale & Enseignement supérieur", 44.0, "~32 % du budget du ministère"),
        ("Intérieur (Police nationale, Gendarmerie, Préfectures)", 16.0, "~30 % du budget du ministère"),
        ("Armées (Militaires & personnels civils Défense)", 14.5, "~12 % du budget du ministère"),
        ("Anciens fonctionnaires La Poste & Orange (« ex-L »)", 8.0, "Contributions dédiées (hors budget État)"),
        ("Économie & Finances (DGFiP, Douanes, Insee)", 7.5, "Agents financiers & fiscaux"),
        ("Justice (Magistrats, greffes, pénitentiaire)", 4.3, "~21 % du budget du ministère"),
        ("Autres ministères (Écologie, Agriculture, Santé, Culture…)", 5.7, "Moyenne interministérielle ~13 %"),
    ]

    lignes = []
    for nom_min, part_etat_pct, note in repartition_plf:
        montant = enveloppe_etat_mde * (part_etat_pct / 100.0)
        lignes.append(
            {
                "Ministère / Périmètre d'origine": nom_min,
                "Dépense pensions (Md€/an)": montant,
                "Part des pensions État (%)": part_etat_pct,
                "Part du total retraites France (%)": 100.0 * montant / masse_tot_pensions,
                "Effectif estimé (k pensionnés)": eff_etat_k * (part_etat_pct / 100.0),
                "Repère budgétaire PLF": note,
            }
        )

    lignes.append(
        {
            "Ministère / Périmètre d'origine": f"TOTAL PENSIONS DE L'ÉTAT (CPS {donnees.annee_cps})",
            "Dépense pensions (Md€/an)": enveloppe_etat_mde,
            "Part des pensions État (%)": 100.0,
            "Part du total retraites France (%)": 100.0 * enveloppe_etat_mde / masse_tot_pensions,
            "Effectif estimé (k pensionnés)": eff_etat_k,
            "Repère budgétaire PLF": f"Inclus dans les {masse_tot_pensions:.2f} Md€ France",
        }
    )
    return pd.DataFrame(lignes)


def _titre(texte: str) -> None:
    ligne = "=" * 88
    print(f"\n{ligne}\n{texte}\n{ligne}")


def _sous_titre(texte: str) -> None:
    print(f"\n--- {texte} ---")


def afficher_analyse_descriptive(donnees: DonneesRetraites) -> None:
    """Affiche le diagnostic complet des sources API : Macro, Qui touche combien, Quand."""
    _titre("0. CONNEXION AUX SOURCES OFFICIELLES (APIs DREES & EUROSTAT)")
    for cle, source in donnees.provenance.items():
        print(f"  - {cle:<24s} : {source}")

    _titre(f"PARTIE 1A — QUI TOUCHE COMBIEN, ET QUAND ? (DONNÉES DREES {donnees.annee_cps})")

    # 1. Macroéconomie Vieillesse-Survie
    _sous_titre(f"1. Enveloppe macroéconomique Vieillesse-Survie ({donnees.annee_cps}, DREES CPS)")
    df_p = donnees.cps_prestations
    total_vs = float(df_p[df_p["ps_code"] == "E11-2"]["montant_mde"].sum())
    print(f"  Total protection sociale Vieillesse-Survie : {total_vs:6.1f} Md€/an")
    print(f"  Dont pensions strictes (Direct+Réversion+ASPA) : {donnees.masse_pensions_brut_mde:6.1f} Md€/an")

    niv3 = df_p[df_p["ps_niveau"].astype(str) == "3"].sort_values("montant_mde", ascending=False)
    print(f"\n  {'Poste de dépense (Niveau 3)':<55s} | {'Montant (Md€)':>13s} | {'Part (%)':>8s}")
    print("  " + "-" * 82)
    for _, row in niv3.iterrows():
        part = 100.0 * row["montant_mde"] / total_vs if total_vs else 0.0
        print(f"  {row['ps_lib'][:55]:<55s} | {row['montant_mde']:13.2f} | {part:7.1f} %")

    _sous_titre(f"2. Qui verse ? Répartition par grand régime / caisse ({donnees.annee_cps})")
    df_r = donnees.cps_regimes
    somme_reg = float(df_r["montant_mde"].sum())
    print(f"  {'Régime / Organisme':<58s} | {'Dépense (Md€)':>13s} | {'Part (%)':>8s}")
    print("  " + "-" * 85)
    for _, row in df_r.head(10).iterrows():
        part = 100.0 * row["montant_mde"] / somme_reg if somme_reg else 0.0
        print(f"  {row['regime_nom'][:58]:<58s} | {row['montant_mde']:13.2f} | {part:7.1f} %")

    # 2. Qui touche combien ?
    _sous_titre(
        f"3. Qui touche combien ? Distribution des pensions nettes (recalibrée {donnees.annee_cps})"
    )
    print(
        f"  Effectifs : {donnees.effectif_total_retraites_k/1000:.2f} M retraités totaux "
        f"(dont {donnees.effectif_droit_direct_k/1000:.2f} M de droit direct)."
    )
    print(
        f"  Calibration EIR 2020 -> {donnees.annee_cps} : facteur {donnees.facteur_revalo_2020_2024:.3f} "
        f"(enveloppe nette cible ≈ {donnees.masse_pensions_brut_mde * donnees.ratio_net_brut:.1f} Md€/an, "
        f"ratio net/brut = {donnees.ratio_net_brut:.1%})."
    )

    df_d = donnees.distrib_pensions_2024
    tranches_macro = [
        ("Moins de 800 €", 0, 800),
        ("800 à 1 200 €", 800, 1200),
        ("1 200 à 1 600 €", 1200, 1600),
        ("1 600 à 2 000 €", 1600, 2000),
        ("2 000 à 2 500 €", 2000, 2500),
        ("2 500 à 3 000 €", 2500, 3000),
        ("3 000 à 4 000 €", 3000, 4000),
        ("Plus de 4 000 €", 4000, 1e9),
    ]
    eff_tot = float(df_d["effectifs_k"].sum())
    masse_tot = float(df_d["masse_nette_mde"].sum())

    print(
        f"\n  {'Tranche mensuelle nette':<22s} | {'Retraités (k)':>13s} | {'% Pop':>7s} | "
        f"{'Masse (Md€/an)':>14s} | {'% Masse':>8s}"
    )
    print("  " + "-" * 74)
    for lbl, lo, hi in tranches_macro:
        sub = df_d[(df_d["centre_2024"] >= lo) & (df_d["centre_2024"] < hi)]
        eff = float(sub["effectifs_k"].sum())
        masse = float(sub["masse_nette_mde"].sum())
        print(
            f"  {lbl:<22s} | {eff:13,.0f} | {100*eff/eff_tot:6.1f}% | "
            f"{masse:14.1f} | {100*masse/masse_tot:7.1f}%"
        )

    micro = construire_micro_points(df_d)
    stats = calculer_indicateurs_distribution(micro["montants"], micro["w_cc"] + micro["w_incomp"])
    print(
        f"\n  Indicateurs d'inégalités ({donnees.annee_cps} net mensuel total) : "
        f"Moyenne = {stats['moyenne']:,.0f} € | Médiane = {stats['mediane']:,.0f} € | "
        f"D1 = {stats['d1']:,.0f} € | D9 = {stats['d9']:,.0f} € | "
        f"D9/D1 = {stats['ratio_d9_d1']:.2f} | Gini = {stats['gini']:.3f}"
    )

    _sous_titre("4. Écarts selon le régime d'affiliation et le sexe (brut droit direct, DREES)")
    print(
        f"  {'Régime principal':<44s} | {'Tous (€)':>8s} | {'Femmes':>8s} | {'Hommes':>8s} | {'Carr. compl.':>12s}"
    )
    print("  " + "-" * 90)
    for _, row in donnees.pensions_par_regime.head(11).iterrows():
        print(
            f"  {row['regime'][:44]:<44s} | {row['tous_ensemble']:8,.0f} | "
            f"{row['tous_femmes']:8,.0f} | {row['tous_hommes']:8,.0f} | {row['cc_ensemble']:12,.0f}"
        )

    _sous_titre("4 bis. Focus Fonction Publique & Dépense Totale Retraites")
    total_vs_fp = float(donnees.cps_prestations[donnees.cps_prestations["ps_code"] == "E11-2"]["montant_mde"].sum())
    print(
        f"  Dépense totale retraites ({donnees.annee_cps}) : {donnees.masse_pensions_brut_mde:.2f} Md€ de pensions strictes "
        f"| {total_vs_fp:.2f} Md€ Vieillesse-Survie (dont {donnees.effectif_total_retraites_k/1000:.2f} M retraités)."
    )
    df_fp = construire_tableau_fonction_publique(donnees)
    print(
        f"  {'Régime / Caisse':<54s} | {'Eff.(k)':>8s} | {'Caisse(€)':>9s} | {'Totale(€)':>9s} | {'Masse':>9s}"
    )
    print("  " + "-" * 101)
    for _, r in df_fp.iterrows():
        eff_s = f"{r['Pensionnes caisse (k)']:,.0f}" if not math.isnan(r["Pensionnes caisse (k)"]) else "—"
        caisse_s = f"{r['Pension caisse (€ brut)']:,.0f}" if not math.isnan(r["Pension caisse (€ brut)"]) else "—"
        tot_s = f"{r['Pension totale tous reg. (€ brut)']:,.0f}" if not math.isnan(r["Pension totale tous reg. (€ brut)"]) else "—"
        masse_s = f"{r['Masse droit direct est. (Md€)']:.1f} Md" if not math.isnan(r["Masse droit direct est. (Md€)"]) else "—"
        print(f"  {r['Regime / Caisse'][:54]:<54s} | {eff_s:>8s} | {caisse_s:>9s} | {tot_s:>9s} | {masse_s:>9s}")

    _sous_titre("4 ter. Ventilation indicative des retraites de l'État par ministère (64,65 Md€)")
    df_min = construire_ventilation_ministeres(donnees)
    print(f"  {'Ministère / Périmètre':<54s} | {'Dépense':>8s} | {'% État':>7s} | {'% France':>8s} | {'Eff.(k)':>7s}")
    print("  " + "-" * 96)
    for _, r in df_min.iterrows():
        nom_min = r["Ministère / Périmètre d'origine"]
        print(
            f"  {nom_min[:54]:<54s} | "
            f"{r['Dépense pensions (Md€/an)']:5.2f} Md | "
            f"{r['Part des pensions État (%)']:5.1f} % | "
            f"{r['Part du total retraites France (%)']:6.1f} % | "
            f"{r['Effectif estimé (k pensionnés)']:7,.0f}"
        )

    # 3. Quand ?
    _sous_titre("5. Quand partent les Français ? (Âge de départ & répartition des primo-liquidants)")
    df_h = donnees.age_depart_historique
    if not df_h.empty:
        first, last = df_h.iloc[0], df_h.iloc[-1]
        print(
            f"  Âge conjoncturel moyen de départ : {first['ensemble']:.2f} ans ({int(first['annee'])}) "
            f"-> {last['ensemble']:.2f} ans ({int(last['annee'])}) "
            f"[Femmes: {last['femme']:.2f}, Hommes: {last['homme']:.2f}]"
        )

    df_p_age = donnees.primo_liquidants_par_age
    tot_primo = float(df_p_age["ensemble"].sum())
    tranches_liq = [
        ("Avant 60 ans (carrières longues/spéc.)", 0, 59),
        ("60 - 61 ans", 60, 61),
        ("62 - 63 ans (âge légal / taux plein)", 62, 63),
        ("64 - 65 ans", 64, 65),
        ("66 - 67+ ans (taux plein auto.)", 66, 120),
    ]
    print(f"\n  {'Âge de liquidation (nouveaux retraités)':<40s} | {'Effectifs':>10s} | {'Part (%)':>8s}")
    print("  " + "-" * 65)
    for lbl, a_min, a_max in tranches_liq:
        sub = df_p_age[(df_p_age["age_num"] >= a_min) & (df_p_age["age_num"] <= a_max)]
        eff = float(sub["ensemble"].sum())
        print(f"  {lbl:<40s} | {eff:10,.0f} | {100*eff/tot_primo:7.1f}%")

    _sous_titre("6. Qui touche combien selon l'âge actuel ? (Retraités de droit direct)")
    df_pa = donnees.pensions_par_age
    tranches_act = [
        ("Moins de 60 ans", 0, 59),
        ("60 - 64 ans", 60, 64),
        ("65 - 69 ans", 65, 69),
        ("70 - 74 ans", 70, 74),
        ("75 - 79 ans", 75, 79),
        ("80 - 84 ans", 80, 84),
        ("85 ans et plus", 85, 120),
    ]
    print(
        f"  {'Tranche âge actuel':<18s} | {'Effectif (k)':>12s} | {'Ens. (€ brut)':>13s} | "
        f"{'Femmes (€)':>10s} | {'Hommes (€)':>10s} | {'Masse (Md€/an)':>14s}"
    )
    print("  " + "-" * 88)
    for lbl, a_min, a_max in tranches_act:
        sub = df_pa[(df_pa["age_num"] >= a_min) & (df_pa["age_num"] <= a_max)]
        w = sub["effectif_estime_k"].values
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            continue
        m_ens = float(np.average(sub["pension_brut_ensemble"], weights=w))
        m_fem = float(np.average(sub["pension_brut_femmes"], weights=w))
        m_hom = float(np.average(sub["pension_brut_hommes"], weights=w))
        masse = (w_sum * 1000.0 * m_ens * 12.0) / 1e9
        print(
            f"  {lbl:<18s} | {w_sum:12,.0f} | {m_ens:13,.0f} | {m_fem:10,.0f} | {m_hom:10,.0f} | {masse:14.1f}"
        )


# ==============================================================================
# 6. PARTIE 1B : SIMULATEUR PARAMÉTRABLE EN RÉPARTITION (MIN / MAX / ÉCRÊTEMENT)
# ==============================================================================

def simuler_reforme_repartition(
    micro: Dict[str, np.ndarray],
    params: ParametresReformeRepartition,
    ratio_net_brut: float = 0.915,
) -> Dict[str, Any]:
    """
    Applique une réforme paramétrique (plancher, plafond, écrêtement, revalorisation)
    sur le nuage pondéré de pensions nettes et calcule les effets budgétaires & redistributifs.
    """
    m0 = micro["montants"]
    w_cc = micro["w_cc"]
    w_inc = micro["w_incomp"]
    w_tot = w_cc + w_inc

    # Étape 1 : revalorisation globale éventuelle
    m_base = m0 * (1.0 + params.revalorisation_globale_pct / 100.0)

    # Étape 2 : écrêtement progressif puis plafond absolu (identique pour tous)
    m_apres_plafond = m_base.copy()
    if params.seuil_ecretement is not None and params.taux_ecretement_pct > 0:
        tau = params.taux_ecretement_pct / 100.0
        exces = np.maximum(0.0, m_apres_plafond - params.seuil_ecretement)
        m_apres_plafond = m_apres_plafond - tau * exces

    if params.pension_maximum is not None:
        m_apres_plafond = np.minimum(m_apres_plafond, params.pension_maximum)

    # Étape 3 : plancher minimum (appliqué soit aux carrières complètes seules, soit à tous)
    m_cc = m_apres_plafond.copy()
    m_inc = m_apres_plafond.copy()
    if params.pension_minimum is not None:
        m_cc = np.maximum(m_cc, params.pension_minimum)
        if not params.appliquer_min_carriere_complete_seule:
            m_inc = np.maximum(m_inc, params.pension_minimum)

    # Agrégation des masses en Md€/an (nets versés + équivalent brut public)
    def masse_an_mde(val_cc: np.ndarray, val_inc: np.ndarray) -> float:
        return float(np.sum((val_cc * w_cc + val_inc * w_inc) * 1000.0 * 12.0)) / 1e9

    masse_initiale_net = masse_an_mde(m0, m0)
    masse_revalo_net = masse_an_mde(m_base, m_base)
    masse_apres_plafond_net = masse_an_mde(m_apres_plafond, m_apres_plafond)
    masse_finale_net = masse_an_mde(m_cc, m_inc)

    cout_min_net = masse_finale_net - masse_apres_plafond_net
    economie_max_net = masse_revalo_net - masse_apres_plafond_net
    cout_revalo_net = masse_revalo_net - masse_initiale_net
    solde_net = masse_initiale_net - masse_finale_net  # > 0 => économie nette ; < 0 => déficit
    solde_brut = solde_net / ratio_net_brut

    # Bénéficiaires du minimum et contributeurs du maximum
    gain_min_cc = m_cc - m_apres_plafond
    gain_min_inc = m_inc - m_apres_plafond
    eff_gagnants_k = float(np.sum(w_cc[gain_min_cc > 1e-6]) + np.sum(w_inc[gain_min_inc > 1e-6]))

    perte_max = m_base - m_apres_plafond
    eff_perdants_k = float(np.sum(w_tot[perte_max > 1e-6]))

    # Nouveaux indicateurs d'inégalités post-réforme
    montants_post = np.concatenate([m_cc, m_inc])
    poids_post = np.concatenate([w_cc, w_inc])
    stats_post = calculer_indicateurs_distribution(montants_post, poids_post)
    stats_init = calculer_indicateurs_distribution(m0, w_tot)

    return {
        "nom": params.nom,
        "params": params,
        "eff_tot_k": float(np.sum(w_tot)),
        "eff_gagnants_k": eff_gagnants_k,
        "pct_gagnants": 100.0 * eff_gagnants_k / float(np.sum(w_tot)),
        "cout_min_net_mde": cout_min_net,
        "eff_perdants_k": eff_perdants_k,
        "pct_perdants": 100.0 * eff_perdants_k / float(np.sum(w_tot)),
        "economie_max_net_mde": economie_max_net,
        "cout_revalo_net_mde": cout_revalo_net,
        "solde_net_mde": solde_net,
        "solde_brut_mde": solde_brut,
        "pct_masse_impact": 100.0 * solde_net / masse_initiale_net,
        "moyenne_init": stats_init["moyenne"],
        "moyenne_post": stats_post["moyenne"],
        "mediane_post": stats_post["mediane"],
        "gini_init": stats_init["gini"],
        "gini_post": stats_post["gini"],
    }


def trouver_plafond_equilibre(
    micro: Dict[str, np.ndarray],
    pension_minimum: float,
    carriere_complete_seule: bool = False,
    ratio_net_brut: float = 0.915,
) -> Optional[float]:
    """Résout par dichotomie le plafond mensuel net qui autofinance exactement le minimum donné."""
    lo, hi = pension_minimum, float(np.max(micro["montants"]))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        res = simuler_reforme_repartition(
            micro,
            ParametresReformeRepartition(
                pension_minimum=pension_minimum,
                appliquer_min_carriere_complete_seule=carriere_complete_seule,
                pension_maximum=mid,
            ),
            ratio_net_brut=ratio_net_brut,
        )
        # Si solde_net > 0, le plafond est trop bas (trop d'économie) -> on peut relever le plafond
        if res["solde_net_mde"] > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def afficher_simulation_repartition(
    donnees: DonneesRetraites,
    params_perso: ParametresReformeRepartition,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Exécute et affiche le comparatif des réformes Min/Max + scénario utilisateur."""
    _titre("PARTIE 1B — SIMULATEUR PARAMÉTRABLE EN RÉPARTITION (MIN / MAX / ÉCRÊTEMENT)")
    micro = construire_micro_points(donnees.distrib_pensions_2024)

    scenarios = [
        params_perso,
        ParametresReformeRepartition(
            nom="Min 1 200 € (carrières complètes seules)",
            pension_minimum=1200.0,
            appliquer_min_carriere_complete_seule=True,
            pension_maximum=None,
        ),
        ParametresReformeRepartition(
            nom="Min 1 200 € universel / Max 3 000 €",
            pension_minimum=1200.0,
            appliquer_min_carriere_complete_seule=False,
            pension_maximum=3000.0,
        ),
        ParametresReformeRepartition(
            nom="SMIC retraite 1 400 € / Plafond 2 500 €",
            pension_minimum=1400.0,
            appliquer_min_carriere_complete_seule=False,
            pension_maximum=2500.0,
        ),
        ParametresReformeRepartition(
            nom="Écrêtement 50% > 2 500 € + Plafond 3 500 €",
            pension_minimum=1200.0,
            appliquer_min_carriere_complete_seule=True,
            pension_maximum=3500.0,
            seuil_ecretement=2500.0,
            taux_ecretement_pct=50.0,
        ),
    ]

    resultats = [
        simuler_reforme_repartition(micro, sc, ratio_net_brut=donnees.ratio_net_brut)
        for sc in scenarios
    ]
    res_perso = resultats[0]

    print(
        f"  {'Scénario':<42s} | {'Gagnants':>9s} | {'Coût Min':>9s} | "
        f"{'Perdants':>9s} | {'Écon Max':>9s} | {'Solde net':>10s} | {'Gini':>6s}"
    )
    print("  " + "-" * 108)
    for r in resultats:
        signe = "+" if r["solde_net_mde"] >= 0 else ""
        print(
            f"  {r['nom'][:42]:<42s} | "
            f"{r['pct_gagnants']:7.1f} % | "
            f"{r['cout_min_net_mde']:6.1f} Md | "
            f"{r['pct_perdants']:7.1f} % | "
            f"{r['economie_max_net_mde']:6.1f} Md | "
            f"{signe}{r['solde_net_mde']:6.1f} Md | "
            f"{r['gini_post']:.3f}"
        )

    # Détail complet du scénario utilisateur + solveur d'autofinancement
    _sous_titre(f"Détail du scénario paramétré : {params_perso.nom}")
    p_min_str = f"{params_perso.pension_minimum:,.0f} €/mois" if params_perso.pension_minimum else "Aucun"
    p_max_str = f"{params_perso.pension_maximum:,.0f} €/mois" if params_perso.pension_maximum else "Aucun"
    mode_min = "carrières complètes seules" if params_perso.appliquer_min_carriere_complete_seule else "universel (tous)"
    print(f"  - Plancher minimum : {p_min_str} ({mode_min})")
    print(f"  - Plafond maximum  : {p_max_str}")
    print(
        f"  - Bénéficiaires revalorisés : {res_perso['eff_gagnants_k']:,.0f} k retraités "
        f"({res_perso['pct_gagnants']:.1f} %) -> coût brut {res_perso['cout_min_net_mde']/donnees.ratio_net_brut:.2f} Md€/an"
    )
    print(
        f"  - Retraités plafonnés       : {res_perso['eff_perdants_k']:,.0f} k retraités "
        f"({res_perso['pct_perdants']:.1f} %) -> économie brute {res_perso['economie_max_net_mde']/donnees.ratio_net_brut:.2f} Md€/an"
    )
    etat = "EXCÉDENT" if res_perso["solde_net_mde"] >= 0 else "DÉFICIT SUPPLÉMENTAIRE"
    print(
        f"  - Solde budgétaire ({etat}) : {res_perso['solde_net_mde']:+.2f} Md€/an nets "
        f"({res_perso['solde_brut_mde']:+.2f} Md€/an bruts, soit {res_perso['pct_masse_impact']:+.2f} % de la masse)"
    )
    print(
        f"  - Redistribution : Moyenne {res_perso['moyenne_init']:,.0f} € -> {res_perso['moyenne_post']:,.0f} € | "
        f"Médiane -> {res_perso['mediane_post']:,.0f} € | Gini {res_perso['gini_init']:.3f} -> {res_perso['gini_post']:.3f}"
    )

    if params_perso.pension_minimum is not None:
        plafond_eq_univ = trouver_plafond_equilibre(
            micro, params_perso.pension_minimum, carriere_complete_seule=False, ratio_net_brut=donnees.ratio_net_brut
        )
        plafond_eq_cc = trouver_plafond_equilibre(
            micro, params_perso.pension_minimum, carriere_complete_seule=True, ratio_net_brut=donnees.ratio_net_brut
        )
        print(
            f"\n  [Solveur d'équilibre à 0 €] Pour financer un minimum à {params_perso.pension_minimum:,.0f} €/mois net :"
        )
        print(f"    * En version universelle        -> plafond max requis = {plafond_eq_univ:,.0f} €/mois net")
        print(f"    * En version carrières complètes -> plafond max requis = {plafond_eq_cc:,.0f} €/mois net")

    return micro, res_perso


# ==============================================================================
# 7. PARTIE 2 : MOTEUR DE TRANSITION VERS LA CAPITALISATION
# ==============================================================================

def facteur_rente_mensuelle(rendement_annuel_reel: float, duree_retraite_annees: float) -> float:
    """
    Calcule la rente viagère mensuelle (en €/mois, indexée sur l'inflation) versée
    par euro de capital accumulé à l'âge de la retraite.
    """
    if duree_retraite_annees <= 0:
        return 0.0
    if abs(rendement_annuel_reel) < 1e-8:
        return 1.0 / (12.0 * duree_retraite_annees)
    annuite_annuelle = rendement_annuel_reel / (1.0 - (1.0 + rendement_annuel_reel) ** (-duree_retraite_annees))
    return annuite_annuelle / 12.0


def part_capitalisation_annee(t: int, params: ParametresCapitalisation) -> float:
    """
    Part des cotisations affectée à la capitalisation lors de l'année t >= 0 après la réforme.
    Si duree_transition_annees == 0, bascule immédiate à (100 - X) %.
    """
    cible_capi = max(0.0, min(1.0, 1.0 - params.socle_repartition_pct / 100.0))
    if params.duree_transition_annees <= 0:
        return cible_capi
    rampe = min(1.0, (t + 1) / float(params.duree_transition_annees))
    return cible_capi * rampe


def simuler_cohorte_individuelle(
    age_actuel: int,
    params: ParametresCapitalisation,
    recevoir_dotation_naissance: Optional[bool] = None,
    salaire_net_override: Optional[float] = None,
    rendement_override_pct: Optional[float] = None,
) -> Dict[str, float]:
    """
    Simule la retraite finale (en € constants à la liquidation, en % du dernier salaire,
    et en équivalent-salaire d'aujourd'hui) pour un individu d'âge `age_actuel`.
    """
    r = (rendement_override_pct if rendement_override_pct is not None else params.rendement_annuel_reel_pct) / 100.0
    g = params.croissance_salaires_reelle_pct / 100.0
    taux_cot = params.taux_cotisation_retraite_pct / 100.0
    taux_remp_rep = params.taux_remplacement_repartition_pct / 100.0

    ratio_brut_net = params.salaire_brut_mensuel_ref / params.salaire_net_mensuel_ref
    s_net_0 = salaire_net_override if salaire_net_override is not None else params.salaire_net_mensuel_ref
    s_brut_0 = s_net_0 * ratio_brut_net

    E = params.age_entree_emploi
    R = params.age_retraite
    duree_carriere = max(1, R - E)

    # Cas d'un individu déjà à la retraite ou au-delà de l'âge de départ
    if age_actuel >= R:
        pension_rep = s_net_0 * taux_remp_rep
        return {
            "age_actuel": float(age_actuel),
            "annees_futures_cotis": 0.0,
            "part_effective_rep_pct": 100.0,
            "part_effective_capi_pct": 0.0,
            "capital_cotis_eur": 0.0,
            "capital_naissance_eur": 0.0,
            "capital_total_eur": 0.0,
            "pension_rep_eur": pension_rep,
            "rente_capi_cotis_eur": 0.0,
            "rente_naissance_eur": 0.0,
            "pension_totale_eur": pension_rep,
            "pension_statu_quo_eur": pension_rep,
            "gain_pct": 0.0,
            "gain_eur_mois": 0.0,
            "taux_remplacement_pct": params.taux_remplacement_repartition_pct,
            "equiv_aujourdhui_eur": pension_rep,
            "statu_quo_equiv_aujourdhui_eur": pension_rep,
        }

    # Salaire réel à la retraite (après `R - age_actuel` années de croissance réelle `g`)
    annees_jusqua_retraite = R - age_actuel
    s_net_final = s_net_0 * ((1.0 + g) ** annees_jusqua_retraite)
    pension_statu_quo = s_net_final * taux_remp_rep

    # Années déjà effectuées avant la réforme (100% acquises en répartition)
    annees_passees = max(0, age_actuel - E)
    age_debut_futur = max(age_actuel, E)

    somme_poids_rep = float(annees_passees)
    capital_cotis = 0.0

    for age_trav in range(age_debut_futur, R):
        t_reforme = age_trav - age_actuel  # Année calendaire après le lancement (t=0)
        alpha_t = part_capitalisation_annee(t_reforme, params)
        somme_poids_rep += (1.0 - alpha_t)

        # Salaire brut réel à cette année de carrière
        s_brut_t = s_brut_0 * ((1.0 + g) ** t_reforme)
        cotis_capi_t = 12.0 * s_brut_t * taux_cot * alpha_t

        # Capitalisation pendant les années restantes jusqu'à l'âge de la retraite
        horizon = R - age_trav - 0.5
        capital_cotis += cotis_capi_t * ((1.0 + r) ** horizon)

    part_rep = somme_poids_rep / duree_carriere
    part_capi = 1.0 - part_rep

    # Dotation à la naissance ("Baby Bond")
    if recevoir_dotation_naissance is None:
        if age_actuel == 0:
            recoit_bb = params.dotation_naissance_active
        elif age_actuel < 18:
            recoit_bb = params.dotation_naissance_active and params.donner_dotation_aux_enfants_actuels
        else:
            recoit_bb = False
    else:
        recoit_bb = recevoir_dotation_naissance and params.dotation_naissance_active

    capital_naissance = (
        params.montant_dotation_naissance * ((1.0 + r) ** annees_jusqua_retraite)
        if recoit_bb
        else 0.0
    )

    f_rente = facteur_rente_mensuelle(r, params.duree_retraite_annees) * params.ratio_net_brut_rente
    pension_rep = pension_statu_quo * part_rep
    rente_cotis = capital_cotis * f_rente
    rente_naissance = capital_naissance * f_rente
    pension_totale = pension_rep + rente_cotis + rente_naissance

    taux_remp_final = 100.0 * pension_totale / s_net_final
    equiv_today = s_net_0 * (pension_totale / s_net_final)
    statu_quo_today = s_net_0 * taux_remp_rep

    return {
        "age_actuel": float(age_actuel),
        "annees_futures_cotis": float(R - age_debut_futur),
        "part_effective_rep_pct": 100.0 * part_rep,
        "part_effective_capi_pct": 100.0 * part_capi,
        "capital_cotis_eur": capital_cotis,
        "capital_naissance_eur": capital_naissance,
        "capital_total_eur": capital_cotis + capital_naissance,
        "pension_rep_eur": pension_rep,
        "rente_capi_cotis_eur": rente_cotis,
        "rente_naissance_eur": rente_naissance,
        "pension_totale_eur": pension_totale,
        "pension_statu_quo_eur": pension_statu_quo,
        "gain_pct": 100.0 * (pension_totale / pension_statu_quo - 1.0),
        "gain_eur_mois": pension_totale - pension_statu_quo,
        "taux_remplacement_pct": taux_remp_final,
        "equiv_aujourdhui_eur": equiv_today,
        "statu_quo_equiv_aujourdhui_eur": statu_quo_today,
    }


def simuler_tranches_age_eurostat(
    df_pop: pd.DataFrame, params: ParametresCapitalisation
) -> pd.DataFrame:
    """Calcule l'issue de la réforme pour chaque tranche d'âge réelle de la population française."""
    tranches = [
        ("0 an (Nouveaux-nés)", 0, 0),
        ("1 - 14 ans (Enfants)", 1, 14),
        ("15 - 21 ans (Pré-actifs)", 15, 21),
        ("22 - 29 ans (Jeunes actifs)", 22, 29),
        ("30 - 39 ans", 30, 39),
        ("40 - 49 ans", 40, 49),
        ("50 - 59 ans", 50, 59),
        ("60 - 64 ans (Fin carrière)", 60, 64),
        ("65+ ans (Déjà retraités)", 65, 100),
    ]

    lignes = []
    for lbl, a_min, a_max in tranches:
        sub = df_pop[(df_pop["age"] >= a_min) & (df_pop["age"] <= a_max)]
        pop_tot = float(sub["population"].sum())
        if pop_tot <= 0:
            continue

        # Calcul pondéré par l'effectif exact de chaque année d'âge Eurostat
        sims = [simuler_cohorte_individuelle(int(r["age"]), params) for _, r in sub.iterrows()]
        poids = sub["population"].values

        ligne: Dict[str, Any] = {"tranche": lbl, "population_m": pop_tot / 1e6}
        cles_num = [k for k in sims[0].keys() if k != "age_actuel"]
        for k in cles_num:
            vals = np.array([s[k] for s in sims], dtype=float)
            ligne[k] = float(np.average(vals, weights=poids))
        lignes.append(ligne)

    return pd.DataFrame(lignes)


def simuler_trajectoire_macro_transition(
    donnees: DonneesRetraites,
    params: ParametresCapitalisation,
    horizon_annees: int = 65,
) -> pd.DataFrame:
    """
    Simule année par année (t = 0 .. horizon) les flux macroéconomiques de transition :
    - Cotisations détournées vers la capitalisation (Md€ constants)
    - Économies progressives sur les prestations du socle en répartition (Md€ constants)
    - Besoin de financement transitoire (« double paiement »)
    - Encours réel des fonds (suivi cohorte par cohorte avec décaissement des rentes à la retraite).
    """
    r = params.rendement_annuel_reel_pct / 100.0
    g = params.croissance_salaires_reelle_pct / 100.0
    df_pop = donnees.population_par_age
    naissances_an = float(df_pop[df_pop["age"] == 0]["population"].sum())
    base_masse_an = donnees.masse_pensions_brut_mde
    pib_2024_mde = 2920.0  # Ordre de grandeur PIB France 2024 (Insee) pour ratio % PIB

    duree_carriere = max(1, params.age_retraite - params.age_entree_emploi)
    duree_ret_int = max(1, int(round(params.duree_retraite_annees)))
    f_annuite_an = facteur_rente_mensuelle(r, float(duree_ret_int)) * 12.0

    # Suivi exact cohorte par cohorte : clé = année de naissance relative à t=0 (cohorte née à t=0 -> 0)
    # À t=0, les âges présents vont de 0 à 100 (cohortes nées entre -100 et 0)
    soldes_cohortes: Dict[int, float] = {(-a): 0.0 for a in range(101)}
    rentes_cohortes: Dict[int, float] = {}

    if params.dotation_naissance_active:
        cout_0 = (naissances_an * params.montant_dotation_naissance) / 1e9
        if params.donner_dotation_aux_enfants_actuels:
            for a in range(18):
                pop_a = float(df_pop[df_pop["age"] == a]["population"].sum())
                soldes_cohortes[-a] = (pop_a * params.montant_dotation_naissance) / 1e9
        else:
            soldes_cohortes[0] = cout_0

    stock_part_rep_moy = 1.0
    lignes = []

    for t in range(horizon_annees + 1):
        masse_ref_t = base_masse_an * ((1.0 + g) ** t)
        pib_t = pib_2024_mde * ((1.0 + g) ** t)
        alpha_t = part_capitalisation_annee(t, params)

        # Flux entrant dans les fonds de capitalisation des actifs
        flux_cotis_capi = masse_ref_t * alpha_t
        cotis_par_cohorte_active = flux_cotis_capi / duree_carriere

        # Dotation naissance pour la nouvelle génération née en t > 0 (t=0 déjà dotée ci-dessus)
        cout_bb_t = (naissances_an * params.montant_dotation_naissance) / 1e9 if params.dotation_naissance_active else 0.0
        if t > 0:
            soldes_cohortes[t] = cout_bb_t

        # Mise à jour de chaque cohorte vivante (de la naissance jusqu'à age_retraite + duree_ret_int)
        for naissance_yr in list(soldes_cohortes.keys()):
            age_t = t - naissance_yr
            if age_t < 0:
                continue
            if age_t < params.age_entree_emploi:
                if t > 0 or naissance_yr < 0:
                    soldes_cohortes[naissance_yr] *= (1.0 + r)
            elif age_t < params.age_retraite:
                soldes_cohortes[naissance_yr] = (
                    soldes_cohortes[naissance_yr] * (1.0 + r) + cotis_par_cohorte_active * ((1.0 + r) ** 0.5)
                )
            elif age_t < params.age_retraite + duree_ret_int:
                if naissance_yr not in rentes_cohortes:
                    rentes_cohortes[naissance_yr] = soldes_cohortes[naissance_yr] * f_annuite_an
                solde_apres = soldes_cohortes[naissance_yr] * (1.0 + r) - rentes_cohortes[naissance_yr]
                soldes_cohortes[naissance_yr] = max(0.0, solde_apres)
            else:
                soldes_cohortes[naissance_yr] = 0.0

        encours_total_mde = sum(soldes_cohortes.values())

        # Part du stock de retraités à l'instant t : renouvellement progressif des générations
        age_a_t0 = params.age_retraite - t
        sim_liquidant_t = simuler_cohorte_individuelle(age_a_t0, params, recevoir_dotation_naissance=False)
        part_rep_nouveau_retraite = sim_liquidant_t["part_effective_rep_pct"] / 100.0
        renouvellement = 1.0 / max(1.0, params.duree_retraite_annees)
        if t > 0:
            stock_part_rep_moy = (1.0 - renouvellement) * stock_part_rep_moy + renouvellement * part_rep_nouveau_retraite

        prestations_rep_versees = masse_ref_t * stock_part_rep_moy
        economie_prestations_rep = masse_ref_t - prestations_rep_versees
        besoin_financement_transition = flux_cotis_capi - economie_prestations_rep

        lignes.append(
            {
                "annee_t": t,
                "taux_capi_cotis_pct": 100.0 * alpha_t,
                "flux_cotis_capi_mde": flux_cotis_capi,
                "economie_rep_mde": economie_prestations_rep,
                "besoin_transition_mde": besoin_financement_transition,
                "cout_dotation_naissance_mde": cout_bb_t,
                "encours_fonds_total_mde": encours_total_mde,
                "encours_pct_pib": 100.0 * encours_total_mde / pib_t,
            }
        )

    return pd.DataFrame(lignes)


def afficher_simulation_capitalisation(
    donnees: DonneesRetraites, params: ParametresCapitalisation
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Affiche l'analyse complète du passage à la capitalisation (nouveaux arrivants, tranches d'âge, macro)."""
    _titre("PARTIE 2 — ANALYSE DU PASSAGE À LA RETRAITE PAR CAPITALISATION")
    capi_cible = 100.0 - params.socle_repartition_pct
    bb_txt = (
        f"OUI ({params.montant_dotation_naissance:,.0f} € à la naissance)"
        if params.dotation_naissance_active
        else "NON (0 € à la naissance)"
    )
    print(f"  Paramètres du scénario :")
    print(f"    - Durée de transition           : {params.duree_transition_annees} ans (0 = bascule immédiate)")
    print(f"    - Socle Répartition / Capital.  : {params.socle_repartition_pct:.0f} % Répartition / {capi_cible:.0f} % Capitalisation")
    print(f"    - Dotation à la naissance       : {bb_txt}")
    print(
        f"    - Hypothèses financières        : Rendement réel = {params.rendement_annuel_reel_pct:.1f} %/an | "
        f"Croissance salaires = {params.croissance_salaires_reelle_pct:.1f} %/an | "
        f"Cotis. = {params.taux_cotisation_retraite_pct:.1f} % brut"
    )
    print(
        f"    - Carrière de référence         : Salaire net = {params.salaire_net_mensuel_ref:,.0f} €/mois "
        f"(brut {params.salaire_brut_mensuel_ref:,.0f} €), {params.age_entree_emploi}->{params.age_retraite} ans, "
        f"espérance retraite = {params.duree_retraite_annees:.1f} ans"
    )

    # A. Focus Nouveaux Arrivants (Nouveau-né vs Nouvel Actif 22 ans)
    _sous_titre("A. Résultat détaillé pour les « Nouveaux Arrivants » (en € constants à la liquidation)")
    nouveau_ne_avec = simuler_cohorte_individuelle(0, params, recevoir_dotation_naissance=True)
    nouveau_ne_sans = simuler_cohorte_individuelle(0, params, recevoir_dotation_naissance=False)
    nouvel_actif_22 = simuler_cohorte_individuelle(params.age_entree_emploi, params, recevoir_dotation_naissance=False)

    profils_na = [
        ("Nouveau-né (0 an) — SANS dotation", nouveau_ne_sans),
        ("Nouveau-né (0 an) — AVEC dotation", nouveau_ne_avec),
        (f"Nouvel actif ({params.age_entree_emploi} ans aujourd'hui)", nouvel_actif_22),
    ]
    print(
        f"  {'Profil':<35s} | {'Capital 64a':>12s} | {'Socle Rép.':>10s} | {'Rente Capi':>10s} | "
        f"{'Rente Naiss.':>12s} | {'TOTAL (€)':>10s} | {'vs 100% Rép.':>12s} | {'Taux Remp.':>10s}"
    )
    print("  " + "-" * 126)
    for lbl, r in profils_na:
        print(
            f"  {lbl:<35s} | {r['capital_total_eur']:10,.0f} € | {r['pension_rep_eur']:8,.0f} € | "
            f"{r['rente_capi_cotis_eur']:8,.0f} € | {r['rente_naissance_eur']:10,.0f} € | "
            f"{r['pension_totale_eur']:8,.0f} € | {r['gain_pct']:+10.1f} % | {r['taux_remplacement_pct']:8.1f} %"
        )

    # Sensibilité au rendement réel et aux niveaux de salaire pour les nouveaux arrivants
    _sous_titre("A bis. Sensibilité par niveau de salaire & rendement (Nouvel actif 22 ans / Nouveau-né)")
    salaires_types = [
        ("SMIC (~1 426 € net)", 1426.0),
        ("Médian (~2 180 € net)", 2180.0),
        ("Moyen (~2 580 € net)", 2580.0),
        ("Cadre (~4 200 € net)", 4200.0),
    ]
    print(
        f"  {'Profil salarial':<23s} | {'Actif 22a (3.0%)':>16s} | {'Actif 22a (4.5%)':>16s} | "
        f"{'Actif 22a (6.0%)':>16s} | {'Nouv.-né (param.)':>18s}"
    )
    print("  " + "-" * 100)
    for lbl_s, s_net in salaires_types:
        r30 = simuler_cohorte_individuelle(params.age_entree_emploi, params, False, s_net, 3.0)
        r45 = simuler_cohorte_individuelle(params.age_entree_emploi, params, False, s_net, 4.5)
        r60 = simuler_cohorte_individuelle(params.age_entree_emploi, params, False, s_net, 6.0)
        rnn = simuler_cohorte_individuelle(0, params, None, s_net, params.rendement_annuel_reel_pct)
        print(
            f"  {lbl_s:<23s} | {r30['equiv_aujourdhui_eur']:6,.0f}€ ({r30['gain_pct']:+4.0f}%) | "
            f"{r45['equiv_aujourdhui_eur']:6,.0f}€ ({r45['gain_pct']:+4.0f}%) | "
            f"{r60['equiv_aujourdhui_eur']:6,.0f}€ ({r60['gain_pct']:+4.0f}%) | "
            f"{rnn['equiv_aujourdhui_eur']:7,.0f}€ ({rnn['gain_pct']:+5.0f}%)"
        )
    print("  (Montants exprimés en équivalent-salaire d'aujourd'hui pour comparaison directe au salaire net actuel.)")

    # B. Chaque tranche d'âge actuelle en France (Pyramide Eurostat)
    _sous_titre(f"B. Impact par tranche d'âge actuelle (Pyramide Eurostat {donnees.annee_eurostat})")
    df_tranches = simuler_tranches_age_eurostat(donnees.population_par_age, params)
    print(
        f"  {'Tranche d’âge actuelle':<27s} | {'Pop (M)':>7s} | {'%Rép/%Cap':>10s} | "
        f"{'Capital 64a':>11s} | {'StatuQuo':>9s} | {'Nouveau':>9s} | {'Écart':>9s} | {'Éq.auj.':>8s}"
    )
    print("  " + "-" * 107)
    for _, row in df_tranches.iterrows():
        mix = f"{row['part_effective_rep_pct']:.0f}/{row['part_effective_capi_pct']:.0f}"
        print(
            f"  {row['tranche']:<27s} | {row['population_m']:7.2f} | {mix:>10s} | "
            f"{row['capital_total_eur']:9,.0f} € | {row['pension_statu_quo_eur']:7,.0f} € | "
            f"{row['pension_totale_eur']:7,.0f} € | {row['gain_pct']:+7.1f} % | "
            f"{row['equiv_aujourdhui_eur']:6,.0f} €"
        )

    # C. Bilan macroéconomique de la transition
    _sous_titre("C. Équilibre macroéconomique de la transition (« double paiement » et encours)")
    df_macro = simuler_trajectoire_macro_transition(donnees, params, horizon_annees=65)
    jalons = [0, 5, 10, 15, 25, 35, 45, 65]
    print(
        f"  {'Horizon':>8s} | {'Part Capi':>9s} | {'Cotis -> Capi':>13s} | "
        f"{'Écon. Rép.':>11s} | {'Besoin Trans.':>13s} | {'Dot. Naiss.':>11s} | {'Encours Fonds':>13s} | {'% PIB':>7s}"
    )
    print("  " + "-" * 104)
    for t in jalons:
        r = df_macro[df_macro["annee_t"] == t].iloc[0]
        print(
            f"  t + {int(r['annee_t']):2d} a | {r['taux_capi_cotis_pct']:7.1f} % | "
            f"{r['flux_cotis_capi_mde']:10.1f} Md | {r['economie_rep_mde']:8.1f} Md | "
            f"{r['besoin_transition_mde']:+10.1f} Md | {r['cout_dotation_naissance_mde']:8.1f} Md | "
            f"{r['encours_fonds_total_mde']:10,.0f} Md | {r['encours_pct_pib']:5.0f} %"
        )
    pic_besoin = df_macro.loc[df_macro["besoin_transition_mde"].idxmax()]
    print(
        f"\n  Pic du besoin de financement transitoire en répartition : "
        f"+{pic_besoin['besoin_transition_mde']:.1f} Md€/an constants à t + {int(pic_besoin['annee_t'])} ans."
    )

    return df_tranches, df_macro


# ==============================================================================
# 8. VISUALISATION GRAPHIQUE
# ==============================================================================

def tracer_distribution_min_max(
    ax: plt.Axes,
    donnees: DonneesRetraites,
    params_rep: ParametresReformeRepartition,
    res_rep: Optional[Dict[str, Any]] = None,
    titre: Optional[str] = None,
) -> None:
    """
    Trace la distribution des pensions nettes avec :
    - fond vert clair à gauche pour la zone revalorisée + % correspondant,
    - fond rouge clair à droite pour la zone plafonnée + % correspondant.
    """
    df_d = donnees.distrib_pensions_2024
    if res_rep is None:
        micro = construire_micro_points(df_d)
        res_rep = simuler_reforme_repartition(micro, params_rep, ratio_net_brut=donnees.ratio_net_brut)

    x_max = 5800.0
    y_max = float(df_d["pct_ensemble_all"].max()) * 1.18

    # Zones colorées en arrière-plan
    if params_rep.pension_minimum:
        p_min = params_rep.pension_minimum
        ax.axvspan(0, p_min, color="#bbf7d0", alpha=0.55, zorder=0)
        ax.axvline(
            p_min,
            color="#15803d",
            linestyle="--",
            linewidth=2,
            zorder=4,
            label=f"Min simulé ({p_min:,.0f} €)",
        )
        mode_min = " (carr. compl.)" if params_rep.appliquer_min_carriere_complete_seule else ""
        ax.text(
            0.5 * p_min,
            y_max * 0.94,
            f"Revalorisés{mode_min}\n{res_rep['pct_gagnants']:.1f} %\n({res_rep['eff_gagnants_k']/1000:.2f} M)",
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="#14532d",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0fdf4", edgecolor="#16a34a", linewidth=1.1),
            zorder=5,
        )

    if params_rep.pension_maximum:
        p_max = params_rep.pension_maximum
        ax.axvspan(p_max, x_max, color="#fecaca", alpha=0.55, zorder=0)
        ax.axvline(
            p_max,
            color="#b91c1c",
            linestyle="--",
            linewidth=2,
            zorder=4,
            label=f"Max simulé ({p_max:,.0f} €)",
        )
        ax.text(
            min(x_max - 420.0, 0.5 * (p_max + x_max)),
            y_max * 0.54,
            f"Plafonnés\n{res_rep['pct_perdants']:.1f} %\n({res_rep['eff_perdants_k']/1000:.2f} M)",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#7f1d1d",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef2f2", edgecolor="#dc2626", linewidth=1.1),
            zorder=5,
        )

    ax.bar(
        df_d["centre_2024"],
        df_d["pct_ensemble_all"],
        width=90,
        color="#2b6cb0",
        alpha=0.88,
        zorder=3,
        label="Ensemble des retraités (%)",
    )
    ax.set_title(titre or f"1. Qui touche combien ? (Distribution nette calibrée {donnees.annee_cps})")
    ax.set_xlabel("Pension mensuelle nette totale (€)")
    ax.set_ylabel("Part des retraités (%)")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)


def generer_graphiques(
    donnees: DonneesRetraites,
    params_rep: ParametresReformeRepartition,
    params_cap: ParametresCapitalisation,
    df_tranches: pd.DataFrame,
    df_macro: pd.DataFrame,
    chemin_sortie: Path = OUTPUT_PNG,
    res_rep: Optional[Dict[str, Any]] = None,
) -> Path:
    """Produit une planche de 4 graphiques haute définition synthétisant les résultats."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), dpi=140)
    fig.suptitle(
        "Système des Retraites en France — Diagnostic (DREES/Eurostat), Réforme Min/Max & Transition Capitalisation",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 1. Distribution des pensions (Qui touche combien) + seuils Min/Max
    tracer_distribution_min_max(axes[0, 0], donnees, params_rep, res_rep=res_rep)

    # 2. Quand : Primo-liquidants et pension moyenne par âge
    ax2 = axes[0, 1]
    df_pr = donnees.primo_liquidants_par_age
    df_pa = donnees.pensions_par_age
    ax2.bar(
        df_pr["age_num"],
        df_pr["ensemble"] / 1000.0,
        color="#dd6b20",
        alpha=0.75,
        label="Nouveaux retraités (milliers, gauche)",
    )
    ax2.set_title("2. Quand part-on & que touche-t-on selon l'âge ? (DREES)")
    ax2.set_xlabel("Âge (ans)")
    ax2.set_ylabel("Nouveaux départs en 2022 (milliers)", color="#c05621")
    ax2.set_xlim(48, 92)

    ax2_r = ax2.twinx()
    ax2_r.plot(
        df_pa["age_num"],
        df_pa["pension_brut_ensemble"],
        color="#1a365d",
        linewidth=2.2,
        label="Pension brute moyenne (€/mois, droite)",
    )
    ax2_r.set_ylabel("Pension brute moyenne (€/mois)", color="#1a365d")
    ax2_r.grid(False)
    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2_r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=8.5)

    # 3. Impact de la capitalisation par tranche d'âge actuelle
    ax3 = axes[1, 0]
    x_idx = np.arange(len(df_tranches))
    w_bar = 0.36
    ax3.bar(
        x_idx - w_bar / 2,
        df_tranches["statu_quo_equiv_aujourdhui_eur"],
        width=w_bar,
        color="#718096",
        label="Statu quo 100 % Répartition",
    )
    ax3.bar(
        x_idx + w_bar / 2,
        df_tranches["equiv_aujourdhui_eur"],
        width=w_bar,
        color="#2f855a",
        label=f"Mixte (Socle {params_cap.socle_repartition_pct:.0f}% + Capi {params_cap.rendement_annuel_reel_pct:.1f}%)",
    )
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels([t.split(" (")[0] for t in df_tranches["tranche"]], rotation=28, ha="right", fontsize=8.5)
    ax3.set_title("3. Pension finale par tranche d'âge (équivalent salaire d'aujourd'hui)")
    ax3.set_ylabel("Pension mensuelle (€/mois)")
    ax3.legend(fontsize=8.5)

    # 4. Macroéconomie de la transition (« double paiement » vs économie)
    ax4 = axes[1, 1]
    ax4.plot(
        df_macro["annee_t"],
        df_macro["flux_cotis_capi_mde"],
        color="#3182ce",
        linewidth=2,
        label="Cotisations basculées en Capi (Md€/an)",
    )
    ax4.plot(
        df_macro["annee_t"],
        df_macro["economie_rep_mde"],
        color="#2f855a",
        linewidth=2,
        label="Économie sur pensions Répartition (Md€/an)",
    )
    ax4.fill_between(
        df_macro["annee_t"],
        df_macro["besoin_transition_mde"],
        0,
        where=(df_macro["besoin_transition_mde"] >= 0),
        color="#e53e3e",
        alpha=0.22,
        label="Besoin financement transitoire (Md€/an)",
    )
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_title(
        f"4. Trajectoire macroéconomique de transition ({params_cap.duree_transition_annees} ans)"
    )
    ax4.set_xlabel("Années après lancement de la réforme (t)")
    ax4.set_ylabel("Milliards € constants / an")
    ax4.legend(fontsize=8.5)

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    plt.savefig(chemin_sortie, dpi=150)
    plt.close(fig)
    return chemin_sortie


# ==============================================================================
# 9. INTERFACE LIGNE DE COMMANDE (CLI) & POINT D'ENTRÉE
# ==============================================================================

def parser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse API des retraites françaises, simulation Min/Max et passage à la capitalisation."
    )
    parser.add_argument("--refresh-cache", action="store_true", help="Force le re-téléchargement depuis les APIs.")
    parser.add_argument("--min-pension", type=float, default=1200.0, help="Pension minimum nette simulée (€/mois).")
    parser.add_argument("--max-pension", type=float, default=3000.0, help="Pension maximum nette simulée (€/mois).")
    parser.add_argument(
        "--min-carriere-complete",
        action="store_true",
        help="Réserve la pension minimum aux retraités ayant une carrière complète.",
    )
    parser.add_argument("--socle-pct", type=float, default=40.0, help="Part X%% maintenue en répartition à terme.")
    parser.add_argument(
        "--duree-transition", type=int, default=25, help="Nombre d'années de bascule vers la capitalisation (0=immédiat)."
    )
    parser.add_argument(
        "--dotation-naissance",
        type=float,
        default=10000.0,
        help="Montant versé à la naissance (€). Mettre 0 pour désactiver.",
    )
    parser.add_argument(
        "--dotation-enfants-actuels",
        action="store_true",
        help="Si activé, verse aussi la dotation aux mineurs (0-17 ans) déjà nés.",
    )
    parser.add_argument("--rendement", type=float, default=4.5, help="Rendement annuel réel de la capitalisation (%%).")
    parser.add_argument("--salaire-net", type=float, default=2580.0, help="Salaire net mensuel de référence (€).")
    return parser.parse_args()


def main() -> None:
    args = parser_arguments()

    connecteur = ConnecteurDonneesRetraites(refresh_cache=args.refresh_cache)
    donnees = connecteur.charger_tout()

    # Partie 1A : Qui touche combien, quand ?
    afficher_analyse_descriptive(donnees)

    # Partie 1B : Simulateur paramétrable en répartition (Min / Max)
    params_rep = ParametresReformeRepartition(
        nom=f"Paramétré (Min {args.min_pension:,.0f}€ / Max {args.max_pension:,.0f}€)",
        pension_minimum=args.min_pension if args.min_pension > 0 else None,
        appliquer_min_carriere_complete_seule=args.min_carriere_complete,
        pension_maximum=args.max_pension if args.max_pension > 0 else None,
    )
    _, res_rep = afficher_simulation_repartition(donnees, params_rep)

    # Partie 2 : Passage à la capitalisation
    ratio_brut_net_ref = 3300.0 / 2580.0
    params_cap = ParametresCapitalisation(
        duree_transition_annees=args.duree_transition,
        socle_repartition_pct=args.socle_pct,
        dotation_naissance_active=args.dotation_naissance > 0,
        montant_dotation_naissance=max(0.0, args.dotation_naissance),
        donner_dotation_aux_enfants_actuels=args.dotation_enfants_actuels,
        rendement_annuel_reel_pct=args.rendement,
        salaire_net_mensuel_ref=args.salaire_net,
        salaire_brut_mensuel_ref=args.salaire_net * ratio_brut_net_ref,
        ratio_net_brut_rente=donnees.ratio_net_brut,
    )
    df_tranches, df_macro = afficher_simulation_capitalisation(donnees, params_cap)

    # Export graphique
    fichier_png = generer_graphiques(donnees, params_rep, params_cap, df_tranches, df_macro, res_rep=res_rep)
    _titre("EXPORTS GÉNÉRÉS")
    print(f"  - Graphiques sauvegardés : {fichier_png}")
    print(f"  - Cache API local        : {CACHE_DIR}")


if __name__ == "__main__":
    main()
