#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparateur des programmes retraites (Présidentielle 2027) & Audit du plan David Lisnard
=======================================================================================

Cet outil réutilise le connecteur officiel DREES & Eurostat (`analyse_retraites.py`)
pour analyser de façon neutre, chiffrée et sourcée les propositions des principaux
courants et candidats sur les retraites :
  - David Lisnard (Nouvelle Énergie) — FOCUS & AUDIT DÉTAILLÉ
  - Édouard Philippe (Horizons)
  - Sarah Knafo / Éric Zemmour (Reconquête)
  - Les Républicains (LR / Bruno Retailleau)
  - Gabriel Attal (Renaissance / Bloc central)
  - Marine Le Pen / Jordan Bardella (Rassemblement National)
  - Parti Socialiste (PS)
  - La France Insoumise (LFI)

Principe méthodologique :
  - Chaque programme est modélisé d'après ses propositions publiques connues.
  - Les ordres de grandeur financiers sont systématiquement recalculés sur les
    données réelles DREES 2024 (406,02 Md€ de pensions, 17,89 M de retraités) et
    Eurostat 2025 (pyramide des âges).
  - Pour David Lisnard, un module d'audit spécifique évalue tranche par tranche
    le passage 60/40 (cible 116 Md€ en 8 tranches), la désindexation > 2 000 €/mois,
    le passage à 65 ans, le minimum à 1 000 € et l'option logement (-30 %).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
if "ipykernel" not in sys.modules and "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse_retraites import (
    ConnecteurDonneesRetraites,
    DonneesRetraites,
    ParametresCapitalisation,
    ParametresReformeRepartition,
    construire_micro_points,
    facteur_rente_mensuelle,
    simuler_cohorte_individuelle,
    simuler_reforme_repartition,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_COMP_PNG = Path(__file__).resolve().parent / "comparateur_programmes_graphiques.png"


# ==============================================================================
# 1. MODÈLE DES PROGRAMMES POLITIQUES & HYPOTHÈSES PARAMÉTRIQUES
# ==============================================================================

@dataclass
class ProgrammeCandidat:
    """Description structurée d'un programme politique sur les retraites."""
    candidat_parti: str
    bloc: str
    age_depart_cible: int
    duree_cotisation_ans: float
    part_repartition_pct: float
    part_capitalisation_pct: float
    duree_transition_capi_ans: int
    pension_minimum_visee: Optional[float]
    min_carriere_complete_seule: bool
    desindexation_seuil_eur: Optional[float]
    financement_principal: str
    # Impact annuel estimé (en Md€/an à horizon 10 ans, base DREES 2024)
    cout_mesures_age_et_min_mde_10a: float
    flux_capi_transition_mde_10a: float
    recettes_flechees_mde_10a: float
    # Répartition indicative de « où il pioche » (% de l'effort total)
    effort_actifs_age_pct: float
    effort_retraites_actuels_pct: float
    effort_entreprises_impots_pct: float
    effort_budget_etat_dette_pct: float
    note_methodologique: str


@dataclass
class ParametresAuditLisnard:
    """Paramètres modifiables pour auditer le plan de David Lisnard (Nouvelle Énergie)."""
    age_retraite: int = 65
    socle_repartition_pct: float = 60.0
    part_capitalisation_pct: float = 40.0
    duree_transition_ans: int = 15            # Durée totale de transition (15 ans demandé)
    nb_tranches: int = 8                      # 8 tranches prévues dans le projet Lisnard
    annees_par_tranche: float = 1.875         # 15 ans / 8 tranches = 1.875 an/tranche
    cible_capi_annuelle_mde: float = 116.0    # Chiffre affiché (40 % du socle contributif privé/base)
    seuil_desindexation_net_eur: float = 2000.0
    inflation_annuelle_pct: float = 1.8
    gel_complet_au_dessus_seuil: bool = True  # True = gel total (plancher 2000€) | False = gel tranche > 2000€
    pension_minimum_65a: float = 1000.0
    min_carriere_complete_seule: bool = True
    rendement_reel_capi_pct: float = 4.5
    retrait_immobilier_pct: float = 30.0      # Option résidence principale
    age_retrait_immobilier: int = 35
    economie_65ans_regime_croisiere_mde: float = 10.5  # Gain net du passage 64->65 ans vs loi 2023


def catalogue_programmes(donnees: DonneesRetraites) -> List[ProgrammeCandidat]:
    """
    Construit la liste des 8 principaux programmes présidentiels sur les retraites,
    avec des ordres de grandeur financiers recalés sur la masse DREES 2024 (406 Md€).
    """
    masse_tot = donnees.masse_pensions_brut_mde

    return [
        ProgrammeCandidat(
            candidat_parti="David Lisnard (Nouvelle Énergie)",
            bloc="Droite libérale",
            age_depart_cible=65,
            duree_cotisation_ans=43.0,
            part_repartition_pct=60.0,
            part_capitalisation_pct=40.0,
            duree_transition_capi_ans=8,
            pension_minimum_visee=1000.0,
            min_carriere_complete_seule=True,
            desindexation_seuil_eur=2000.0,
            financement_principal=(
                "Désindexation pensions > 2 000 € + âge à 65 ans + baisse dépenses publiques"
            ),
            cout_mesures_age_et_min_mde_10a=-9.5,   # -10.5 Md€ (65 ans) + ~1 Md€ (min 1000€)
            flux_capi_transition_mde_10a=116.0,     # Cible 8 tranches affichée (~116 Md€/an)
            recettes_flechees_mde_10a=21.0,         # Désindexation >2000€ (~20 Md€ à t+8 en gel total)
            effort_actifs_age_pct=25.0,
            effort_retraites_actuels_pct=25.0,
            effort_entreprises_impots_pct=0.0,
            effort_budget_etat_dette_pct=50.0,
            note_methodologique=(
                "Système 60/40 en 8 tranches (~116 Md€), socle 1 000 € à 65 ans, "
                "fonds souverain par défaut, déblocage 30 % pour résidence principale."
            ),
        ),
        ProgrammeCandidat(
            candidat_parti="Édouard Philippe (Horizons)",
            bloc="Centre droit",
            age_depart_cible=65,
            duree_cotisation_ans=45.0,
            part_repartition_pct=85.0,
            part_capitalisation_pct=15.0,
            duree_transition_capi_ans=20,
            pension_minimum_visee=1200.0,
            min_carriere_complete_seule=True,
            desindexation_seuil_eur=None,
            financement_principal=(
                "Âge légal à 65 ans + 45 annuités + abondement PER jeunes actifs"
            ),
            cout_mesures_age_et_min_mde_10a=-16.0,  # 65 ans + 45 annuités
            flux_capi_transition_mde_10a=0.15 * masse_tot * 0.5,  # À mi-transition (10 ans sur 20)
            recettes_flechees_mde_10a=0.0,
            effort_actifs_age_pct=65.0,
            effort_retraites_actuels_pct=5.0,
            effort_entreprises_impots_pct=10.0,
            effort_budget_etat_dette_pct=20.0,
            note_methodologique=(
                "Fourchette de 10 à 20 % de capitalisation (modélisé à 15 % sur 20 ans) "
                "complétée par un recul progressif à 65 ans et 45 annuités."
            ),
        ),
        ProgrammeCandidat(
            candidat_parti="Sarah Knafo / É. Zemmour (Reconquête)",
            bloc="Droite identitaire / libérale",
            age_depart_cible=64,
            duree_cotisation_ans=43.0,
            part_repartition_pct=0.0,
            part_capitalisation_pct=100.0,
            duree_transition_capi_ans=25,
            pension_minimum_visee=None,
            min_carriere_complete_seule=True,
            desindexation_seuil_eur=None,
            financement_principal=(
                "Plan d'économies publiques (140 Md€ affichés : administration, aides sociales)"
            ),
            cout_mesures_age_et_min_mde_10a=0.0,
            flux_capi_transition_mde_10a=masse_tot * (10.0 / 25.0),
            recettes_flechees_mde_10a=80.0,         # Fraction du plan d'économies fléchée transition
            effort_actifs_age_pct=10.0,
            effort_retraites_actuels_pct=0.0,
            effort_entreprises_impots_pct=0.0,
            effort_budget_etat_dette_pct=90.0,
            note_methodologique=(
                "Sortie complète de la répartition vers 100 % capitalisation sur 20-30 ans, "
                "maintien intégral des pensions actuelles financé par coupes budgétaires."
            ),
        ),
        ProgrammeCandidat(
            candidat_parti="Les Républicains (LR / B. Retailleau)",
            bloc="Droite républicaine",
            age_depart_cible=65,
            duree_cotisation_ans=43.0,
            part_repartition_pct=90.0,
            part_capitalisation_pct=10.0,
            duree_transition_capi_ans=15,
            pension_minimum_visee=1200.0,
            min_carriere_complete_seule=True,
            desindexation_seuil_eur=None,
            financement_principal=(
                "Âge légal à 64-65 ans + maîtrise dépense sociale + étage fonds souverain"
            ),
            cout_mesures_age_et_min_mde_10a=-10.0,
            flux_capi_transition_mde_10a=0.10 * masse_tot * (10.0 / 15.0),
            recettes_flechees_mde_10a=5.0,
            effort_actifs_age_pct=60.0,
            effort_retraites_actuels_pct=0.0,
            effort_entreprises_impots_pct=10.0,
            effort_budget_etat_dette_pct=30.0,
            note_methodologique=(
                "Priorité à l'équilibre de la répartition par allongement d'activité (65 ans) "
                "plus un étage complémentaire capitalisé (~10 %)."
            ),
        ),
        ProgrammeCandidat(
            candidat_parti="Gabriel Attal (Renaissance)",
            bloc="Bloc central",
            age_depart_cible=64,
            duree_cotisation_ans=43.0,
            part_repartition_pct=95.0,
            part_capitalisation_pct=5.0,
            duree_transition_capi_ans=15,
            pension_minimum_visee=1200.0,
            min_carriere_complete_seule=True,
            desindexation_seuil_eur=None,
            financement_principal=(
                "43 annuités (suppression de l'âge légal rigide) + PER incitatif"
            ),
            cout_mesures_age_et_min_mde_10a=0.0,    # Référence = loi 2023 actuelle
            flux_capi_transition_mde_10a=0.05 * masse_tot * (10.0 / 15.0),
            recettes_flechees_mde_10a=5.0,
            effort_actifs_age_pct=70.0,
            effort_retraites_actuels_pct=0.0,
            effort_entreprises_impots_pct=15.0,
            effort_budget_etat_dette_pct=15.0,
            note_methodologique=(
                "Maintien des 43 annuités (référence 64 ans) avec liberté de départ selon "
                "la durée cotisée (décote/surcote) et développement de l'épargne retraite volontaire."
            ),
        ),
        ProgrammeCandidat(
            candidat_parti="Marine Le Pen / J. Bardella (RN)",
            bloc="Rassemblement National",
            age_depart_cible=62,
            duree_cotisation_ans=42.0,
            part_repartition_pct=100.0,
            part_capitalisation_pct=0.0,
            duree_transition_capi_ans=0,
            pension_minimum_visee=1200.0,
            min_carriere_complete_seule=True,
            desindexation_seuil_eur=None,
            financement_principal=(
                "Économies immigration/UE/fraude + natalité et hausses de salaires"
            ),
            cout_mesures_age_et_min_mde_10a=18.0,   # Abrogation partielle 64->60/62 ans (~15-20 Md€)
            flux_capi_transition_mde_10a=0.0,
            recettes_flechees_mde_10a=15.0,         # Recettes affichées hors périmètre direct retraites
            effort_actifs_age_pct=10.0,
            effort_retraites_actuels_pct=0.0,
            effort_entreprises_impots_pct=10.0,
            effort_budget_etat_dette_pct=80.0,
            note_methodologique=(
                "100 % répartition ; départ dès 60 ans (40 annuités) pour les entrées avant 20 ans, "
                "62 ans en moyenne, financement affiché sur le budget général (immigration/UE)."
            ),
        ),
        ProgrammeCandidat(
            candidat_parti="Parti Socialiste (PS)",
            bloc="Gauche réformiste",
            age_depart_cible=62,
            duree_cotisation_ans=43.0,
            part_repartition_pct=100.0,
            part_capitalisation_pct=0.0,
            duree_transition_capi_ans=0,
            pension_minimum_visee=1300.0,
            min_carriere_complete_seule=True,
            desindexation_seuil_eur=None,
            financement_principal=(
                "Taxation hauts patrimoines/superprofits + niches fiscales + Fonds de réserve (FRR)"
            ),
            cout_mesures_age_et_min_mde_10a=16.0,   # Retour à 62 ans + revalo minima + pénibilité
            flux_capi_transition_mde_10a=0.0,
            recettes_flechees_mde_10a=14.0,
            effort_actifs_age_pct=10.0,
            effort_retraites_actuels_pct=0.0,
            effort_entreprises_impots_pct=70.0,
            effort_budget_etat_dette_pct=20.0,
            note_methodologique=(
                "100 % répartition ; retour à 62 ans (43 annuités modulées selon pénibilité), "
                "financement via fiscalité du capital et mobilisation du FRR."
            ),
        ),
        ProgrammeCandidat(
            candidat_parti="La France Insoumise (LFI)",
            bloc="Gauche radicale",
            age_depart_cible=60,
            duree_cotisation_ans=40.0,
            part_repartition_pct=100.0,
            part_capitalisation_pct=0.0,
            duree_transition_capi_ans=0,
            pension_minimum_visee=1426.0,
            min_carriere_complete_seule=True,
            desindexation_seuil_eur=None,
            financement_principal=(
                "Hausse cotisations patronales + fin exonérations + taxe revenus financiers"
            ),
            cout_mesures_age_et_min_mde_10a=36.0,   # Retour 60 ans & 40 annuités + min au SMIC
            flux_capi_transition_mde_10a=0.0,
            recettes_flechees_mde_10a=32.0,
            effort_actifs_age_pct=0.0,
            effort_retraites_actuels_pct=0.0,
            effort_entreprises_impots_pct=90.0,
            effort_budget_etat_dette_pct=10.0,
            note_methodologique=(
                "100 % répartition ; retour à 60 ans et 40 annuités, minimum carrière complète "
                "au SMIC (~1 426 € net), financé par cotisations et fiscalité sur le capital."
            ),
        ),
    ]


# ==============================================================================
# 2. AUDIT DÉTAILLÉ DU PLAN DAVID LISNARD (NOUVELLE ÉNERGIE)
# ==============================================================================

def analyser_assiette_desindexation(
    donnees: DonneesRetraites,
    seuil_net: float = 2000.0,
    inflation_pct: float = 1.8,
) -> Dict[str, Any]:
    """
    Mesure exactement sur la distribution DREES 2024 combien de retraités sont
    concernés par une désindexation au-dessus de `seuil_net` (ex: 2 000 €),
    la masse concernée et l'économie générée année par année selon deux modalités :
      - Gel total des pensions > seuil (avec clause plancher au seuil indexé)
      - Gel marginal (seule la fraction dépassant le seuil est désindexée).
    """
    micro = construire_micro_points(donnees.distrib_pensions_2024)
    m0 = micro["montants"]
    w = micro["w_cc"] + micro["w_incomp"]
    w_tot = float(np.sum(w))
    ratio_nb = donnees.ratio_net_brut

    mask_sup = m0 > seuil_net
    eff_sup_k = float(np.sum(w[mask_sup]))
    pct_retraites_touches = 100.0 * eff_sup_k / w_tot

    masse_tot_net_mde = float(np.sum(m0 * w * 1000.0 * 12.0)) / 1e9
    masse_sup_net_mde = float(np.sum(m0[mask_sup] * w[mask_sup] * 1000.0 * 12.0)) / 1e9
    masse_excedent_mde = float(np.sum((m0[mask_sup] - seuil_net) * w[mask_sup] * 1000.0 * 12.0)) / 1e9

    inf = inflation_pct / 100.0
    annees = []
    for t in range(1, 21):
        facteur_ero = (1.0 + inf) ** t
        # Modalité A : gel total avec plancher (personne > seuil ne passe sous le seuil réel)
        m_gel_total = np.where(mask_sup, np.maximum(seuil_net, m0 / facteur_ero), m0)
        eco_tot_net = float(np.sum((m0 - m_gel_total) * w * 1000.0 * 12.0)) / 1e9

        # Modalité B : gel uniquement de la tranche dépassant le seuil
        m_gel_marg = np.where(mask_sup, seuil_net + (m0 - seuil_net) / facteur_ero, m0)
        eco_marg_net = float(np.sum((m0 - m_gel_marg) * w * 1000.0 * 12.0)) / 1e9

        # Perte de pouvoir d'achat d'une pension élevée (ex. 2 800 € net) en gel total
        perte_2800_pct = 100.0 * (max(seuil_net, 2800.0 / facteur_ero) / 2800.0 - 1.0)

        annees.append(
            {
                "annee": t,
                "eco_gel_total_net_mde": eco_tot_net,
                "eco_gel_total_brut_mde": eco_tot_net / ratio_nb,
                "eco_gel_marginal_net_mde": eco_marg_net,
                "eco_gel_marginal_brut_mde": eco_marg_net / ratio_nb,
                "perte_pouvoir_achat_2800_pct": perte_2800_pct,
            }
        )

    return {
        "seuil_net": seuil_net,
        "inflation_pct": inflation_pct,
        "eff_touches_k": eff_sup_k,
        "pct_retraites_touches": pct_retraites_touches,
        "masse_totale_pensions_net_mde": masse_tot_net_mde,
        "masse_pensions_sup_net_mde": masse_sup_net_mde,
        "masse_pensions_sup_brut_mde": masse_sup_net_mde / ratio_nb,
        "pct_masse_concernee": 100.0 * masse_sup_net_mde / masse_tot_net_mde,
        "masse_excedent_net_mde": masse_excedent_mde,
        "trajectoire_annuelle": pd.DataFrame(annees),
    }


def simuler_tranches_lisnard(
    donnees: DonneesRetraites,
    params: ParametresAuditLisnard,
) -> Dict[str, Any]:
    """
    Simule la montée en charge en 8 tranches vers 40 % de capitalisation (116 Md€/an
    dans le chiffrage Lisnard, ou 40 % de la masse totale DREES 406 Md€ = 162,4 Md€)
    et confronte le besoin de financement aux recettes fléchées.
    """
    micro = construire_micro_points(donnees.distrib_pensions_2024)
    assiette = analyser_assiette_desindexation(
        donnees,
        seuil_net=params.seuil_desindexation_net_eur,
        inflation_pct=params.inflation_annuelle_pct,
    )
    df_desidx = assiette["trajectoire_annuelle"]

    # Coût du minimum à 1 000 € dès 65 ans (sur la distribution DREES)
    res_min_cc = simuler_reforme_repartition(
        micro,
        ParametresReformeRepartition(
            nom="Socle 1000€ (carr. compl.)",
            pension_minimum=params.pension_minimum_65a,
            appliquer_min_carriere_complete_seule=True,
            pension_maximum=None,
        ),
        ratio_net_brut=donnees.ratio_net_brut,
    )
    res_min_univ = simuler_reforme_repartition(
        micro,
        ParametresReformeRepartition(
            nom="Socle 1000€ (universel)",
            pension_minimum=params.pension_minimum_65a,
            appliquer_min_carriere_complete_seule=False,
            pension_maximum=None,
        ),
        ratio_net_brut=donnees.ratio_net_brut,
    )
    cout_min_brut_mde = (
        res_min_cc["cout_min_net_mde"]
        if params.min_carriere_complete_seule
        else res_min_univ["cout_min_net_mde"]
    ) / donnees.ratio_net_brut

    cible_40pct_totale_drees = donnees.masse_pensions_brut_mde * (params.part_capitalisation_pct / 100.0)

    lignes = []
    duree_totale = params.nb_tranches * params.annees_par_tranche
    duree_retraite = 20.5  # espérance de vie moyenne à 65 ans

    # Suivi simplifié des économies sur prestations répartition à mesure du renouvellement
    stock_part_capi_retraites = 0.0
    for k in range(1, params.nb_tranches + 1):
        annee_t = int(round(k * params.annees_par_tranche))
        frac_cible = k / float(params.nb_tranches)
        part_capi_pct = params.part_capitalisation_pct * frac_cible

        besoin_tranche_lisnard = params.cible_capi_annuelle_mde * frac_cible
        besoin_40pct_drees = cible_40pct_totale_drees * frac_cible

        # Recette désindexation à l'année t
        row_d = df_desidx[df_desidx["annee"] == min(20, max(1, annee_t))].iloc[0]
        recette_desidx = float(
            row_d["eco_gel_total_brut_mde"]
            if params.gel_complet_au_dessus_seuil
            else row_d["eco_gel_marginal_brut_mde"]
        )

        # Montée en charge du passage 64->65 ans (sur 4 ans à compter du lancement)
        rampe_65a = min(1.0, annee_t / 4.0)
        gain_net_65_et_min = params.economie_65ans_regime_croisiere_mde * rampe_65a - cout_min_brut_mde

        # Économie progressive sur le socle répartition (les nouveaux retraités ont peu capitalisé au début)
        part_capi_nouv_liq = part_capi_pct / 100.0 * min(1.0, (annee_t / 43.0) * 0.5)
        for _ in range(max(1, int(round(params.annees_par_tranche)))):
            stock_part_capi_retraites = (
                (1.0 - 1.0 / duree_retraite) * stock_part_capi_retraites
                + (1.0 / duree_retraite) * part_capi_nouv_liq
            )
        eco_socle_rep = donnees.masse_pensions_brut_mde * stock_part_capi_retraites

        total_ressources_propres = recette_desidx + gain_net_65_et_min + eco_socle_rep
        ecart_a_financer = besoin_tranche_lisnard - total_ressources_propres

        lignes.append(
            {
                "Tranche": f"Tranche {k}",
                "Année": annee_t,
                "% Capitalisation": part_capi_pct,
                "Flux Capi (cible 116 Md€)": besoin_tranche_lisnard,
                "Équiv. 40% total DREES": besoin_40pct_drees,
                "Désindexation > seuil": recette_desidx,
                "Solde 65 ans - Min 1000€": gain_net_65_et_min,
                "Écon. socle répartition": eco_socle_rep,
                "Total ressources propres": total_ressources_propres,
                "Reste à financer (Budget/Éco.)": ecart_a_financer,
                "Taux couverture propre (%)": 100.0 * total_ressources_propres / besoin_tranche_lisnard,
            }
        )

    return {
        "assiette": assiette,
        "cout_min_1000_cc_brut_mde": res_min_cc["cout_min_net_mde"] / donnees.ratio_net_brut,
        "cout_min_1000_univ_brut_mde": res_min_univ["cout_min_net_mde"] / donnees.ratio_net_brut,
        "cible_40pct_drees_mde": cible_40pct_totale_drees,
        "duree_totale_ans": duree_totale,
        "tableau_tranches": pd.DataFrame(lignes),
        "tableau_15_ans": simuler_lisnard_trajectoire_15ans(donnees, params, duree_annees=params.duree_transition_ans),
    }


def simuler_lisnard_trajectoire_15ans(
    donnees: DonneesRetraites,
    params: ParametresAuditLisnard,
    duree_annees: int = 15,
) -> pd.DataFrame:
    """
    Simule la trajectoire année par année (1 à 15 ans) du plan David Lisnard :
      - Progression de la capitalisation de 0 % à 40 % (flux annuel versé jusqu'à 116 Md€/an).
      - Ventilation détaillée des recettes et économies :
          * Désindexation des pensions > seuil (ex: 2 000 € net/mois)
          * Recul à 65 ans net du socle garanti à 1 000 € dès 65 ans
          * Économies acquises sur le socle par répartition au fur et à mesure du renouvellement
      - Reste à financer / Besoin annuel et cumulé à la charge du budget de l'État (coupes de dépenses).
    """
    assiette = analyser_assiette_desindexation(
        donnees,
        seuil_net=params.seuil_desindexation_net_eur,
        inflation_pct=params.inflation_annuelle_pct,
    )
    df_desidx = assiette["trajectoire_annuelle"]

    # Coût du socle 1 000 €
    micro = construire_micro_points(donnees.distrib_pensions_2024)
    res_min_cc = simuler_reforme_repartition(
        micro,
        ParametresReformeRepartition(
            nom="Socle 1000€ (carr. compl.)",
            pension_minimum=params.pension_minimum_65a,
            appliquer_min_carriere_complete_seule=True,
            pension_maximum=None,
        ),
        ratio_net_brut=donnees.ratio_net_brut,
    )
    res_min_univ = simuler_reforme_repartition(
        micro,
        ParametresReformeRepartition(
            nom="Socle 1000€ (universel)",
            pension_minimum=params.pension_minimum_65a,
            appliquer_min_carriere_complete_seule=False,
            pension_maximum=None,
        ),
        ratio_net_brut=donnees.ratio_net_brut,
    )
    cout_min_brut_mde = (
        res_min_cc["cout_min_net_mde"]
        if params.min_carriere_complete_seule
        else res_min_univ["cout_min_net_mde"]
    ) / donnees.ratio_net_brut

    duree_retraite = 20.5
    masse_tot = donnees.masse_pensions_brut_mde
    cible_40pct_drees = masse_tot * (params.part_capitalisation_pct / 100.0)

    lignes = []
    stock_part_capi_retraites = 0.0
    cumul_reste_a_financer = 0.0

    for t in range(1, duree_annees + 1):
        pct_avancement = t / float(duree_annees)
        part_capi_pct = params.part_capitalisation_pct * pct_avancement
        flux_capi_lisnard = params.cible_capi_annuelle_mde * pct_avancement
        flux_capi_drees = cible_40pct_drees * pct_avancement

        # Recette désindexation à l'année t
        row_d = df_desidx[df_desidx["annee"] == min(20, max(1, t))].iloc[0]
        recette_desidx = float(
            row_d["eco_gel_total_brut_mde"]
            if params.gel_complet_au_dessus_seuil
            else row_d["eco_gel_marginal_brut_mde"]
        )

        # 65 ans net du minimum 1000€ (montée en charge sur 4 ans)
        rampe_65a = min(1.0, t / 4.0)
        gain_net_65_et_min = params.economie_65ans_regime_croisiere_mde * rampe_65a - cout_min_brut_mde

        # Économies progressives sur le socle par répartition
        nouv_liq_capi = sum(
            params.part_capitalisation_pct * (u / float(duree_annees))
            for u in range(1, t + 1)
        ) / (43.0 * 100.0)
        stock_part_capi_retraites = (
            (1.0 - 1.0 / duree_retraite) * stock_part_capi_retraites
            + (1.0 / duree_retraite) * nouv_liq_capi
        )
        eco_socle_rep = masse_tot * stock_part_capi_retraites

        total_ressources_propres = recette_desidx + gain_net_65_et_min + eco_socle_rep
        reste_a_financer = flux_capi_lisnard - total_ressources_propres
        cumul_reste_a_financer += reste_a_financer

        taux_couverture = 100.0 * total_ressources_propres / flux_capi_lisnard if flux_capi_lisnard > 0 else 100.0
        tranche_num = min(8, int(np.ceil(pct_avancement * 8.0)))

        lignes.append(
            {
                "Année": t,
                "Tranche": f"T{tranche_num}",
                "% Capitalisation": part_capi_pct,
                "Flux Capi (cible 116 Md€)": flux_capi_lisnard,
                "Équiv. 40% total DREES": flux_capi_drees,
                "Désindexation > seuil": recette_desidx,
                "Solde 65 ans - Min 1000€": gain_net_65_et_min,
                "Écon. socle répartition": eco_socle_rep,
                "Total ressources propres": total_ressources_propres,
                "Besoin annuel budget État": reste_a_financer,
                "Besoin cumulé budget État": cumul_reste_a_financer,
                "Taux couverture propre (%)": taux_couverture,
            }
        )

    return pd.DataFrame(lignes)


def simuler_option_logement_lisnard(
    donnees: DonneesRetraites,
    params: ParametresAuditLisnard,
) -> pd.DataFrame:
    """
    Simule l'option du programme Lisnard permettant de débloquer `retrait_immobilier_pct`
    (30 %) du capital accumulé pour l'achat de la résidence principale à `age_retrait_immobilier`.
    """
    r = params.rendement_reel_capi_pct / 100.0
    g = 0.01  # 1 % de croissance réelle des salaires
    taux_cot = 0.278
    alpha = params.part_capitalisation_pct / 100.0
    age_entree = 22
    age_ret = params.age_retraite
    duree_ret = 20.5  # espérance de vie à 65 ans
    f_rente = facteur_rente_mensuelle(r, duree_ret) * donnees.ratio_net_brut

    profils = [
        ("SMIC (1 426 € net)", 1426.0),
        ("Salaire médian (2 180 € net)", 2180.0),
        ("Salaire moyen (2 580 € net)", 2580.0),
        ("Cadre (4 200 € net)", 4200.0),
    ]
    ratio_bn = 3300.0 / 2580.0

    lignes = []
    for lbl, s_net_0 in profils:
        s_brut_0 = s_net_0 * ratio_bn
        capital = 0.0
        apport_immo = 0.0
        capital_sans_retrait = 0.0

        for age in range(age_entree, age_ret):
            t = age - age_entree
            cotis = 12.0 * s_brut_0 * ((1.0 + g) ** t) * taux_cot * alpha
            capital = capital * (1.0 + r) + cotis * ((1.0 + r) ** 0.5)
            capital_sans_retrait = capital_sans_retrait * (1.0 + r) + cotis * ((1.0 + r) ** 0.5)
            if age == params.age_retrait_immobilier:
                retrait = capital * (params.retrait_immobilier_pct / 100.0)
                apport_immo = retrait
                capital -= retrait

        s_net_fin = s_net_0 * ((1.0 + g) ** (age_ret - age_entree))
        # Socle répartition (60 % d'une pension de référence à 74 % à 65 ans)
        socle_rep = s_net_fin * 0.74 * (params.socle_repartition_pct / 100.0)
        pension_sans_immo = socle_rep + capital_sans_retrait * f_rente
        pension_avec_immo = socle_rep + capital * f_rente

        facteur_actu = s_net_0 / s_net_fin
        lignes.append(
            {
                "Profil salarial": lbl,
                f"Apport débloqué à {params.age_retrait_immobilier} ans (€)": apport_immo,
                "Capital à 65 ans sans retrait": capital_sans_retrait * facteur_actu,
                "Pension 65a SANS retrait (éq. auj.)": pension_sans_immo * facteur_actu,
                "Pension 65a AVEC retrait 30% (éq. auj.)": pension_avec_immo * facteur_actu,
                "Baisse rente à 65a (€/mois)": (pension_avec_immo - pension_sans_immo) * facteur_actu,
            }
        )
    return pd.DataFrame(lignes)


# ==============================================================================
# 3. COMPARATEUR TRANSVERSAL DES 8 PROGRAMMES
# ==============================================================================

def simuler_comparatif_programmes(
    donnees: DonneesRetraites,
    programmes: List[ProgrammeCandidat],
    salaire_net_ref: float = 2580.0,
    rendement_reel_pct: float = 4.5,
) -> pd.DataFrame:
    """
    Simule l'issue individuelle (nouvel actif 22 ans & actif de 40 ans) et
    synthétise l'équilibre financier à 10 ans pour chaque programme.
    """
    lignes = []
    for prog in programmes:
        # Ajustement du taux de remplacement PAYG selon l'âge de départ / durée cotisée
        ecart_duree = prog.duree_cotisation_ans - 43.0
        taux_remp_rep = max(55.0, min(80.0, 72.0 + ecart_duree * 1.2 + (prog.age_depart_cible - 64) * 2.0))
        duree_retraite_adj = max(16.0, 21.5 - (prog.age_depart_cible - 64))

        params_cap = ParametresCapitalisation(
            duree_transition_annees=prog.duree_transition_capi_ans,
            socle_repartition_pct=prog.part_repartition_pct,
            dotation_naissance_active=False,
            rendement_annuel_reel_pct=rendement_reel_pct,
            taux_remplacement_repartition_pct=taux_remp_rep,
            age_retraite=prog.age_depart_cible,
            duree_retraite_annees=duree_retraite_adj,
            salaire_net_mensuel_ref=salaire_net_ref,
            salaire_brut_mensuel_ref=salaire_net_ref * (3300.0 / 2580.0),
            ratio_net_brut_rente=donnees.ratio_net_brut,
        )

        sim_22 = simuler_cohorte_individuelle(22, params_cap, False)
        sim_40 = simuler_cohorte_individuelle(40, params_cap, False)

        besoin_total_10a = prog.cout_mesures_age_et_min_mde_10a + prog.flux_capi_transition_mde_10a
        solde_net_10a = prog.recettes_flechees_mde_10a - besoin_total_10a

        lignes.append(
            {
                "Candidat / Programme": prog.candidat_parti,
                "Âge départ": prog.age_depart_cible,
                "Annuités": prog.duree_cotisation_ans,
                "Rép. / Capi.": f"{prog.part_repartition_pct:.0f} / {prog.part_capitalisation_pct:.0f} %",
                "Minimum (€)": prog.pension_minimum_visee if prog.pension_minimum_visee else float("nan"),
                "Où il pioche (financement)": prog.financement_principal,
                "Besoin brut t+10a (Md€)": besoin_total_10a,
                "Recettes fléchées t+10a (Md€)": prog.recettes_flechees_mde_10a,
                "Solde non couvert t+10a (Md€)": solde_net_10a,
                "Pension actif 22a (éq. auj.)": sim_22["equiv_aujourdhui_eur"],
                "Pension actif 40a (éq. auj.)": sim_40["equiv_aujourdhui_eur"],
                "Taux remp. 22a (%)": sim_22["taux_remplacement_pct"],
            }
        )
    return pd.DataFrame(lignes)


# ==============================================================================
# 4. VISUALISATION COMPARATIVE (4 PANNEAUX)
# ==============================================================================

def generer_graphiques_comparateur(
    donnees: DonneesRetraites,
    programmes: List[ProgrammeCandidat],
    df_comp: pd.DataFrame,
    audit_lisnard: Dict[str, Any],
    df_immo_lisnard: pd.DataFrame,
    chemin_sortie: Path = OUTPUT_COMP_PNG,
) -> Path:
    """Produit une planche graphique de 4 visualisations comparant les programmes et le plan Lisnard."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11), dpi=140)
    fig.suptitle(
        "Comparateur des Programmes Retraites (2027) & Audit chiffré du plan David Lisnard",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 1. Audit du plan Lisnard : Trajectoire sur 15 ans (0 à 40 % de capitalisation)
    ax1 = axes[0, 0]
    df_15 = audit_lisnard.get("tableau_15_ans", audit_lisnard["tableau_tranches"])
    x_15 = np.arange(len(df_15))
    w_bar = 0.38

    ax1.bar(
        x_15 - w_bar / 2,
        df_15["Flux Capi (cible 116 Md€)"],
        width=w_bar,
        color="#2b6cb0",
        label="Cotisations versées en Capi (0 → 40 %)",
    )
    # Décomposition empilée des recettes propres
    r_desidx = df_15["Désindexation > seuil"].values
    r_65min = df_15["Solde 65 ans - Min 1000€"].values
    r_econ = df_15["Écon. socle répartition"].values

    ax1.bar(x_15 + w_bar / 2, r_desidx, width=w_bar, color="#dd6b20", label="Désindexation > 2 000 €")
    ax1.bar(x_15 + w_bar / 2, r_65min, bottom=r_desidx, width=w_bar, color="#2f855a", label="Âge 65 ans - Min 1 000 €")
    ax1.bar(x_15 + w_bar / 2, r_econ, bottom=r_desidx + r_65min, width=w_bar, color="#319795", label="Écon. socle répartition")

    ax1.plot(
        x_15,
        df_15["Besoin annuel budget État"],
        color="#c53030",
        marker="o",
        linewidth=2.0,
        label="Écart annuel (budget de l'État)",
    )
    ax1.set_xticks(x_15)
    ax1.set_xticklabels([f"A{int(a)}\n({p:.0f}%)" for a, p in zip(df_15["Année"], df_15["% Capitalisation"])], fontsize=7.5)
    ax1.set_ylabel("Milliards € / an")
    ax1.set_title("1. Lisnard : trajectoire sur 15 ans (0 → 40 % Capi vs recettes propres)")
    ax1.legend(fontsize=7.5, loc="upper left")

    # 2. « Où ils piochent » : répartition de l'effort selon les programmes
    ax2 = axes[0, 1]
    noms_courts = [p.candidat_parti.split(" (")[0] for p in programmes]
    y_p = np.arange(len(programmes))
    actifs = np.array([p.effort_actifs_age_pct for p in programmes])
    retraites = np.array([p.effort_retraites_actuels_pct for p in programmes])
    entrep = np.array([p.effort_entreprises_impots_pct for p in programmes])
    budget = np.array([p.effort_budget_etat_dette_pct for p in programmes])

    ax2.barh(y_p, actifs, color="#3182ce", label="Actifs (âge / durée)")
    ax2.barh(y_p, retraites, left=actifs, color="#dd6b20", label="Retraités actuels (désindexation)")
    ax2.barh(y_p, entrep, left=actifs + retraites, color="#805ad5", label="Entreprises / Hauts patrimoines")
    ax2.barh(y_p, budget, left=actifs + retraites + entrep, color="#e53e3e", label="Budget État / Coupes / Dette")
    ax2.set_yticks(y_p)
    ax2.set_yticklabels(noms_courts, fontsize=8.5)
    ax2.set_ylim(len(programmes) + 1.4, -0.6)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Part indicative de l'effort (%)")
    ax2.set_title("2. « Où ils piochent » : qui finance chaque programme ?")
    ax2.legend(fontsize=7.8, loc="lower center", ncol=2, framealpha=0.95)

    # 3. Pension finale estimée pour un actif de 22 ans et de 40 ans
    ax3 = axes[1, 0]
    w_b = 0.36
    ax3.bar(
        y_p - w_b / 2,
        df_comp["Pension actif 22a (éq. auj.)"],
        width=w_b,
        color="#2f855a",
        label="Nouvel actif (22 ans aujourd'hui)",
    )
    ax3.bar(
        y_p + w_b / 2,
        df_comp["Pension actif 40a (éq. auj.)"],
        width=w_b,
        color="#718096",
        label="Actif de 40 ans aujourd'hui",
    )
    ax3.axhline(2580.0 * 0.72, color="#1a365d", linestyle="--", lw=1.4, label="Référence loi actuelle (~1 858 €)")
    ax3.set_xticks(y_p)
    ax3.set_xticklabels(
        [f"{n}\n({a}a)" for n, a in zip(noms_courts, df_comp["Âge départ"])],
        rotation=25,
        ha="right",
        fontsize=8,
    )
    ax3.set_ylabel("Pension mensuelle (€ équiv. salaire auj.)")
    ax3.set_title("3. Pension estimée au salaire moyen (2 580 € net, r=4,5 %)")
    ax3.legend(fontsize=8)

    # 4. Option logement (-30 % à 35 ans) dans le modèle Lisnard
    ax4 = axes[1, 1]
    x_im = np.arange(len(df_immo_lisnard))
    col_apport = [c for c in df_immo_lisnard.columns if "Apport" in c][0]
    ax4.bar(
        x_im - 0.18,
        df_immo_lisnard["Pension 65a SANS retrait (éq. auj.)"],
        width=0.35,
        color="#2b6cb0",
        label="Pension 65a SANS retrait immo (€/mois)",
    )
    ax4.bar(
        x_im + 0.18,
        df_immo_lisnard["Pension 65a AVEC retrait 30% (éq. auj.)"],
        width=0.35,
        color="#63b3ed",
        label="Pension 65a AVEC retrait 30% (€/mois)",
    )
    ax4_r = ax4.twinx()
    ax4_r.plot(
        x_im,
        df_immo_lisnard[col_apport] / 1000.0,
        color="#c53030",
        marker="s",
        lw=2.2,
        label="Apport logement à 35 ans (k€, droite)",
    )
    ax4_r.set_ylabel("Apport immobilier débloqué (k€)", color="#c53030")
    ax4_r.grid(False)
    ax4.set_xticks(x_im)
    ax4.set_xticklabels([p.split(" (")[0] for p in df_immo_lisnard["Profil salarial"]], fontsize=8.5)
    ax4.set_ylabel("Pension mensuelle 65a (€ équiv. auj.)")
    ax4.set_title("4. Lisnard : effet du déblocage 30 % pour le logement (à 35 ans)")
    l1, lb1 = ax4.get_legend_handles_labels()
    l2, lb2 = ax4_r.get_legend_handles_labels()
    ax4.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=7.8)

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    plt.savefig(chemin_sortie, dpi=150)
    plt.close(fig)
    return chemin_sortie


# ==============================================================================
# 5. AFFICHAGE CONSOLE & POINT D'ENTRÉE CLI
# ==============================================================================

def afficher_rapport_complet(
    donnees: DonneesRetraites,
    params_lisnard: ParametresAuditLisnard,
    rendement_pct: float = 4.5,
) -> None:
    programmes = catalogue_programmes(donnees)
    df_comp = simuler_comparatif_programmes(donnees, programmes, rendement_reel_pct=rendement_pct)
    audit = simuler_tranches_lisnard(donnees, params_lisnard)
    df_immo = simuler_option_logement_lisnard(donnees, params_lisnard)

    sep = "=" * 96
    print(f"\n{sep}\n1. COMPARATIF DES 8 PROGRAMMES SUR LES RETRAITES (BASE DREES {donnees.annee_cps} = {donnees.masse_pensions_brut_mde:.1f} Md€)\n{sep}")
    for _, r in df_comp.iterrows():
        print(
            f"  - {r['Candidat / Programme']:<38s} | {r['Âge départ']}a / {r['Annuités']:.0f} ann. | "
            f"{r['Rép. / Capi.']:<10s} | Solde 10a: {r['Solde non couvert t+10a (Md€)']:+6.1f} Md€ | "
            f"22a: {r['Pension actif 22a (éq. auj.)']:,.0f} € | 40a: {r['Pension actif 40a (éq. auj.)']:,.0f} €"
        )
        print(f"      Où il pioche : {r['Où il pioche (financement)']}")

    print(f"\n{sep}\n2. AUDIT CHIFFRÉ DU PLAN DAVID LISNARD (NOUVELLE ÉNERGIE — 60 % RÉP. / 40 % CAPI.)\n{sep}")
    asst = audit["assiette"]
    print(
        f"  A. Assiette désindexation (> {asst['seuil_net']:,.0f} € net/mois, inflation {asst['inflation_pct']:.1f} %/an) :\n"
        f"     - Retraités touchés : {asst['eff_touches_k']/1000:.2f} M ({asst['pct_retraites_touches']:.1f} % des retraités)\n"
        f"     - Masse brute de leurs pensions : {asst['masse_pensions_sup_brut_mde']:.1f} Md€/an "
        f"({asst['pct_masse_concernee']:.1f} % des pensions totales)\n"
        f"     - Coût du minimum 1 000 € à 65 ans : {audit['cout_min_1000_cc_brut_mde']:.2f} Md€/an (carr. compl.) "
        f"vs {audit['cout_min_1000_univ_brut_mde']:.2f} Md€/an (universel)\n"
        f"     - Note périmètre : 40 % de la masse totale DREES ({donnees.masse_pensions_brut_mde:.1f} Md€) = "
        f"{audit['cible_40pct_drees_mde']:.1f} Md€ (vs 116 Md€ ciblés sur le socle contributif direct)"
    )

    print("\n  B. Trajectoire détaillée année par année sur 15 ans (0 % → 40 % de capitalisation) :")
    df_15 = audit["tableau_15_ans"]
    print(
        f"  {'An':>3s} | {'% Cap':>5s} | {'Flux Capi':>10s} | {'Désindex.':>10s} | {'65a - 1k€':>10s} | "
        f"{'Écon Rép':>9s} | {'Tot. Propre':>12s} | {'Écart Budget':>13s} | {'Cumul Budget':>13s} | {'Couvert.':>8s}"
    )
    print("  " + "-" * 112)
    for _, r in df_15.iterrows():
        print(
            f"  {int(r['Année']):3d} | {r['% Capitalisation']:4.1f}% | "
            f"{r['Flux Capi (cible 116 Md€)']:7.1f} Md€ | "
            f"{r['Désindexation > seuil']:7.1f} Md€ | "
            f"{r['Solde 65 ans - Min 1000€']:+7.1f} Md€ | "
            f"{r['Écon. socle répartition']:6.1f} Md€ | "
            f"{r['Total ressources propres']:9.1f} Md€ | "
            f"{r['Besoin annuel budget État']:+10.1f} Md€ | "
            f"{r['Besoin cumulé budget État']:+10.1f} Md€ | "
            f"{r['Taux couverture propre (%)']:6.1f} %"
        )

    print("\n  C. Synthèse par tranche (8 tranches étalées sur 15 ans) :")
    df_t = audit["tableau_tranches"]
    print(
        f"  {'Tranche':<10s} | {'An':>3s} | {'Cible':>8s} | {'Désidx.':>8s} | {'65a-Min':>8s} | "
        f"{'ÉconRép':>8s} | {'Écart Budget':>12s} | {'Couvert.':>8s}"
    )
    print("  " + "-" * 85)
    for _, r in df_t.iterrows():
        print(
            f"  {r['Tranche']:<10s} | {int(r['Année']):3d} | "
            f"{r['Flux Capi (cible 116 Md€)']:6.1f}Md | "
            f"{r['Désindexation > seuil']:6.1f}Md | "
            f"{r['Solde 65 ans - Min 1000€']:+6.1f}Md | "
            f"{r['Écon. socle répartition']:6.1f}Md | "
            f"{r['Reste à financer (Budget/Éco.)']:+10.1f}Md | "
            f"{r['Taux couverture propre (%)']:6.1f} %"
        )

    print(f"\n  D. Option résidence principale (retrait de {params_lisnard.retrait_immobilier_pct:.0f} % du capital à {params_lisnard.age_retrait_immobilier} ans) :")
    col_app = [c for c in df_immo.columns if "Apport" in c][0]
    for _, r in df_immo.iterrows():
        print(
            f"     - {r['Profil salarial']:<28s} : Apport {r[col_app]:8,.0f} € | "
            f"Pension 65a : {r['Pension 65a SANS retrait (éq. auj.)']:,.0f} € -> "
            f"{r['Pension 65a AVEC retrait 30% (éq. auj.)']:,.0f} € ({r['Baisse rente à 65a (€/mois)']:+.0f} €/mois)"
        )

    png = generer_graphiques_comparateur(donnees, programmes, df_comp, audit, df_immo)
    print(f"\nGraphiques comparatifs sauvegardés : {png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparateur des programmes retraites & Audit Lisnard")
    parser.add_argument("--refresh-cache", action="store_true", help="Rafraîchir les APIs DREES/Eurostat.")
    parser.add_argument("--duree-transition", type=int, default=15, help="Durée totale de transition (ans, défaut 15).")
    parser.add_argument("--seuil-desidx", type=float, default=2000.0, help="Seuil désindexation Lisnard (€ net).")
    parser.add_argument("--inflation", type=float, default=1.8, help="Hypothèse d'inflation annuelle (%).")
    parser.add_argument("--annees-par-tranche", type=float, default=1.875, help="Durée d'une tranche Lisnard (ans, 15/8 = 1.875).")
    parser.add_argument("--gel-marginal", action="store_true", help="Geler uniquement la fraction > seuil.")
    parser.add_argument("--rendement", type=float, default=4.5, help="Rendement réel capitalisation (%).")
    args = parser.parse_args()

    donnees = ConnecteurDonneesRetraites(refresh_cache=args.refresh_cache).charger_tout()
    params_lisnard = ParametresAuditLisnard(
        duree_transition_ans=args.duree_transition,
        seuil_desindexation_net_eur=args.seuil_desidx,
        inflation_annuelle_pct=args.inflation,
        annees_par_tranche=args.annees_par_tranche,
        gel_complet_au_dessus_seuil=not args.gel_marginal,
        rendement_reel_capi_pct=args.rendement,
    )
    afficher_rapport_complet(donnees, params_lisnard, rendement_pct=args.rendement)


if __name__ == "__main__":
    main()
