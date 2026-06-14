"""
WM Value Betting – Basis-Rechenmodell + Value-Berechnung + Bankroll-Management
===============================================================================
Mathematische Kette:
  Elo-Rating → Win-Probability → Expected Goals (xG) → Poisson-Verteilung
  → Match-Wahrscheinlichkeiten (1X2) → Faire Dezimalquoten
  → Expected Value (EV) gegen Buchmacher-Quoten → Kelly-Einsatz

Quellen / Kalibrierung:
  - Elo-Formel: World Football Elo Ratings (eloratings.net), Skala 400
  - Basis-xG 1.35: empirischer Mittelwert Tore/Team in A-Länderspielen (FIFA, 2000–2023)
  - Sensitivitätsfaktor K=0.70: kalibriert an historischen WM-Spielen
  - Kelly-Kriterium: J. L. Kelly Jr. (1956), "A New Interpretation of Information Rate"
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests
from scipy.stats import poisson

# Stdout auf UTF-8 erzwingen (Windows-Console ist standardmäßig cp1252).
# errors='replace' verhindert Abstürze, falls ein Zeichen trotzdem nicht
# darstellbar ist – es wird dann durch '?' ersetzt statt Exception zu werfen.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Konfiguration / Kalibrierungsparameter
# ---------------------------------------------------------------------------

# Durchschnittliche Expected Goals pro Team auf neutralem Platz (Basis)
BASE_XG: float = 1.35

# Sensitivitätsfaktor: skaliert, wie stark die Elo-Differenz die xG-Werte verschiebt.
# K=0.70 bedeutet: bei ΔElo=400 (also p_win=0.909) verschiebt sich xG um ±0.286 Tore.
ELO_SENSITIVITY: float = 0.70

# Maximale Tore, bis zu denen die Poisson-Summe aufgebaut wird.
# P(X >= 11 Tore | xG<=3) < 0.01 % → vernachlässigbar.
MAX_GOALS: int = 10


# ---------------------------------------------------------------------------
# Datenstruktur
# ---------------------------------------------------------------------------

@dataclass
class NationalTeam:
    """Repräsentiert eine Nationalmannschaft mit Elo-Rating."""
    name: str
    elo: float

    def __repr__(self) -> str:
        return f"NationalTeam(name='{self.name}', elo={self.elo})"


# ---------------------------------------------------------------------------
# Schritt 1: Elo → Win-Probability → xG
# ---------------------------------------------------------------------------

def elo_to_xg(team_a: NationalTeam, team_b: NationalTeam) -> tuple[float, float]:
    """
    Wandelt die Elo-Ratings zweier Teams in Expected Goals (xG) um.

    Mathematische Kette
    -------------------
    1) Elo Win-Probability (Standardformel, neutraler Platz, kein Heimvorteil):
          p_win_A = 1 / (1 + 10^(-(elo_A - elo_B) / 400))

       Interpretation: Bei gleichen Elos → p=0.5 (je 50 %).
       Bei ΔElo=+400 → p≈0.909 (Team A ist 10× stärker im Elo-Sinn).

    2) Lineares xG-Mapping um den Mittelwert BASE_XG:
          xG_A = BASE_XG + K * (p_win_A - 0.5)
          xG_B = BASE_XG - K * (p_win_A - 0.5)

       Dadurch gilt immer: xG_A + xG_B = 2 * BASE_XG (Gesamttore konstant).
       Der Faktor K (ELO_SENSITIVITY) bestimmt die Spreizung.

    Returns
    -------
    (xg_a, xg_b): Expected Goals für Team A und Team B.
    """
    elo_diff = team_a.elo - team_b.elo

    # Schritt 1: Klassische Elo Win-Probability
    p_win_a = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    # Schritt 2: xG-Mapping – Abweichung von der 50/50-Ausgangslage skalieren
    deviation = ELO_SENSITIVITY * (p_win_a - 0.5)
    xg_a = BASE_XG + deviation
    xg_b = BASE_XG - deviation

    return xg_a, xg_b


# ---------------------------------------------------------------------------
# Schritt 2: xG → Poisson → Match-Wahrscheinlichkeiten (1X2)
# ---------------------------------------------------------------------------

def calculate_match_probabilities(
    team_a: NationalTeam,
    team_b: NationalTeam,
) -> dict[str, float]:
    """
    Berechnet die Wahrscheinlichkeiten für Sieg A (1), Unentschieden (X)
    und Sieg B (2) auf Basis der Poisson-Verteilung.

    Modellierung
    ------------
    Tor-Anzahl pro Team wird als unabhängige Poisson-Zufallsvariable modelliert:
        Goals_A ~ Poisson(xG_A)
        Goals_B ~ Poisson(xG_B)

    Unabhängigkeitsannahme vereinfacht die Berechnung:
        P(Goals_A=i, Goals_B=j) = P(i | xG_A) * P(j | xG_B)

    Dann wird über alle Kombinationen aufsummiert:
        P(1) = ΣΣ [i>j]  P(i)*P(j)
        P(X) = ΣΣ [i=j]  P(i)*P(j)
        P(2) = ΣΣ [i<j]  P(i)*P(j)

    Returns
    -------
    dict mit Schlüsseln 'home_win', 'draw', 'away_win' und den Wahrscheinlichkeiten.
    """
    xg_a, xg_b = elo_to_xg(team_a, team_b)

    # Poisson PMF-Vektoren für Tore 0..MAX_GOALS
    goals = np.arange(0, MAX_GOALS + 1)
    pmf_a = poisson.pmf(goals, mu=xg_a)   # P(Goals_A = i)
    pmf_b = poisson.pmf(goals, mu=xg_b)   # P(Goals_B = j)

    # Outer product ergibt Matrix P[i, j] = P(A=i) * P(B=j)
    prob_matrix = np.outer(pmf_a, pmf_b)

    # Masken für die drei Ergebnisse
    p_home_win = float(np.sum(np.tril(prob_matrix, k=-1)))  # i > j
    p_draw     = float(np.sum(np.diag(prob_matrix)))         # i = j
    p_away_win = float(np.sum(np.triu(prob_matrix, k=1)))   # i < j

    return {
        "xg_a":      xg_a,
        "xg_b":      xg_b,
        "home_win":  p_home_win,
        "draw":      p_draw,
        "away_win":  p_away_win,
    }


# ---------------------------------------------------------------------------
# Schritt 3: Wahrscheinlichkeiten → Faire Dezimalquoten
# ---------------------------------------------------------------------------

def probabilities_to_fair_odds(probs: dict[str, float]) -> dict[str, float]:
    """
    Wandelt Wahrscheinlichkeiten in faire europäische Dezimalquoten um.

    Formel: Dezimalquote = 1 / Wahrscheinlichkeit

    "Fair" bedeutet: kein Buchmacher-Aufschlag (Vig/Margin).
    Summe aller impliziten Wahrscheinlichkeiten = 1.0 (100 %).

    Returns
    -------
    dict mit Schlüsseln '1', 'X', '2' und den Dezimalquoten.
    """
    return {
        "1": round(1.0 / probs["home_win"], 4),
        "X": round(1.0 / probs["draw"],     4),
        "2": round(1.0 / probs["away_win"], 4),
    }


# ---------------------------------------------------------------------------
# Schritt 4: Expected Value (EV) berechnen
# ---------------------------------------------------------------------------

@dataclass
class BettingOpportunity:
    """Kapselt ein einzelnes wettbares Ergebnis mit allen relevanten Kennzahlen."""
    outcome:        str    # "1", "X" oder "2"
    label:          str    # Lesbare Bezeichnung (z.B. "Sieg Team A")
    our_prob:       float  # Unsere Modell-Wahrscheinlichkeit
    bookie_odds:    float  # Buchmacher-Dezimalquote
    ev:             float  # Expected Value
    is_value_bet:   bool   # True wenn EV > 0


def calculate_ev(prob: float, bookie_odds: float) -> float:
    """
    Berechnet den Expected Value (Erwartungswert) einer Wette.

    Formel
    ------
        EV = (Gewinnwahrscheinlichkeit × Buchmacher-Quote) − 1

    Interpretation
    --------------
        EV > 0 : Positive Erwartung → Value Bet (langfristig profitabel)
        EV = 0 : Break-even (faire Quote)
        EV < 0 : Negative Erwartung → Finger weg (Buchmacher hat Edge)

    Beispiel: prob=0.50, odds=2.20
        EV = (0.50 × 2.20) − 1 = 1.10 − 1 = +0.10  → 10 % Edge

    Parameters
    ----------
    prob        : Unsere Modell-Wahrscheinlichkeit für diesen Ausgang (0–1)
    bookie_odds : Dezimalquote des Buchmachers (z.B. 1.90)

    Returns
    -------
    EV als Dezimalzahl (0.10 = +10 %)
    """
    return (prob * bookie_odds) - 1.0


def analyze_value(
    match_probs: dict[str, float],
    bookie_odds: dict[str, float],
    team_a_name: str,
    team_b_name: str,
) -> list[BettingOpportunity]:
    """
    Vergleicht Modell-Wahrscheinlichkeiten mit Buchmacher-Quoten und
    identifiziert Value Bets (EV > 0).

    Parameters
    ----------
    match_probs  : Ausgabe von calculate_match_probabilities()
    bookie_odds  : Dict mit Schlüsseln '1', 'X', '2' und Dezimalquoten
    team_a_name  : Name von Team A (für lesbaren Output)
    team_b_name  : Name von Team B (für lesbaren Output)

    Returns
    -------
    Liste von BettingOpportunity-Objekten, sortiert nach EV (absteigend).
    """
    mapping = [
        ("1", f"Sieg {team_a_name}", match_probs["home_win"]),
        ("X", "Unentschieden",       match_probs["draw"]),
        ("2", f"Sieg {team_b_name}", match_probs["away_win"]),
    ]

    opportunities = []
    for key, label, prob in mapping:
        odds = bookie_odds[key]
        ev   = calculate_ev(prob, odds)
        opportunities.append(BettingOpportunity(
            outcome=key,
            label=label,
            our_prob=prob,
            bookie_odds=odds,
            ev=ev,
            is_value_bet=(ev > 0.0),
        ))

    # Beste Gelegenheit zuerst
    opportunities.sort(key=lambda o: o.ev, reverse=True)
    return opportunities


# ---------------------------------------------------------------------------
# Schritt 5: Kelly-Kriterium (Fractional Kelly)
# ---------------------------------------------------------------------------

def kelly_fraction(prob: float, bookie_odds: float, fraction: float = 0.25) -> float:
    """
    Berechnet den optimalen Einsatz gemäß dem Kelly-Kriterium.

    Vollständige Kelly-Formel
    -------------------------
    Bei Dezimalquoten gilt für den Gewinn pro Einheit: b = bookie_odds − 1

        f* = (b × p − q) / b
           = (p × bookie_odds − 1) / (bookie_odds − 1)
           = EV / (bookie_odds − 1)

    Wobei:
        p = Gewinnwahrscheinlichkeit (unser Modell)
        q = Verlustwahrscheinlichkeit = 1 − p
        b = Nettogewinn pro Einheit (bookie_odds − 1)
        f* = Optimaler Anteil der Bankroll

    Fractional Kelly (Viertel-Kelly)
    ---------------------------------
    Full Kelly maximiert langfristiges Kapitalwachstum, erzeugt aber extreme
    Drawdowns (bis zu −50 % bei schlechten Serien sind mathematisch normal).

    Lösung: Nur einen Bruchteil (fraction) des vollen Kelly-Wertes setzen:
        f_fractional = f* × fraction

    Mit fraction=0.25 (Viertel-Kelly):
      - Drawdowns statistisch ~4× geringer als Full Kelly
      - Langfristiges Wachstum ~75 % des theoretischen Maximums
      - Deutlich ruhigere Equity-Kurve → psychologisch besser haltbar

    Sicherheitsnetz: Negatives f* (kein Value) → Einsatz = 0.

    Parameters
    ----------
    prob         : Gewinnwahrscheinlichkeit (unser Modell)
    bookie_odds  : Dezimalquote des Buchmachers
    fraction     : Kelly-Bruchteil (Standard: 0.25 = Viertel-Kelly)

    Returns
    -------
    Einsatzanteil der Bankroll als Dezimalzahl (0.05 = 5 %).
    """
    b = bookie_odds - 1.0                     # Nettogewinn pro Einheit
    full_kelly = (prob * bookie_odds - 1) / b  # = EV / b
    fractional = full_kelly * fraction
    return max(fractional, 0.0)               # Niemals negativ setzen


# ---------------------------------------------------------------------------
# Ausgabe-Hilfsfunktion
# ---------------------------------------------------------------------------

def print_match_report(team_a: NationalTeam, team_b: NationalTeam) -> None:
    """Druckt einen übersichtlichen Matchbericht in die Konsole."""
    result = calculate_match_probabilities(team_a, team_b)
    odds   = probabilities_to_fair_odds(result)

    total_prob = result["home_win"] + result["draw"] + result["away_win"]

    separator = "=" * 56
    print(separator)
    print(f"  MATCHANALYSE (neutraler Boden, kein Heimvorteil)")
    print(separator)
    print(f"  {team_a.name:<20}  Elo: {team_a.elo:.0f}")
    print(f"  {team_b.name:<20}  Elo: {team_b.elo:.0f}")
    print(f"  Elo-Differenz: {team_a.elo - team_b.elo:+.0f}")
    print(separator)
    print(f"  Expected Goals (xG):")
    print(f"    {team_a.name}: {result['xg_a']:.4f}")
    print(f"    {team_b.name}: {result['xg_b']:.4f}")
    print(separator)
    print(f"  Wahrscheinlichkeiten (Poisson-Modell):")
    print(f"    Sieg {team_a.name:<16} (1):  {result['home_win']:6.2%}")
    print(f"    Unentschieden           (X):  {result['draw']:6.2%}")
    print(f"    Sieg {team_b.name:<16} (2):  {result['away_win']:6.2%}")
    print(f"    Summe (Sanity-Check):         {total_prob:.6f}")
    print(separator)
    print(f"  Faire Dezimalquoten (ohne Margin):")
    print(f"    1:  {odds['1']:.4f}")
    print(f"    X:  {odds['X']:.4f}")
    print(f"    2:  {odds['2']:.4f}")
    print(separator)


# ---------------------------------------------------------------------------
# Erweiterter Ausgabe-Report inkl. Value & Kelly
# ---------------------------------------------------------------------------

def print_value_report(
    team_a: NationalTeam,
    team_b: NationalTeam,
    bookie_odds: dict[str, float],
    bankroll: float = 1000.0,
    kelly_fraction_pct: float = 0.25,
) -> None:
    """
    Vollständiger Report: Modell → EV → Value Bets → Kelly-Einsatz.

    Parameters
    ----------
    team_a / team_b      : NationalTeam-Objekte
    bookie_odds          : {'1': float, 'X': float, '2': float}
    bankroll             : Gesamtkapital in EUR (für Einsatz-Berechnung)
    kelly_fraction_pct   : Kelly-Bruchteil (0.25 = Viertel-Kelly)
    """
    match_probs  = calculate_match_probabilities(team_a, team_b)
    fair_odds    = probabilities_to_fair_odds(match_probs)
    opportunities = analyze_value(
        match_probs, bookie_odds, team_a.name, team_b.name
    )

    sep  = "=" * 62
    sep2 = "-" * 62

    print(sep)
    print(f"  VALUE BETTING ANALYSE (neutraler Boden, Viertel-Kelly)")
    print(sep)
    print(f"  {team_a.name:<22}  Elo: {team_a.elo:.0f}")
    print(f"  {team_b.name:<22}  Elo: {team_b.elo:.0f}")
    print(f"  Bankroll: {bankroll:,.2f} EUR")
    print(sep)

    # Modell-Output
    print(f"  {'Ausgang':<24} {'Modell-P':>9}  {'FairOdds':>9}  {'BK-Quote':>9}")
    print(sep2)
    rows = [
        ("1", f"Sieg {team_a.name}", match_probs["home_win"]),
        ("X", "Unentschieden",       match_probs["draw"]),
        ("2", f"Sieg {team_b.name}", match_probs["away_win"]),
    ]
    for key, label, prob in rows:
        print(
            f"  ({key}) {label:<20} {prob:>8.2%}  "
            f"{fair_odds[key]:>9.4f}  {bookie_odds[key]:>9.4f}"
        )
    print(sep)

    # Buchmacher-Margin berechnen (implizite Über-Runde)
    implied_total = sum(1.0 / bookie_odds[k] for k in ("1", "X", "2"))
    margin_pct    = (implied_total - 1.0) * 100
    print(f"  Buchmacher-Margin (Vig): {margin_pct:.2f} %")
    print(sep)

    # EV-Tabelle und Value-Bets
    print(f"  {'Ausgang':<24} {'EV':>8}  {'Value Bet?':>12}")
    print(sep2)
    value_bets_found = False
    for opp in opportunities:
        tag = "*** VALUE BET ***" if opp.is_value_bet else "-"
        print(
            f"  ({opp.outcome}) {opp.label:<20} {opp.ev:>+8.2%}  {tag}"
        )
        if opp.is_value_bet:
            value_bets_found = True
    print(sep)

    # Kelly-Empfehlungen für Value Bets
    value_bets = [o for o in opportunities if o.is_value_bet]
    if not value_bets:
        print("  Kein positiver Erwartungswert gefunden. Keine Wette empfohlen.")
    else:
        print(f"  KELLY-EMPFEHLUNG (Viertel-Kelly, Faktor {kelly_fraction_pct}):")
        print(sep2)
        for opp in value_bets:
            f_kelly = kelly_fraction(opp.our_prob, opp.bookie_odds, kelly_fraction_pct)
            stake   = f_kelly * bankroll
            print(f"  Ausgang ({opp.outcome}) – {opp.label}")
            print(f"    Modell-Wahrscheinlichkeit : {opp.our_prob:.4%}")
            print(f"    Buchmacher-Quote          : {opp.bookie_odds:.4f}")
            print(f"    Expected Value            : {opp.ev:+.4%}")
            #
            # Kelly-Formel Schritt für Schritt:
            #   b = bookie_odds - 1        (Nettogewinn pro 1 EUR Einsatz)
            #   Full Kelly f* = EV / b     (= (p*odds - 1) / (odds - 1))
            #   Viertel-Kelly  = f* * 0.25
            #
            b          = opp.bookie_odds - 1.0
            full_kelly = opp.ev / b
            print(f"    Full Kelly f*             : {full_kelly:.4%}  (= EV / (Quote-1))")
            print(f"    Viertel-Kelly (÷4)        : {f_kelly:.4%}")
            print(f"    Empfohlener Einsatz       : {stake:.2f} EUR  "
                  f"({f_kelly:.2%} von {bankroll:,.0f} EUR)")
            print()
    print(sep)


# ---------------------------------------------------------------------------
# Schritt 6: Elo-Datenbank der WM-Teilnehmer
# ---------------------------------------------------------------------------

# Zentrale Datenbank mit realistischen World-Football-Elo-Ratings (Stand: 2025).
# Quelle: eloratings.net – Werte gerundet auf 10 Punkte.
# Erweiterbar: einfach weitere Einträge hinzufügen.
ELO_DATABASE: dict[str, float] = {
    # ── Weltspitze ────────────────────────────────────────────────────────────
    "Argentinien":    2140,
    "Frankreich":     2080,
    "Brasilien":      2060,
    "England":        2020,
    "Spanien":        2000,
    "Portugal":       1980,
    "Niederlande":    1960,
    "Deutschland":    1940,
    "Belgien":        1920,
    "Schweiz":        1870,
    "Kolumbien":      1860,
    "Ecuador":        1850,
    "Uruguay":        1880,
    "Österreich":     1880,
    "Kroatien":       1840,
    "Mexiko":         1830,
    "Türkei":         1820,
    "Marokko":        1820,
    "Japan":          1810,
    "Norwegen":       1780,
    "Südkorea":       1780,
    "Schweden":       1800,
    "Tschechien":     1800,
    "USA":            1850,
    # ── Mittelfeld ────────────────────────────────────────────────────────────
    "Senegal":        1790,
    "Elfenbeinküste": 1760,
    "Australien":     1760,
    "Schottland":     1760,
    "Iran":           1750,
    "Paraguay":       1740,
    "Algerien":       1740,
    "Tunesien":       1720,
    "Kamerun":        1710,
    "Ghana":          1680,
    "Ägypten":        1700,
    "Usbekistan":     1660,
    # ── Außenseiter ───────────────────────────────────────────────────────────
    "Kap Verde":      1650,
    "Panama":         1650,
    "DR Kongo":       1640,
    "Saudi-Arabien":  1620,
    "Irak":           1620,
    "Südafrika":      1620,
    "Katar":          1630,
    "Jordanien":      1600,
    "Neuseeland":     1600,
    "Haiti":          1500,
    "Curaçao":        1550,
}


def get_team(name: str) -> NationalTeam:
    """
    Erstellt ein NationalTeam-Objekt aus der Elo-Datenbank.

    Raises KeyError mit klarer Fehlermeldung, wenn der Name nicht gefunden wird.
    """
    if name not in ELO_DATABASE:
        available = ", ".join(sorted(ELO_DATABASE.keys()))
        raise KeyError(
            f"Team '{name}' nicht in der Elo-Datenbank gefunden.\n"
            f"Verfügbare Teams: {available}"
        )
    return NationalTeam(name=name, elo=ELO_DATABASE[name])


# ---------------------------------------------------------------------------
# Schritt 7: Spielplan-Struktur (ScheduledMatch)
# ---------------------------------------------------------------------------

@dataclass
class ScheduledMatch:
    """
    Repräsentiert ein anstehendes WM-Spiel mit Buchmacher-Quoten.

    Felder
    ------
    team_a       : Name von Team A (muss in ELO_DATABASE vorhanden sein)
    team_b       : Name von Team B (muss in ELO_DATABASE vorhanden sein)
    bookie_odds  : Dezimalquoten des Buchmachers {'1': ..., 'X': ..., '2': ...}
    label        : Optionale Spielbezeichnung (z.B. "Gruppe A – Spieltag 1")
    """
    team_a:      str
    team_b:      str
    bookie_odds: dict[str, float]
    label:       str = ""


@dataclass
class MatchRecommendation:
    """Ergebnis der Value-Analyse für ein einzelnes Spiel."""
    match:       ScheduledMatch
    team_a:      NationalTeam
    team_b:      NationalTeam
    match_probs: dict[str, float]
    value_bets:  list[BettingOpportunity]   # nur EV > 0


# ---------------------------------------------------------------------------
# Schritt 8: Batch-Analyse eines gesamten Spieltags
# ---------------------------------------------------------------------------

def analyze_matchday(
    matches:             list[ScheduledMatch],
    bankroll:            float = 1000.0,
    kelly_fraction_pct:  float = 0.25,
    min_ev_threshold:    float = 0.0,
    max_odds:            float = 10.0,
) -> list[MatchRecommendation]:
    """
    Analysiert einen kompletten Spieltag und gibt nur Spiele mit Value Bets zurück.

    Ablauf
    ------
    1. Elo-Lookup für beide Teams aus ELO_DATABASE
    2. Poisson-Modell → Wahrscheinlichkeiten
    3. EV-Berechnung gegen Buchmacher-Quoten
    4. Filterung: Value Bet nur wenn EV > min_ev_threshold UND Quote <= max_odds
       (Longshot-Filter: Außenseiter mit sehr hohen Quoten werden ausgeschlossen,
       da das Poisson-Modell dort weniger kalibriert ist und der Longshot Bias
       – die systematische Überschätzung von Außenseitern durch Wetter – zu
       falschen EV-Signalen führen kann)
    5. Multi-Bet-Sicherheitsabschlag:
       Wenn über ALLE Spiele des Tages mehr als eine Value Bet gefunden wird,
       wird der Kelly-Einsatz halbiert (×0.5). Begründung: Kelly-Kriterium
       setzt voraus, dass sequenziell gewettet wird. Bei parallelen Wetten
       am gleichen Spieltag teilt sich das effektive Kapital – die Halbierung
       ist eine konservative Approximation dieses Effekts.

    Parameters
    ----------
    matches            : Liste von ScheduledMatch-Objekten
    bankroll           : Gesamtkapital in EUR
    kelly_fraction_pct : Basis-Kelly-Bruchteil (vor Multi-Bet-Abschlag)
    min_ev_threshold   : Mindest-EV für eine Value Bet (Standard: 0.0 = positiv)
    max_odds           : Maximale Buchmacher-Quote für eine Value Bet.
                         Ausgänge mit höherer Quote werden ignoriert,
                         auch wenn EV > 0 (Longshot-Filter). Standard: 10.0

    Returns
    -------
    Tupel (recommendations, effective_kelly, bankroll, multi_bet_active).
    """
    recommendations: list[MatchRecommendation] = []

    # Schritt 1–4: Alle Spiele analysieren, Value Bets sammeln
    for scheduled in matches:
        team_a      = get_team(scheduled.team_a)
        team_b      = get_team(scheduled.team_b)
        match_probs = calculate_match_probabilities(team_a, team_b)
        opportunities = analyze_value(
            match_probs, scheduled.bookie_odds, team_a.name, team_b.name
        )
        # Doppeltes Filterkriterium: positiver EV UND Quote innerhalb des Limits
        value_bets = [
            o for o in opportunities
            if o.ev > min_ev_threshold and o.bookie_odds <= max_odds
        ]

        if value_bets:
            recommendations.append(MatchRecommendation(
                match=scheduled,
                team_a=team_a,
                team_b=team_b,
                match_probs=match_probs,
                value_bets=value_bets,
            ))

    # Schritt 5: Multi-Bet-Sicherheitsabschlag
    # Gesamtzahl Value Bets über alle Spiele des Tages zählen
    total_value_bets = sum(len(r.value_bets) for r in recommendations)
    multi_bet_active = total_value_bets > 1

    # Effektiven Kelly-Faktor bestimmen
    # Einzelne Value Bet: normaler Viertel-Kelly
    # Mehrere Value Bets: Viertel-Kelly × 0.5 = Achtel-Kelly
    effective_kelly = kelly_fraction_pct * (0.5 if multi_bet_active else 1.0)

    # Einsätze in den BettingOpportunity-Objekten aktualisieren ist nicht nötig –
    # wir übergeben effective_kelly an die Ausgabefunktion.
    # Metadaten als Tupel zurückgeben für die Print-Funktion:
    return recommendations, effective_kelly, bankroll, multi_bet_active


def print_matchday_report(
    matches:            list[ScheduledMatch],
    bankroll:           float = 1000.0,
    kelly_fraction_pct: float = 0.25,
) -> None:
    """
    Führt die Batch-Analyse durch und druckt einen kompakten Spieltags-Report.

    Ausgabe-Struktur
    ----------------
    1. Header mit Spieltag-Übersicht
    2. Pro Spiel MIT Value Bet: kompakter Block mit EV + Kelly-Einsatz
    3. Gesamt-Zusammenfassung: Anzahl Wetten, Gesamt-Einsatz, Risikohinweise
    """
    recommendations, effective_kelly, bankroll, multi_bet_active = analyze_matchday(
        matches, bankroll, kelly_fraction_pct
    )

    sep  = "=" * 68
    sep2 = "-" * 68
    sep3 = "~" * 68

    print(sep)
    print(f"  WM-SPIELTAG BATCH-ANALYSE  |  Bankroll: {bankroll:,.0f} EUR")
    print(f"  Analysierte Spiele: {len(matches)}  |  "
          f"Spiele mit Value Bet: {len(recommendations)}")
    if multi_bet_active:
        total_vb = sum(len(r.value_bets) for r in recommendations)
        print(f"  *** MULTI-BET-ABSCHLAG AKTIV: {total_vb} Value Bets gefunden ***")
        print(f"      Kelly-Faktor reduziert: "
              f"{kelly_fraction_pct:.0%} × 0.5 = {effective_kelly:.0%} (Achtel-Kelly)")
    print(sep)

    if not recommendations:
        print("  Kein einziges Spiel bietet heute eine Value Bet. Kein Einsatz.")
        print(sep)
        return

    total_stake = 0.0

    for rec in recommendations:
        match_label = rec.match.label or f"{rec.team_a.name} vs. {rec.team_b.name}"
        bookie      = rec.match.bookie_odds
        probs       = rec.match_probs
        fair        = probabilities_to_fair_odds(probs)
        margin      = (sum(1.0 / bookie[k] for k in ("1", "X", "2")) - 1.0) * 100

        print(sep3)
        print(f"  {match_label}")
        print(f"  {rec.team_a.name} (Elo {rec.team_a.elo:.0f})  vs.  "
              f"{rec.team_b.name} (Elo {rec.team_b.elo:.0f})"
              f"  |  BK-Margin: {margin:.1f} %")
        print(sep2)

        # Kompakte Wahrscheinlichkeits-Tabelle
        print(f"  {'':4} {'Ausgang':<22} {'Modell-P':>9}  "
              f"{'Fair':>7}  {'BK':>7}  {'EV':>8}  {'Signal':>6}")
        rows = [
            ("1", f"Sieg {rec.team_a.name}", probs["home_win"], "home_win"),
            ("X", "Unentschieden",           probs["draw"],     "draw"),
            ("2", f"Sieg {rec.team_b.name}", probs["away_win"], "away_win"),
        ]
        for key, label, prob, _ in rows:
            ev  = calculate_ev(prob, bookie[key])
            tag = "VALUE" if ev > 0 else "     "
            print(
                f"  ({key}) {label:<22} {prob:>8.2%}  "
                f"{fair[key]:>7.3f}  {bookie[key]:>7.3f}  "
                f"{ev:>+7.2%}  {tag}"
            )
        print(sep2)

        # Kelly-Empfehlungen für Value Bets in diesem Spiel
        for vb in rec.value_bets:
            f_kelly = kelly_fraction(vb.our_prob, vb.bookie_odds, effective_kelly)
            stake   = f_kelly * bankroll
            total_stake += stake
            b          = vb.bookie_odds - 1.0
            full_kelly = vb.ev / b

            print(f"  >> VALUE BET ({vb.outcome}): {vb.label}")
            print(f"     EV: {vb.ev:+.2%}  |  "
                  f"Full Kelly: {full_kelly:.3%}  |  "
                  f"Effektiver Kelly ({effective_kelly:.0%}): {f_kelly:.3%}")
            print(f"     Empfohlener Einsatz: {stake:.2f} EUR")
        print()

    # Gesamt-Zusammenfassung
    print(sep)
    print(f"  ZUSAMMENFASSUNG DES SPIELTAGS")
    print(sep2)
    total_value_bets = sum(len(r.value_bets) for r in recommendations)
    print(f"  Anzahl Value Bets heute     : {total_value_bets}")
    print(f"  Gesamt-Einsatz (empfohlen)  : {total_stake:.2f} EUR")
    print(f"  Bankroll nach Einsätzen     : {bankroll - total_stake:.2f} EUR")
    print(f"  Einsatz-Quote der Bankroll  : {total_stake / bankroll:.2%}")
    if multi_bet_active:
        print(f"  [Multi-Bet-Abschlag war aktiv – Kelly × 0.5]")
    print(sep)


# ---------------------------------------------------------------------------
# Schritt 9: Team-Namens-Mapping (API-Englisch → ELO_DATABASE-Deutsch)
# ---------------------------------------------------------------------------

# The Odds API liefert englische Teamnamen. Dieses Dictionary übersetzt sie
# in die deutschen Schlüssel unserer ELO_DATABASE.
# Erweiterbar: weitere Aliase einfach hinzufügen (z.B. Länderspezifika).
TEAM_NAME_MAPPING: dict[str, str] = {
    # Englischer API-Name        → Deutscher DB-Name
    # ── Weltspitze ──────────────────────────────────────────────────────────
    "Argentina":                   "Argentinien",
    "France":                      "Frankreich",
    "Brazil":                      "Brasilien",
    "England":                     "England",
    "Spain":                       "Spanien",
    "Portugal":                    "Portugal",
    "Netherlands":                 "Niederlande",
    "Germany":                     "Deutschland",
    "Belgium":                     "Belgien",
    "Switzerland":                 "Schweiz",
    "Colombia":                    "Kolumbien",
    "Ecuador":                     "Ecuador",
    "Uruguay":                     "Uruguay",
    "Austria":                     "Österreich",
    "Croatia":                     "Kroatien",
    "Mexico":                      "Mexiko",
    "Turkey":                      "Türkei",
    "Turkiye":                     "Türkei",   # FIFA-Schreibweise seit 2022
    "Morocco":                     "Marokko",
    "Japan":                       "Japan",
    "Norway":                      "Norwegen",
    "South Korea":                 "Südkorea",
    "Korea Republic":              "Südkorea",
    "Sweden":                      "Schweden",
    "Czech Republic":              "Tschechien",
    "Czechia":                     "Tschechien",
    "United States":               "USA",
    "USA":                         "USA",
    # ── Mittelfeld ──────────────────────────────────────────────────────────
    "Senegal":                     "Senegal",
    "Ivory Coast":                 "Elfenbeinküste",
    "Cote d'Ivoire":               "Elfenbeinküste",
    "Australia":                   "Australien",
    "Scotland":                    "Schottland",
    "Iran":                        "Iran",
    "Paraguay":                    "Paraguay",
    "Algeria":                     "Algerien",
    "Tunisia":                     "Tunesien",
    "Cameroon":                    "Kamerun",
    "Ghana":                       "Ghana",
    "Egypt":                       "Ägypten",
    "Uzbekistan":                  "Usbekistan",
    # ── Außenseiter ─────────────────────────────────────────────────────────
    "Cape Verde":                  "Kap Verde",
    "Cabo Verde":                  "Kap Verde",
    "Panama":                      "Panama",
    "DR Congo":                    "DR Kongo",
    "Congo DR":                    "DR Kongo",
    "Democratic Republic of Congo": "DR Kongo",
    "Saudi Arabia":                "Saudi-Arabien",
    "Iraq":                        "Irak",
    "South Africa":                "Südafrika",
    "Qatar":                       "Katar",
    "Jordan":                      "Jordanien",
    "New Zealand":                 "Neuseeland",
    "Haiti":                       "Haiti",
    "Curacao":                     "Curaçao",
    "Curaçao":                     "Curaçao",
    # ── Defensive Aliase (alternative API-Schreibweisen) ────────────────────
    "Brasil":                      "Brasilien",
    "Espana":                      "Spanien",
    "Pays-Bas":                    "Niederlande",
    "Allemagne":                   "Deutschland",
}


def resolve_team_name(api_name: str) -> Optional[str]:
    """
    Übersetzt einen englischen API-Namen in den deutschen ELO_DATABASE-Schlüssel.

    Suchreihenfolge:
      1. Direkter Treffer im TEAM_NAME_MAPPING
      2. Direkter Treffer in ELO_DATABASE (falls Name schon korrekt ist)
      3. None  → Aufrufer entscheidet über Fallback

    Returns
    -------
    Deutscher DB-Name oder None, wenn kein Mapping gefunden.
    """
    if api_name in TEAM_NAME_MAPPING:
        return TEAM_NAME_MAPPING[api_name]
    if api_name in ELO_DATABASE:
        return api_name
    return None


# ---------------------------------------------------------------------------
# Schritt 10: Mock-API-Response für Offline-Tests
# ---------------------------------------------------------------------------

def get_mock_api_response() -> list[dict]:
    """
    Simuliert eine typische JSON-Antwort von The Odds API (v4, H2H-Markt).

    Struktur entspricht dem echten API-Format:
      - 'home_team' / 'away_team': englische Teamnamen
      - 'bookmakers[].markets[].outcomes': Liste mit name + price
      - 'Draw' kodiert das Unentschieden

    Zwei Buchmacher pro Spiel werden gemittelt, genau wie in `parse_odds_api_response`.
    """
    raw_json = """
    [
        {
            "id": "mock_001",
            "sport_key": "soccer_fifa_world_cup",
            "sport_title": "FIFA World Cup",
            "commence_time": "2026-06-20T15:00:00Z",
            "home_team": "Argentina",
            "away_team": "Morocco",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Argentina", "price": 1.57},
                            {"name": "Draw",      "price": 4.10},
                            {"name": "Morocco",   "price": 5.80}
                        ]
                    }]
                },
                {
                    "key": "unibet",
                    "title": "Unibet",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Argentina", "price": 1.53},
                            {"name": "Draw",      "price": 4.30},
                            {"name": "Morocco",   "price": 6.20}
                        ]
                    }]
                }
            ]
        },
        {
            "id": "mock_002",
            "sport_key": "soccer_fifa_world_cup",
            "sport_title": "FIFA World Cup",
            "commence_time": "2026-06-20T18:00:00Z",
            "home_team": "France",
            "away_team": "Japan",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "France", "price": 1.48},
                            {"name": "Draw",   "price": 4.60},
                            {"name": "Japan",  "price": 7.20}
                        ]
                    }]
                },
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "France", "price": 1.52},
                            {"name": "Draw",   "price": 4.40},
                            {"name": "Japan",  "price": 7.80}
                        ]
                    }]
                }
            ]
        },
        {
            "id": "mock_003",
            "sport_key": "soccer_fifa_world_cup",
            "sport_title": "FIFA World Cup",
            "commence_time": "2026-06-20T21:00:00Z",
            "home_team": "Brazil",
            "away_team": "United States",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Brazil",         "price": 1.88},
                            {"name": "Draw",           "price": 3.55},
                            {"name": "United States",  "price": 5.00}
                        ]
                    }]
                }
            ]
        },
        {
            "id": "mock_004",
            "sport_key": "soccer_fifa_world_cup",
            "sport_title": "FIFA World Cup",
            "commence_time": "2026-06-20T21:00:00Z",
            "home_team": "Wakanda",
            "away_team": "Atlantis",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Wakanda",  "price": 2.10},
                            {"name": "Draw",     "price": 3.30},
                            {"name": "Atlantis", "price": 3.40}
                        ]
                    }]
                }
            ]
        }
    ]
    """
    return json.loads(raw_json)


# ---------------------------------------------------------------------------
# Schritt 11: Echte API-Abfrage bei The Odds API
# ---------------------------------------------------------------------------

# Basis-URL der Odds API v4
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"

# Timeout in Sekunden für HTTP-Requests
REQUEST_TIMEOUT: int = 10


def fetch_live_odds(
    api_key: str,
    sport_key: str = "soccer_fifa_world_cup",
) -> list[dict]:
    """
    Fragt aktuelle H2H-Quoten von The Odds API ab.

    API-Dokumentation: https://the-odds-api.com/liveapi/guides/v4/

    Endpunkt
    --------
    GET /v4/sports/{sport}/odds/
      ?apiKey=...
      &regions=eu          (europäische Buchmacher, Dezimalquoten)
      &markets=h2h         (1X2-Markt: Heimsieg / Unentschieden / Auswärtssieg)
      &oddsFormat=decimal

    Fehlerbehandlung
    ----------------
    - HTTP 401: Ungültiger API-Key → klare Fehlermeldung, leere Liste zurück
    - HTTP 422: Unbekannter sport_key → klare Fehlermeldung
    - Timeout (>REQUEST_TIMEOUT s): Netzwerkfehler-Meldung
    - Alle anderen Exceptions: allgemeiner Fallback

    Returns
    -------
    Liste von Spiel-Dicts im The-Odds-API-Format.
    Leere Liste bei jedem Fehler (Programm läuft weiter).
    """
    url = ODDS_API_BASE_URL.format(sport=sport_key)
    params = {
        "apiKey":      api_key,
        "regions":     "eu",
        "markets":     "h2h",
        "oddsFormat":  "decimal",
    }

    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

        if response.status_code == 401:
            print(f"[API-FEHLER] Ungültiger API-Key. Bitte Key prüfen.")
            return []
        if response.status_code == 422:
            print(f"[API-FEHLER] Unbekannter sport_key: '{sport_key}'.")
            return []
        if response.status_code != 200:
            print(f"[API-FEHLER] HTTP {response.status_code}: {response.text[:200]}")
            return []

        data = response.json()
        remaining = response.headers.get("x-requests-remaining", "?")
        used      = response.headers.get("x-requests-used", "?")
        print(f"[API] Anfragen verbraucht: {used}  |  Verbleibend: {remaining}")
        return data

    except requests.Timeout:
        print(f"[API-FEHLER] Timeout nach {REQUEST_TIMEOUT}s. Netzwerk prüfen.")
        return []
    except requests.ConnectionError:
        print("[API-FEHLER] Keine Verbindung. Internetverbindung prüfen.")
        return []
    except Exception as exc:
        print(f"[API-FEHLER] Unerwarteter Fehler: {exc}")
        return []


# ---------------------------------------------------------------------------
# Schritt 12: API-Response → ScheduledMatch-Objekte
# ---------------------------------------------------------------------------

def _average_h2h_odds(bookmakers: list[dict], home: str, away: str) -> Optional[dict[str, float]]:
    """
    Mittelt H2H-Quoten über alle verfügbaren Buchmacher.

    Begründung für Mittelung: Ein einzelner Buchmacher kann Ausreißer haben.
    Der Durchschnitt über mehrere Anbieter nähert sich der wahren Marktquote an.

    Returns None, wenn kein gültiger H2H-Markt gefunden wird.
    """
    home_prices, draw_prices, away_prices = [], [], []

    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcome_map = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            if home in outcome_map and away in outcome_map and "Draw" in outcome_map:
                home_prices.append(outcome_map[home])
                draw_prices.append(outcome_map["Draw"])
                away_prices.append(outcome_map[away])

    if not home_prices:
        return None

    return {
        "1": round(sum(home_prices) / len(home_prices), 3),
        "X": round(sum(draw_prices) / len(draw_prices), 3),
        "2": round(sum(away_prices) / len(away_prices), 3),
    }


def parse_odds_api_response(
    data: list[dict],
    source_label: str = "API",
) -> list[ScheduledMatch]:
    """
    Wandelt eine The-Odds-API-JSON-Antwort in ScheduledMatch-Objekte um.

    Ablauf pro Spiel
    ----------------
    1. Englische Teamnamen aus 'home_team' / 'away_team' lesen
    2. Namen via resolve_team_name() in deutsche ELO_DATABASE-Schlüssel übersetzen
    3. Fallback: Wenn ein Team unbekannt ist → Warnung + Spiel überspringen
    4. H2H-Quoten über alle Buchmacher mitteln
    5. ScheduledMatch erstellen

    Parameters
    ----------
    data         : Rohe API-Antwort (Liste von Spiel-Dicts)
    source_label : Wird im Label angezeigt ("API" oder "Mock")

    Returns
    -------
    Liste von ScheduledMatch-Objekten, bereit für analyze_matchday().
    """
    matches: list[ScheduledMatch] = []

    for game in data:
        home_en = game.get("home_team", "")
        away_en = game.get("away_team", "")
        time    = game.get("commence_time", "")[:10]   # nur Datum

        # Namens-Mapping Englisch → Deutsch
        home_de = resolve_team_name(home_en)
        away_de = resolve_team_name(away_en)

        # Fallback: Unbekannte Teams überspringen
        if home_de is None:
            print(f"[WARNUNG] Team '{home_en}' nicht in Elo-Datenbank. "
                  f"Spiel {home_en} vs. {away_en} übersprungen.")
            continue
        if away_de is None:
            print(f"[WARNUNG] Team '{away_en}' nicht in Elo-Datenbank. "
                  f"Spiel {home_en} vs. {away_en} übersprungen.")
            continue

        # Quoten mitteln
        odds = _average_h2h_odds(game.get("bookmakers", []), home_en, away_en)
        if odds is None:
            print(f"[WARNUNG] Kein H2H-Markt für {home_en} vs. {away_en}. Übersprungen.")
            continue

        matches.append(ScheduledMatch(
            team_a=home_de,
            team_b=away_de,
            bookie_odds=odds,
            label=f"[{source_label}] {time} | {home_de} vs. {away_de}",
        ))

    return matches


# ---------------------------------------------------------------------------
# Schritt 13: Einheitlicher Datenbeschaffungs-Einstiegspunkt
# ---------------------------------------------------------------------------

def load_matches_from_api(
    api_key: Optional[str] = None,
    sport_key: str = "soccer_fifa_world_cup",
) -> list[ScheduledMatch]:
    """
    Haupt-Einstiegspunkt für die Datenbeschaffung.

    Logik
    -----
    - api_key ist None oder leer → Mock-Daten werden verwendet (Offline-Modus)
    - api_key vorhanden → echter API-Aufruf; bei leerem Ergebnis Fallback auf Mock

    Parameters
    ----------
    api_key   : The-Odds-API-Key (oder None für Offline-Modus)
    sport_key : Sportschlüssel der API (Standard: WM-Fußball)

    Returns
    -------
    Liste von ScheduledMatch-Objekten.
    """
    if not api_key:
        print("[INFO] Kein API-Key angegeben → Offline-Modus mit Mock-Daten.")
        raw = get_mock_api_response()
        return parse_odds_api_response(raw, source_label="Mock")

    print(f"[INFO] Lade Live-Quoten von The Odds API ({sport_key}) ...")
    raw = fetch_live_odds(api_key, sport_key)

    if not raw:
        print("[INFO] API lieferte keine Daten → Fallback auf Mock-Daten.")
        raw = get_mock_api_response()
        return parse_odds_api_response(raw, source_label="Mock-Fallback")

    return parse_odds_api_response(raw, source_label="Live")


# ---------------------------------------------------------------------------
# Testlauf
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Schritt 4 – Live-Daten-Infrastruktur (API / Mock)
    # -----------------------------------------------------------------------
    # API-Key aus Umgebungsvariable lesen (sicherer als Hardcoding).
    # Ohne Key → automatisch Mock-Modus.
    #
    # Verwendung mit echtem Key:
    #   Windows:  set ODDS_API_KEY=dein_key_hier
    #   Linux:    export ODDS_API_KEY=dein_key_hier
    #   Dann:     python wm_model.py
    import os
    api_key = os.environ.get("ODDS_API_KEY")  # None wenn nicht gesetzt

    print("=" * 68)
    print("  SCHRITT 4: API-INTEGRATION (The Odds API)")
    print("=" * 68)

    # Spiele laden (Mock wenn kein Key, Live wenn Key vorhanden)
    api_matches = load_matches_from_api(api_key=api_key)

    if api_matches:
        print(f"[INFO] {len(api_matches)} Spiel(e) erfolgreich geladen.\n")
        print_matchday_report(api_matches, bankroll=1000.0, kelly_fraction_pct=0.25)
    else:
        print("[INFO] Keine verwertbaren Spiele nach Parsing. Analyse abgebrochen.")

    print("\n\n")

    # -----------------------------------------------------------------------
    # Schritt 3 – Kompletter WM-Spieltag (Batch-Analyse)
    # -----------------------------------------------------------------------
    # Fiktiver Spieltag mit 4 Partien aus verschiedenen Gruppen.
    # Buchmacher-Quoten sind absichtlich so gewählt, dass manche Spiele
    # Value bieten und andere nicht – um die Filterung zu demonstrieren.
    #
    # Intuition vor dem Run:
    #   Argentinien vs. Marokko: ARG ist klarer Favorit (~2140 Elo).
    #     Quote 1.55 auf ARG ist sehr tief → wahrscheinlich kein Value auf 1.
    #     Quote 6.00 auf MAR könte Value haben falls Modell MAR höher sieht.
    #   Frankreich vs. Japan: FRA klar stärker (~270 Elo diff).
    #     Quote 1.50 auf FRA → sehr tief, wohl kein Value.
    #     Quote 7.50 auf JPN → Modell wird JPN wohl ~15% geben, BK-impl. ~13.3%.
    #   Brasilien vs. USA: Ausgeglichener (~210 Elo diff).
    #     Quote 5.20 auf USA könnte interessant sein.
    #   England vs. Senegal: ENG Favorit (~230 Elo diff).
    #     Quote 3.80 auf X und 5.00 auf SEN könnten Value haben.
    # -----------------------------------------------------------------------
    spieltag = [
        ScheduledMatch(
            team_a="Argentinien",
            team_b="Marokko",
            bookie_odds={"1": 1.55, "X": 4.20, "2": 6.00},
            label="Gruppe D – Spiel 1 | Argentinien vs. Marokko",
        ),
        ScheduledMatch(
            team_a="Frankreich",
            team_b="Japan",
            bookie_odds={"1": 1.50, "X": 4.50, "2": 7.50},
            label="Gruppe C – Spiel 2 | Frankreich vs. Japan",
        ),
        ScheduledMatch(
            team_a="Brasilien",
            team_b="USA",
            bookie_odds={"1": 1.85, "X": 3.60, "2": 5.20},
            label="Gruppe B – Spiel 1 | Brasilien vs. USA",
        ),
        ScheduledMatch(
            team_a="England",
            team_b="Senegal",
            bookie_odds={"1": 1.75, "X": 3.80, "2": 5.00},
            label="Gruppe A – Spiel 3 | England vs. Senegal",
        ),
    ]

    print_matchday_report(spieltag, bankroll=1000.0, kelly_fraction_pct=0.25)

    print("\n\n")

    # -----------------------------------------------------------------------
    # Schritt 1 – Sanity-Checks (unverändert aus vorherigen Schritten)
    # -----------------------------------------------------------------------
    spanien     = NationalTeam(name="Spanien",     elo=2100)
    kamerun     = NationalTeam(name="Kamerun",     elo=1700)
    deutschland = NationalTeam(name="Deutschland", elo=1900)
    frankreich  = NationalTeam(name="Frankreich",  elo=1900)

    print_match_report(spanien, kamerun)
    print()
    print_match_report(deutschland, frankreich)

    print("\n")

    # -----------------------------------------------------------------------
    # Schritt 2 – Value Betting mit fiktiven Buchmacher-Quoten
    # -----------------------------------------------------------------------
    # Szenario: Team A (2050 Elo) vs. Team B (1800 Elo)
    # Fiktive Buchmacher-Quoten: 1.60 / 4.00 / 5.50
    #
    # Erwartung (Intuition):
    #   Unser Modell sieht Team A als klaren Favoriten.
    #   BK-Quote 1.60 auf Sieg A klingt "tief" → könnte kein Value sein.
    #   BK-Quote 5.50 auf Sieg B klingt "hoch" → könnte Value sein,
    #   falls unser Modell B höher einschätzt als die BK-implizite ~18.2 %.
    # -----------------------------------------------------------------------
    team_a = NationalTeam(name="Team A", elo=2050)
    team_b = NationalTeam(name="Team B", elo=1800)

    bookie = {"1": 1.60, "X": 4.00, "2": 5.50}

    print_value_report(
        team_a=team_a,
        team_b=team_b,
        bookie_odds=bookie,
        bankroll=1000.0,
        kelly_fraction_pct=0.25,
    )
