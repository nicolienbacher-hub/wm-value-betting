"""
WM Value Betting – Bet Tracker Backend
=======================================
CSV-basiertes System zum Speichern, Aktualisieren und Auswerten von Wetten.

Datenfluss
----------
add_bet()           → neue Zeile mit Status "Offen", PnL = 0
update_bet_result() → Status → "Gewonnen"/"Verloren", PnL wird berechnet
get_*()             → Lesezugriffe
calculate_kpis()    → aggregierte Kennzahlen (Bankroll, ROI, Win-Rate)
get_bankroll_history() → zeitliche Entwicklung für Chart
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

CSV_PATH: Path = Path("betting_history.csv")

# Kanonische Spaltenliste – Reihenfolge definiert CSV-Layout
COLUMNS: list[str] = [
    "id",
    "Datum",
    "Spiel",
    "Tipp",           # "1", "X", "2"
    "Beschreibung",   # z.B. "Sieg Argentinien"
    "Quote",
    "EV_bei_Abgabe",  # EV zum Zeitpunkt der Wettabgabe (als Dezimalzahl, z.B. 0.5587)
    "Einsatz_EUR",
    "Status",         # "Offen" | "Gewonnen" | "Verloren"
    "PnL",            # Profit / Loss in EUR (0 solange "Offen")
]

Status = Literal["Offen", "Gewonnen", "Verloren"]


# ---------------------------------------------------------------------------
# Interne Helfer
# ---------------------------------------------------------------------------

def _load() -> pd.DataFrame:
    """
    Lädt die CSV. Legt sie an, falls sie nicht existiert.
    Ergänzt fehlende Spalten (Rückwärtskompatibilität bei Schema-Updates).
    """
    if not CSV_PATH.exists():
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(CSV_PATH, index=False)
        return df

    df = pd.read_csv(CSV_PATH, dtype={"id": "Int64"})

    # Neue Spalten sicher ergänzen ohne bestehende Daten zu verlieren
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[COLUMNS].copy()


def _save(df: pd.DataFrame) -> None:
    """Schreibt den DataFrame atomar in die CSV."""
    df.to_csv(CSV_PATH, index=False)


def _next_id(df: pd.DataFrame) -> int:
    """Gibt die nächste freie ID zurück (einfacher Auto-Increment)."""
    if df.empty or df["id"].isna().all():
        return 1
    return int(df["id"].max()) + 1


# ---------------------------------------------------------------------------
# Schreib-Operationen
# ---------------------------------------------------------------------------

def add_bet(
    spiel:        str,
    tipp:         str,
    beschreibung: str,
    quote:        float,
    ev:           float,
    einsatz:      float,
) -> int:
    """
    Fügt eine neue Wette mit Status "Offen" hinzu.

    Parameters
    ----------
    spiel        : Matchbezeichnung, z.B. "Argentinien vs. Marokko"
    tipp         : Gewetteter Ausgang ("1", "X" oder "2")
    beschreibung : Lesbare Bezeichnung, z.B. "Sieg Marokko"
    quote        : Buchmacher-Dezimalquote zum Zeitpunkt der Abgabe
    ev           : Expected Value als Dezimalzahl (z.B. 0.5587 für +55.87 %)
    einsatz      : Empfohlener Kelly-Einsatz in EUR

    Returns
    -------
    ID der neu angelegten Wette (int).
    """
    df = _load()
    new_id = _next_id(df)

    new_row = pd.DataFrame([{
        "id":           new_id,
        "Datum":        date.today().isoformat(),
        "Spiel":        spiel,
        "Tipp":         tipp,
        "Beschreibung": beschreibung,
        "Quote":        round(quote, 4),
        "EV_bei_Abgabe": round(ev, 6),
        "Einsatz_EUR":  round(einsatz, 2),
        "Status":       "Offen",
        "PnL":          0.0,
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    _save(df)
    return new_id


def update_bet_result(bet_id: int, status: Status) -> float:
    """
    Schließt eine offene Wette ab und berechnet den PnL.

    PnL-Formel
    ----------
    Gewonnen : PnL = Einsatz × (Quote − 1)   [Nettogewinn ohne Einsatzrückgabe]
    Verloren : PnL = −Einsatz

    Parameters
    ----------
    bet_id : ID der Wette (aus add_bet() oder get_open_bets())
    status : "Gewonnen" oder "Verloren"

    Returns
    -------
    Berechneter PnL in EUR.

    Raises
    ------
    ValueError bei unbekannter ID oder ungültigem Status.
    """
    if status not in ("Gewonnen", "Verloren"):
        raise ValueError(f"Ungültiger Status '{status}'. Erlaubt: Gewonnen, Verloren")

    df   = _load()
    mask = df["id"] == bet_id

    if not mask.any():
        raise ValueError(f"Wette mit ID {bet_id} nicht gefunden.")

    row    = df.loc[mask].iloc[0]
    einsatz = float(row["Einsatz_EUR"])
    quote   = float(row["Quote"])

    pnl = einsatz * (quote - 1.0) if status == "Gewonnen" else -einsatz
    pnl = round(pnl, 2)

    df.loc[mask, "Status"] = status
    df.loc[mask, "PnL"]    = pnl
    _save(df)
    return pnl


def delete_bet(bet_id: int) -> None:
    """
    Löscht eine Wette dauerhaft aus der CSV.
    Nur sinnvoll für "Offen"-Wetten (z.B. irrtümlich eingetragen).
    """
    df   = _load()
    mask = df["id"] == bet_id
    if not mask.any():
        raise ValueError(f"Wette mit ID {bet_id} nicht gefunden.")
    df = df[~mask].copy()
    _save(df)


# ---------------------------------------------------------------------------
# Lese-Operationen
# ---------------------------------------------------------------------------

def get_all_bets() -> pd.DataFrame:
    """Gibt alle Wetten zurück (sortiert: neueste zuerst)."""
    df = _load()
    return df.sort_values("id", ascending=False).reset_index(drop=True)


def get_open_bets() -> pd.DataFrame:
    """Gibt nur offene Wetten zurück."""
    df = _load()
    return df[df["Status"] == "Offen"].sort_values("id").reset_index(drop=True)


def get_settled_bets() -> pd.DataFrame:
    """Gibt nur abgerechnete Wetten zurück (Gewonnen + Verloren)."""
    df = _load()
    return (
        df[df["Status"] != "Offen"]
        .sort_values("id")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Performance-Kennzahlen
# ---------------------------------------------------------------------------

def calculate_kpis(start_bankroll: float) -> dict:
    """
    Berechnet alle Performance-KPIs aus der bisherigen Wetthistorie.

    Parameters
    ----------
    start_bankroll : Startkapital in EUR (wird vom Benutzer konfiguriert)

    Returns
    -------
    dict mit folgenden Schlüsseln:

    current_bankroll : start_bankroll + Summe aller abgerechneten PnL
    net_profit       : Summe aller PnL (positiv = Gewinn, negativ = Verlust)
    roi              : net_profit / total_staked (nur abgerechnete Einsätze)
    total_bets       : Gesamtzahl aller Wetten (inkl. Offen)
    won              : Anzahl gewonnener Wetten
    lost             : Anzahl verlorener Wetten
    open             : Anzahl offener Wetten
    win_rate         : won / (won + lost) – nur abgerechnete Wetten
    total_staked     : Summierter Einsatz aller abgerechneten Wetten
    avg_ev           : Durchschnittlicher EV bei Abgabe (alle Wetten)
    avg_odds         : Durchschnittliche Quote (alle Wetten)
    """
    df      = get_all_bets()
    settled = df[df["Status"] != "Offen"]
    won     = df[df["Status"] == "Gewonnen"]
    lost    = df[df["Status"] == "Verloren"]
    open_b  = df[df["Status"] == "Offen"]

    net_profit   = float(settled["PnL"].sum())     if not settled.empty else 0.0
    total_staked = float(settled["Einsatz_EUR"].sum()) if not settled.empty else 0.0
    roi          = net_profit / total_staked        if total_staked > 0  else 0.0
    win_rate     = len(won) / (len(won) + len(lost)) if (len(won) + len(lost)) > 0 else 0.0
    avg_ev       = float(df["EV_bei_Abgabe"].mean()) if not df.empty else 0.0
    avg_odds     = float(df["Quote"].mean())          if not df.empty else 0.0

    return {
        "current_bankroll": start_bankroll + net_profit,
        "net_profit":        net_profit,
        "roi":               roi,
        "total_bets":        len(df),
        "won":               len(won),
        "lost":              len(lost),
        "open":              len(open_b),
        "win_rate":          win_rate,
        "total_staked":      total_staked,
        "avg_ev":            avg_ev,
        "avg_odds":          avg_odds,
    }


def get_bankroll_history(start_bankroll: float) -> pd.DataFrame:
    """
    Gibt die Bankroll-Entwicklung als DataFrame zurück (für Linien-Chart).

    Nur abgerechnete Wetten fließen ein – offene Wetten haben PnL=0
    und würden die Kurve verfälschen.

    Returns
    -------
    DataFrame mit Spalten:
      Wette_Nr   : 0 (Start), 1, 2, ... (laufende Nummer)
      Bankroll   : kumulierte Bankroll nach jeder Wette
      Label      : Kurzbezeichnung für Tooltip
      PnL        : PnL der einzelnen Wette
      Status     : "Gewonnen" | "Verloren"
    """
    settled = get_settled_bets()

    # Startpunkt (Wette 0)
    start_row = pd.DataFrame([{
        "Wette_Nr":  0,
        "Bankroll":  start_bankroll,
        "Label":     "Start",
        "PnL":       0.0,
        "Status":    "Start",
        "Datum":     "",
    }])

    if settled.empty:
        return start_row

    # Abgerechnete Wetten chronologisch sortieren
    hist = settled.sort_values(["Datum", "id"]).copy()
    hist = hist.reset_index(drop=True)
    hist["Wette_Nr"]  = hist.index + 1
    hist["Bankroll"]  = start_bankroll + hist["PnL"].cumsum()
    hist["Label"]     = (
        hist["Wette_Nr"].astype(str) + ". "
        + hist["Beschreibung"] + " @ " + hist["Quote"].astype(str)
    )

    result = pd.concat(
        [start_row, hist[["Wette_Nr", "Bankroll", "Label", "PnL", "Status", "Datum"]]],
        ignore_index=True,
    )
    return result
