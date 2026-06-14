"""
WM Value Betting Dashboard – Streamlit Web Interface
=====================================================
Startet mit:  streamlit run app.py

Tab 1 – Live-Analyse : Spieldaten laden, Value Bets identifizieren,
                        Wetten direkt ins Tagebuch eintragen.
Tab 2 – Meine Performance : Bankroll-Chart, KPIs, offene und
                              abgerechnete Wetten verwalten.
"""

from __future__ import annotations

import contextlib
import io
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Backend-Imports (mathematischer Kern + Tracker – beide unberührt)
# ─────────────────────────────────────────────────────────────────────────────
from wm_model import (
    ELO_DATABASE,
    ScheduledMatch,
    analyze_matchday,
    calculate_ev,
    calculate_match_probabilities,
    fetch_live_odds,
    get_mock_api_response,
    get_team,
    kelly_fraction,
    parse_odds_api_response,
    probabilities_to_fair_odds,
)
from bet_tracker import (
    add_bet,
    calculate_kpis,
    delete_bet,
    get_all_bets,
    get_bankroll_history,
    get_open_bets,
    get_settled_bets,
    update_bet_result,
)

# ─────────────────────────────────────────────────────────────────────────────
# Seiten-Konfiguration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WM Value Betting",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Kompaktere Metriken */
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    /* Expander-Rahmen */
    div[data-testid="stExpander"] {
        border: 1px solid #2d2d44;
        border-radius: 8px;
        margin-bottom: 6px;
    }
    /* Tabs etwas größer */
    button[data-baseweb="tab"] { font-size: 1.05rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session-State
# ─────────────────────────────────────────────────────────────────────────────
_SS_DEFAULTS: dict = {
    "all_matches":      [],
    "recommendations":  [],
    "effective_kelly":  0.25,
    "multi_bet_active": False,
    "data_loaded":      False,
    "load_messages":    [],
    # Menge der in dieser Session bereits eingetragenen Bet-Keys
    # (verhindert doppeltes Eintragen ohne Page-Reload)
    "added_bet_keys":   set(),
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen – Datenbeschaffung
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_with_feedback(api_key: Optional[str]) -> list[ScheduledMatch]:
    """
    Ruft Spieldaten ab (Live oder Mock) und sammelt alle Statusmeldungen
    in session_state, damit sie nach dem Rerun als st.info/warning/error
    im Browser erscheinen statt als Konsolenausgabe.
    """
    messages: list[tuple[str, str]] = []
    matches:  list[ScheduledMatch]  = []
    captured = io.StringIO()

    try:
        with contextlib.redirect_stdout(captured):
            if not api_key:
                raw     = get_mock_api_response()
                matches = parse_odds_api_response(raw, source_label="Mock")
                messages.append(("info",
                    "Kein API-Key → Offline-Modus mit simulierten WM-Daten."))
            else:
                raw = fetch_live_odds(api_key)
                if not raw:
                    messages.append(("warning",
                        "API lieferte keine Daten → Fallback auf Mock-Daten."))
                    raw     = get_mock_api_response()
                    matches = parse_odds_api_response(raw, source_label="Mock-Fallback")
                else:
                    matches = parse_odds_api_response(raw, source_label="Live")
                    messages.append(("success",
                        f"{len(matches)} Live-Spiele erfolgreich geladen."))
    except Exception as exc:
        messages.append(("error", f"Unerwarteter Fehler: {exc}"))
        st.session_state.load_messages = messages
        return []

    # Warnungen aus abgefangenem print()-Output parsen
    for line in captured.getvalue().splitlines():
        line = line.strip()
        if not line:
            continue
        if "[WARNUNG]" in line:
            messages.append(("warning", line.replace("[WARNUNG] ", "")))
        elif "[API-FEHLER]" in line:
            messages.append(("error",   line.replace("[API-FEHLER] ", "")))
        elif "[API]" in line:
            messages.append(("info",    line.replace("[API] ", "")))

    st.session_state.load_messages = messages
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen – Analyse-DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_df(
    all_matches: list[ScheduledMatch],
    effective_kelly: float,
    bankroll: float,
) -> pd.DataFrame:
    """
    Erstellt einen DataFrame mit ALLEN Spielen × 3 Ausgängen.
    Value Bets sind durch die bool-Spalte 'Value Bet' markiert.
    Interne Metadaten (Prefix '_') werden in Tab-1-Expander genutzt.
    """
    rows: list[dict] = []

    for match in all_matches:
        try:
            team_a = get_team(match.team_a)
            team_b = get_team(match.team_b)
        except KeyError:
            continue

        probs  = calculate_match_probabilities(team_a, team_b)
        fair   = probabilities_to_fair_odds(probs)
        margin = (
            sum(1.0 / match.bookie_odds[k] for k in ("1", "X", "2")) - 1.0
        ) * 100

        date_str = ""
        if "|" in match.label:
            prefix   = match.label.split("|")[0].strip()
            date_str = prefix.rsplit(" ", 1)[-1]

        for key, label, prob_key in [
            ("1", f"Sieg {match.team_a}", "home_win"),
            ("X", "Unentschieden",        "draw"),
            ("2", f"Sieg {match.team_b}", "away_win"),
        ]:
            prob    = probs[prob_key]
            bk_odds = match.bookie_odds[key]
            ev      = calculate_ev(prob, bk_odds)
            is_vb   = ev > 0.0
            f_k     = kelly_fraction(prob, bk_odds, effective_kelly) if is_vb else 0.0
            stake   = f_k * bankroll

            rows.append({
                "Match":         f"{match.team_a} vs. {match.team_b}",
                "Datum":         date_str,
                "Ausgang":       key,
                "Beschreibung":  label,
                "Modell-P":      prob,
                "Faire Quote":   float(fair[key]),
                "BK-Quote":      bk_odds,
                "EV":            ev,
                "Kelly (%)":     f_k,
                "Einsatz (EUR)": stake,
                "Value Bet":     is_vb,
                # Metadaten für Expander
                "_xg_a":  probs["xg_a"],
                "_xg_b":  probs["xg_b"],
                "_margin": margin,
                "_team_a": match.team_a,
                "_team_b": match.team_b,
                "_elo_a":  team_a.elo,
                "_elo_b":  team_b.elo,
            })

    return pd.DataFrame(rows)


def _style_overview_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Formatiert die Übersichts-Tabelle; Value Bets grün."""
    display_cols = [
        "Match", "Datum", "Ausgang", "Beschreibung",
        "Modell-P", "Faire Quote", "BK-Quote",
        "EV", "Kelly (%)", "Einsatz (EUR)", "Value Bet",
    ]
    disp = df[display_cols].copy()
    disp["Modell-P"]      = df["Modell-P"].map("{:.1%}".format)
    disp["Faire Quote"]   = df["Faire Quote"].map("{:.3f}".format)
    disp["BK-Quote"]      = df["BK-Quote"].map("{:.3f}".format)
    disp["EV"]            = df["EV"].map("{:+.2%}".format)
    disp["Kelly (%)"]     = df["Kelly (%)"].map("{:.2%}".format)
    disp["Einsatz (EUR)"] = df["Einsatz (EUR)"].map("{:.2f} €".format)
    disp["Value Bet"]     = df["Value Bet"].map(lambda v: "✅ VALUE" if v else "—")

    vb_flags = df["Value Bet"].values

    def _hl(row: pd.Series) -> list[str]:
        if vb_flags[row.name]:
            return ["background-color: #14532d; color: #86efac"] * len(row)
        return [""] * len(row)

    return disp.style.apply(_hl, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen – Performance-Chart
# ─────────────────────────────────────────────────────────────────────────────

def _build_bankroll_chart(history_df: pd.DataFrame, start_bankroll: float) -> go.Figure:
    """
    Erstellt einen Plotly-Linien-Chart der Bankroll-Entwicklung.
    Gewonnene Punkte = grün, verlorene = rot, Startpunkt = grau.
    """
    # Farben je nach Status
    colors = []
    for s in history_df["Status"]:
        if s == "Gewonnen":
            colors.append("#4ade80")
        elif s == "Verloren":
            colors.append("#f87171")
        else:
            colors.append("#94a3b8")

    fig = go.Figure()

    # Verbindungslinie
    fig.add_trace(go.Scatter(
        x=history_df["Wette_Nr"],
        y=history_df["Bankroll"],
        mode="lines",
        line=dict(color="#6366f1", width=2),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Punkte mit Tooltip
    hover_texts = []
    for _, row in history_df.iterrows():
        if row["Status"] == "Start":
            hover_texts.append(f"<b>Start</b><br>Bankroll: {row['Bankroll']:.2f} €")
        else:
            sign = "+" if row["PnL"] >= 0 else ""
            hover_texts.append(
                f"<b>{row['Label']}</b><br>"
                f"Status: {row['Status']}<br>"
                f"PnL: {sign}{row['PnL']:.2f} €<br>"
                f"Bankroll: {row['Bankroll']:.2f} €"
            )

    fig.add_trace(go.Scatter(
        x=history_df["Wette_Nr"],
        y=history_df["Bankroll"],
        mode="markers",
        marker=dict(color=colors, size=10, line=dict(color="#1e1e2e", width=1)),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    # Start-Bankroll als horizontale Referenzlinie
    fig.add_hline(
        y=start_bankroll,
        line_dash="dash",
        line_color="#64748b",
        annotation_text=f"Start: {start_bankroll:.0f} €",
        annotation_position="bottom right",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(
            title="Wette Nr.",
            gridcolor="#2d2d44",
            tickmode="linear",
            dtick=1,
        ),
        yaxis=dict(
            title="Bankroll (EUR)",
            gridcolor="#2d2d44",
        ),
        margin=dict(l=20, r=20, t=20, b=40),
        height=340,
        hoverlabel=dict(bgcolor="#1e293b", font_size=13),
    )
    return fig


def _style_history_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Formatiert die History-Tabelle; Gewonnen grün, Verloren rot."""
    def _hl(row: pd.Series) -> list[str]:
        s = row.get("Status", "")
        if s == "Gewonnen":
            return ["background-color: #14532d; color: #86efac"] * len(row)
        if s == "Verloren":
            return ["background-color: #450a0a; color: #fca5a5"] * len(row)
        return [""] * len(row)

    disp = df.copy()
    if "EV_bei_Abgabe" in disp.columns:
        disp["EV_bei_Abgabe"] = disp["EV_bei_Abgabe"].map("{:+.2%}".format)
    if "Einsatz_EUR" in disp.columns:
        disp["Einsatz_EUR"]   = disp["Einsatz_EUR"].map("{:.2f} €".format)
    if "PnL" in disp.columns:
        disp["PnL"]           = disp["PnL"].map(lambda v: f"{v:+.2f} €")
    if "Quote" in disp.columns:
        disp["Quote"]         = disp["Quote"].map("{:.3f}".format)

    return disp.style.apply(_hl, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚽ WM Value Betting")
    st.caption("Poisson-Modell · Elo-Ratings · Kelly-Kriterium")
    st.divider()

    st.subheader("🔑 API-Konfiguration")
    api_key_input = st.text_input(
        "The Odds API Key",
        type="password",
        placeholder="Leer lassen für Mock-Modus",
        help="API-Key von the-odds-api.com",
    )

    st.divider()
    st.subheader("💰 Bankroll-Management")

    bankroll = st.number_input(
        "Start-Bankroll (EUR)",
        min_value=10.0,
        max_value=1_000_000.0,
        value=1_000.0,
        step=50.0,
        format="%.2f",
        help="Dein Startkapital. Wird als Basis für Kelly-Einsätze und ROI-Berechnung genutzt.",
    )

    kelly_pct = st.slider(
        "Kelly-Fraktion",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
        format="%.2f",
        help="0.25 = Viertel-Kelly (empfohlen) · 0.5 = Halb-Kelly · 1.0 = Full Kelly",
    )
    kelly_label = {0.25: "Viertel-Kelly ✓", 0.50: "Halb-Kelly",
                   1.00: "Full Kelly ⚠️"}.get(kelly_pct, f"{kelly_pct:.0%}-Kelly")
    st.caption(f"Modus: **{kelly_label}**")

    st.divider()
    max_odds = st.slider(
        "Maximale Quote (Risk Limit)",
        min_value=1.5,
        max_value=10.0,
        value=2.5,
        step=0.1,
        format="%.1f",
        help=(
            "Longshot-Filter: Value Bets werden nur empfohlen, wenn die "
            "Buchmacher-Quote ≤ diesem Wert liegt.\n\n"
            "Hintergrund: Bei sehr hohen Quoten (Außenseitern) neigen "
            "Wetter zu Longshot Bias – die Wahrscheinlichkeiten wirken "
            "attraktiver als sie sind. Das Poisson-Modell ist dort "
            "weniger zuverlässig kalibriert.\n\n"
            "Empfehlung: 2.5–3.5 für konservative Strategie."
        ),
    )
    # Kontextinfo: welche Quoten-Zone ist das?
    if max_odds <= 2.0:
        st.caption(f"Filter: Quoten bis **{max_odds:.1f}** – nur klare Favoriten")
    elif max_odds <= 3.0:
        st.caption(f"Filter: Quoten bis **{max_odds:.1f}** – Favoriten & leichte Außenseiter")
    elif max_odds <= 5.0:
        st.caption(f"Filter: Quoten bis **{max_odds:.1f}** – moderate Außenseiter")
    else:
        st.caption(f"Filter: Quoten bis **{max_odds:.1f}** ⚠️ Longshot-Risiko steigt")

    st.divider()
    st.subheader("ℹ️ Elo-Datenbank")
    st.caption(f"{len(ELO_DATABASE)} Teams")
    with st.expander("Teams anzeigen"):
        db_df = pd.DataFrame(
            sorted(ELO_DATABASE.items(), key=lambda x: -x[1]),
            columns=["Team", "Elo"],
        )
        st.dataframe(db_df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Seitentitel + Tabs
# ─────────────────────────────────────────────────────────────────────────────
st.title("WM Value Betting Dashboard")
tab1, tab2 = st.tabs(["📊 Live-Analyse", "📈 Meine Performance"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – LIVE-ANALYSE
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Fetch-Button ──────────────────────────────────────────────────────────
    fetch_col, _ = st.columns([2, 5])
    with fetch_col:
        fetch_clicked = st.button(
            "🔄 Live-Daten abrufen / Aktualisieren",
            type="primary",
            use_container_width=True,
        )

    if fetch_clicked:
        with st.spinner("Lade Spieldaten ..."):
            loaded = _fetch_with_feedback(api_key_input or None)
        if loaded:
            st.session_state.all_matches    = loaded
            st.session_state.data_loaded    = True
            st.session_state.added_bet_keys = set()

    # Statusmeldungen
    for level, text in st.session_state.get("load_messages", []):
        getattr(st, level)(text)

    if not st.session_state.data_loaded:
        st.info(
            "Klicke auf **🔄 Live-Daten abrufen / Aktualisieren** um die Analyse zu starten.  \n"
            "Ohne API-Key werden automatisch simulierte WM-Daten genutzt."
        )
        st.stop()

    all_matches = st.session_state.all_matches

    # Analyse läuft bei JEDEM Render neu — so wirkt der max_odds-Slider
    # sofort, ohne dass der Nutzer erneut abrufen muss.
    recs, effective_kelly, _, multi_active = analyze_matchday(
        all_matches, bankroll, kelly_pct, max_odds=max_odds
    )
    recommendations = recs
    total_vb        = sum(len(r.value_bets) for r in recommendations)

    # ── Übersichtskarten ──────────────────────────────────────────────────────
    st.subheader("📊 Spieltags-Übersicht")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analysierte Spiele",    len(all_matches))
    c2.metric("Spiele mit Value Bet",  len(recommendations),
              f"{len(recommendations)/max(len(all_matches),1):.0%} der Spiele")
    c3.metric("Value Bets (Ausgänge)", total_vb)
    with c4:
        if multi_active:
            st.metric("Multi-Bet-Abschlag", "AKTIV",
                      f"Kelly × 0.5 = {effective_kelly:.0%}", delta_color="inverse")
        else:
            st.metric("Multi-Bet-Abschlag", "inaktiv",
                      f"Kelly = {effective_kelly:.0%}")

    st.divider()

    # ── Vollständige Ergebnistabelle ──────────────────────────────────────────
    st.subheader("📋 Alle Spiele im Überblick")
    full_df = _build_full_df(all_matches, effective_kelly, bankroll)

    if full_df.empty:
        st.warning("Keine Spiele konnten analysiert werden.")
        st.stop()

    n_vb = int(full_df["Value Bet"].sum())
    st.caption(
        f"{len(all_matches)} Spiele · {len(full_df)} Ausgänge · "
        f"**{n_vb} Value Bets** (grün hervorgehoben)"
    )
    st.dataframe(
        _style_overview_df(full_df),
        hide_index=True,
        use_container_width=True,
        height=min(80 + len(full_df) * 36, 560),
    )

    st.divider()

    # ── Detail-Expander pro Spiel ─────────────────────────────────────────────
    st.subheader("🔍 Detailanalyse pro Spiel")
    unique_matches = full_df.drop_duplicates(subset=["Match"])

    for _, mrow in unique_matches.iterrows():
        match_name = mrow["Match"]
        team_a     = mrow["_team_a"]
        team_b     = mrow["_team_b"]
        xg_a       = mrow["_xg_a"]
        xg_b       = mrow["_xg_b"]
        margin     = mrow["_margin"]
        elo_a      = mrow["_elo_a"]
        elo_b      = mrow["_elo_b"]
        has_vb     = full_df[(full_df["Match"] == match_name) & full_df["Value Bet"]].shape[0] > 0

        exp_label = (
            f"{'✅ ' if has_vb else '  '}{match_name}"
            f"  |  Elo: {elo_a:.0f} vs. {elo_b:.0f}"
            f"  |  BK-Margin: {margin:.1f} %"
        )

        with st.expander(exp_label, expanded=has_vb):
            d1, d2, d3 = st.columns(3)
            elo_diff = elo_a - elo_b
            win_prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400))

            with d1:
                st.markdown(f"**{team_a}**")
                st.metric("Elo-Rating",     f"{elo_a:.0f}")
                st.metric("Expected Goals", f"{xg_a:.3f}")
            with d2:
                st.markdown("**Match-Info**")
                st.metric("Elo-Differenz",   f"{elo_diff:+.0f}")
                st.metric("Elo-Win-Prob A",  f"{win_prob:.1%}")
                st.metric("BK-Margin (Vig)", f"{margin:.2f} %")
            with d3:
                st.markdown(f"**{team_b}**")
                st.metric("Elo-Rating",     f"{elo_b:.0f}")
                st.metric("Expected Goals", f"{xg_b:.3f}")

            # Tabelle der 3 Ausgänge
            match_sub = full_df[full_df["Match"] == match_name].copy()
            match_sub = match_sub[[
                "Ausgang", "Beschreibung", "Modell-P", "Faire Quote",
                "BK-Quote", "EV", "Kelly (%)", "Einsatz (EUR)", "Value Bet",
            ]].reset_index(drop=True)

            vb_flags_exp = match_sub["Value Bet"].values
            disp_exp     = match_sub.drop(columns=["Value Bet"]).copy()
            disp_exp["Modell-P"]      = match_sub["Modell-P"].map("{:.2%}".format)
            disp_exp["Faire Quote"]   = match_sub["Faire Quote"].map("{:.4f}".format)
            disp_exp["BK-Quote"]      = match_sub["BK-Quote"].map("{:.4f}".format)
            disp_exp["EV"]            = match_sub["EV"].map("{:+.2%}".format)
            disp_exp["Kelly (%)"]     = match_sub["Kelly (%)"].map("{:.3%}".format)
            disp_exp["Einsatz (EUR)"] = match_sub["Einsatz (EUR)"].map("{:.2f} €".format)

            def _hl_exp(row: pd.Series, flags=vb_flags_exp) -> list[str]:
                if flags[row.name]:
                    return ["background-color: #14532d; color: #86efac"] * len(row)
                return [""] * len(row)

            st.dataframe(
                disp_exp.style.apply(_hl_exp, axis=1),
                hide_index=True,
                use_container_width=True,
            )

    st.divider()

    # ── Shopping-Liste + "Wette eintragen" ────────────────────────────────────
    st.subheader("🛒 Wettempfehlungen + Eintragen")

    value_rows = full_df[full_df["Value Bet"]].sort_values("EV", ascending=False).copy()

    if value_rows.empty:
        st.success("Heute keine Value Bets – Kapital schonen.")
    else:
        total_stake = value_rows["Einsatz (EUR)"].sum()
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Anzahl Value Bets",    total_vb)
        sc2.metric("Gesamt-Einsatz",       f"{total_stake:.2f} €",
                   f"{total_stake/bankroll:.2%} der Bankroll")
        sc3.metric("Verbleibende Bankroll", f"{bankroll - total_stake:.2f} €")

        if multi_active:
            st.info(
                f"**Multi-Bet-Abschlag aktiv:** {total_vb} Value Bets gefunden → "
                f"Kelly {kelly_pct:.0%} × 0.5 = {effective_kelly:.0%}"
            )

        st.markdown("---")

        added_keys: set = st.session_state.added_bet_keys

        for i, (_, vr) in enumerate(value_rows.iterrows()):
            # Eindeutiger Schlüssel für diese Wettempfehlung
            bet_key = f"{vr['Match']}|{vr['Ausgang']}"

            col_nr, col_info, col_ev, col_stake, col_btn = st.columns([0.4, 3.5, 1.5, 1.8, 2])

            with col_nr:
                st.markdown(f"**#{i+1}**")

            with col_info:
                st.markdown(
                    f"**{vr['Beschreibung']}**  \n"
                    f"{vr['Match']}  ·  Ausgang **{vr['Ausgang']}**  ·  "
                    f"Quote: **{vr['BK-Quote']:.3f}**"
                )

            with col_ev:
                st.metric("EV", f"{vr['EV']:+.2%}", label_visibility="collapsed")
                st.caption(f"EV: {vr['EV']:+.2%}")

            with col_stake:
                st.metric("Einsatz", f"{vr['Einsatz (EUR)']:.2f} €",
                          label_visibility="collapsed")
                st.caption(f"{vr['Einsatz (EUR)']:.2f} € ({vr['Kelly (%)']:.2%})")

            with col_btn:
                if bet_key in added_keys:
                    st.success("✅ Eingetragen", icon=None)
                else:
                    if st.button(
                        "📝 Wette eintragen",
                        key=f"add_{i}",
                        use_container_width=True,
                        type="secondary",
                    ):
                        try:
                            add_bet(
                                spiel        = vr["Match"],
                                tipp         = vr["Ausgang"],
                                beschreibung = vr["Beschreibung"],
                                quote        = float(vr["BK-Quote"]),
                                ev           = float(vr["EV"]),
                                einsatz      = float(vr["Einsatz (EUR)"]),
                            )
                            st.session_state.added_bet_keys.add(bet_key)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Fehler beim Eintragen: {exc}")

        st.divider()
        st.success(
            f"✅ Gesamt-Einsatz: **{total_stake:.2f} €** "
            f"({total_stake/bankroll:.2%} der {bankroll:,.0f} € Bankroll)"
        )

    st.caption(
        "⚠️ Dieses Tool dient ausschließlich zu Analyse- und Bildungszwecken. "
        "Wetten enthält immer Risiken."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – MEINE PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
with tab2:

    kpis    = calculate_kpis(bankroll)
    history = get_bankroll_history(bankroll)
    open_b  = get_open_bets()
    settled = get_settled_bets()
    all_b   = get_all_bets()

    # ── KPI-Karten ────────────────────────────────────────────────────────────
    st.subheader("📈 Performance-Übersicht")

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        delta_br = kpis["current_bankroll"] - bankroll
        st.metric(
            "Aktuelle Bankroll",
            f"{kpis['current_bankroll']:.2f} €",
            delta=f"{delta_br:+.2f} €",
            delta_color="normal",
        )
    with k2:
        st.metric(
            "Netto-Profit",
            f"{kpis['net_profit']:+.2f} €",
            delta=f"aus {kpis['total_staked']:.2f} € Einsatz" if kpis["total_staked"] > 0 else None,
        )
    with k3:
        st.metric(
            "ROI",
            f"{kpis['roi']:+.1%}",
            help="Return on Investment = Netto-Profit / Summe aller Einsätze (abgerechnete Wetten)",
        )
    with k4:
        win_rate_str = f"{kpis['win_rate']:.0%}" if (kpis["won"] + kpis["lost"]) > 0 else "—"
        st.metric(
            "Win-Rate",
            win_rate_str,
            delta=f"{kpis['won']}W / {kpis['lost']}L" if (kpis["won"] + kpis["lost"]) > 0 else None,
        )
    with k5:
        st.metric(
            "Wetten gesamt",
            kpis["total_bets"],
            delta=f"{kpis['open']} offen" if kpis["open"] > 0 else "alle abgerechnet",
        )

    st.divider()

    # ── Bankroll-Chart ────────────────────────────────────────────────────────
    st.subheader("📉 Bankroll-Entwicklung")

    if len(history) <= 1:
        st.info(
            "Noch keine abgerechneten Wetten vorhanden.  \n"
            "Trage in **Tab 1** Value Bets ein und markiere sie hier als "
            "Gewonnen / Verloren – dann erscheint der Chart."
        )
    else:
        fig = _build_bankroll_chart(history, bankroll)
        st.plotly_chart(fig, use_container_width=True)

        # Mini-Statistiken unter dem Chart
        cs1, cs2, cs3 = st.columns(3)
        max_br = history["Bankroll"].max()
        min_br = history["Bankroll"].min()
        cs1.metric("Peak Bankroll",   f"{max_br:.2f} €", f"{max_br - bankroll:+.2f} €")
        cs2.metric("Tiefpunkt",       f"{min_br:.2f} €", f"{min_br - bankroll:+.2f} €",
                   delta_color="inverse")
        cs3.metric("Ø EV bei Abgabe", f"{kpis['avg_ev']:+.2%}",
                   help="Durchschnittlicher Expected Value aller eingetragener Wetten")

    st.divider()

    # ── Offene Wetten – Ergebnis eintragen ───────────────────────────────────
    st.subheader(f"⏳ Offene Wetten ({len(open_b)})")

    if open_b.empty:
        st.success("Keine offenen Wetten – alles abgerechnet.")
    else:
        for _, row in open_b.iterrows():
            bet_id  = int(row["id"])
            pnl_win = round(float(row["Einsatz_EUR"]) * (float(row["Quote"]) - 1), 2)
            pnl_los = -float(row["Einsatz_EUR"])

            with st.container():
                oc1, oc2, oc3, oc4, oc5 = st.columns([0.5, 3, 1.5, 2, 2])

                with oc1:
                    st.caption(f"#{bet_id}")
                with oc2:
                    st.markdown(
                        f"**{row['Beschreibung']}**  \n"
                        f"{row['Spiel']}  ·  Ausgang **{row['Tipp']}**"
                    )
                    st.caption(
                        f"Datum: {row['Datum']}  ·  "
                        f"Quote: {float(row['Quote']):.3f}  ·  "
                        f"EV: {float(row['EV_bei_Abgabe']):+.2%}"
                    )
                with oc3:
                    st.metric(
                        "Einsatz",
                        f"{float(row['Einsatz_EUR']):.2f} €",
                        label_visibility="visible",
                    )
                with oc4:
                    if st.button(
                        f"✅ Gewonnen  (+{pnl_win:.2f} €)",
                        key=f"win_{bet_id}",
                        use_container_width=True,
                        type="primary",
                    ):
                        try:
                            update_bet_result(bet_id, "Gewonnen")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                with oc5:
                    if st.button(
                        f"❌ Verloren  ({pnl_los:.2f} €)",
                        key=f"lose_{bet_id}",
                        use_container_width=True,
                    ):
                        try:
                            update_bet_result(bet_id, "Verloren")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

                # Löschen (Notfall-Funktion, dezent)
                with st.expander("⚙️ Wette löschen (irrtümlich eingetragen)", expanded=False):
                    if st.button(
                        f"🗑️ Wette #{bet_id} dauerhaft löschen",
                        key=f"del_{bet_id}",
                        type="secondary",
                    ):
                        try:
                            delete_bet(bet_id)
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

                st.markdown("---")

    st.divider()

    # ── History-Tabelle (abgerechnete Wetten) ─────────────────────────────────
    st.subheader(f"📚 Wetthistorie ({len(settled)} abgerechnete Wetten)")

    if settled.empty:
        st.info("Noch keine abgerechneten Wetten in der Historie.")
    else:
        display_cols = ["id", "Datum", "Spiel", "Beschreibung",
                        "Tipp", "Quote", "EV_bei_Abgabe",
                        "Einsatz_EUR", "Status", "PnL"]
        hist_display = settled[display_cols].copy()

        st.dataframe(
            _style_history_df(hist_display),
            hide_index=True,
            use_container_width=True,
            height=min(80 + len(hist_display) * 36, 480),
            column_config={
                "id":            st.column_config.NumberColumn("ID",     width="small"),
                "Datum":         st.column_config.TextColumn("Datum",    width="small"),
                "Spiel":         st.column_config.TextColumn("Spiel",    width="large"),
                "Beschreibung":  st.column_config.TextColumn("Tipp"),
                "Tipp":          st.column_config.TextColumn("1X2",      width="small"),
                "Quote":         st.column_config.NumberColumn("Quote",  format="%.3f"),
                "EV_bei_Abgabe": st.column_config.TextColumn("EV"),
                "Einsatz_EUR":   st.column_config.TextColumn("Einsatz"),
                "Status":        st.column_config.TextColumn("Status"),
                "PnL":           st.column_config.TextColumn("PnL"),
            },
        )

        # CSV-Export
        csv_bytes = settled.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Historie als CSV exportieren",
            data=csv_bytes,
            file_name="wm_betting_history_export.csv",
            mime="text/csv",
        )
