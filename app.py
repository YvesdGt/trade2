import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="L'Algorithme Caméléon Prédictif", layout="wide")
st.title("🏆 Stratégie d'Arbitrage par Corrélation")

# --- BARRE LATÉRALE (PARAMÈTRES) ---
st.sidebar.header("Configuration de l'Expert")
target_ticker = st.sidebar.text_input("Actif à prédire (ex: TSLA)", "TSLA")
guide_ticker = st.sidebar.text_input("Actif Guide (ex: MSFT)", "MSFT")

lookback = st.sidebar.slider("Fenêtre de Corrélation (Jours)", 5, 100, 20)
lag_days = st.sidebar.slider("Jour(s) d'avance connus", 1, 5, 1)
trading_fees = st.sidebar.slider("Frais de transaction (%)", 0.0, 1.0, 0.3) / 100

initial_cap = 10000

@st.cache_data
def load_data(t1, t2):
    # On télécharge un historique large pour le calcul des moyennes mobiles/corrélations
    d = yf.download([t1, t2], start="2023-01-01", end="2025-01-01", auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d = d['Close']
    return d

try:
    df_raw = load_data(target_ticker, guide_ticker).copy()

    # --- LOGIQUE PRÉDICTIVE ---
    # On récupère le prix du futur pour l'actif guide (ton avantage de connaissance)
    df_raw['Guide_Future'] = df_raw[guide_ticker].shift(-lag_days)
    df_raw['Guide_Return_Future'] = df_raw['Guide_Future'].pct_change(lag_days)

    # Calcul de la corrélation entre les deux actifs sur la fenêtre choisie
    df_raw['Corr'] = df_raw[target_ticker].pct_change().rolling(lookback).corr(df_raw[guide_ticker].pct_change())

    # --- FILTRAGE 2024 ---
    df = df_raw.loc['2024-01-01':'2024-12-31'].copy()

    # SIGNAL : On achète si MSFT monte dans le futur ET que la corrélation historique est positive
    df['Position'] = np.where((df['Guide_Return_Future'] > 0) & (df['Corr'] > 0.2), 1, 0)
    df['Trade'] = df['Position'].diff().abs().fillna(0)

    # --- BACKTEST ---
    rets = df[target_ticker].pct_change().fillna(0).values
    pos_vals = df['Position'].values
    trade_vals = df['Trade'].values

    strat_path = []
    c_strat = initial_cap

    for i in range(len(df)):
        if i > 0 and pos_vals[i-1] == 1:
            c_strat *= (1 + rets[i])
        # Application des frais à l'achat ET à la vente
        if trade_vals[i] == 1:
            c_strat *= (1 - trading_fees)
        strat_path.append(c_strat)

    df['Strategy'] = strat_path
    df['BuyHold'] = (df[target_ticker] / df[target_ticker].iloc[0]) * initial_cap

    # --- DCA ---
    dca_cap = 0
    monthly_budget = initial_cap / 12
    cash_spent = 0
    shares = 0
    for date, row in df.iterrows():
        if date == df[df.index.month == date.month].index[0] and cash_spent < initial_cap:
            shares += (monthly_budget * (1 - trading_fees)) / row[target_ticker]
            cash_spent += monthly_budget
        df.loc[date, 'DCA'] = (shares * row[target_ticker]) + (initial_cap - cash_spent)

    # --- AFFICHAGE ---
    perf_algo = ((df['Strategy'].iloc[-1]/initial_cap)-1)*100
    perf_bh = ((df['BuyHold'].iloc[-1]/initial_cap)-1)*100
    perf_dca = ((df['DCA'].iloc[-1]/initial_cap)-1)*100

    c1, c2, c3 = st.columns(3)
    c1.metric(f"ALGO ({target_ticker})", f"{perf_algo:.2f}%", f"{perf_algo-perf_bh:.2f}% vs B&H")
    c2.metric("BUY & HOLD", f"{perf_bh:.2f}%")
    c3.metric("DCA Mensuel", f"{perf_dca:.2f}%")

    # Graphique interactif
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Strategy'], name="Stratégie Algorithmique", line=dict(color='#00ffcc', width=3)))
    fig.add_trace(go.Scatter(x=df.index, y=df['BuyHold'], name="Buy & Hold (Référence)", line=dict(color='#ff9900')))
    fig.add_trace(go.Scatter(x=df.index, y=df['DCA'], name="DCA Mensuel", line=dict(color='white', dash='dot')))

    fig.update_layout(
        template="plotly_dark",
        title=f"Comparaison des performances : {target_ticker} guidé par {guide_ticker}",
        xaxis_title="Date",
        yaxis_title="Valeur du Portefeuille ($)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Zone d'analyse technique
    with st.expander("Voir les outils de corrélation"):
        st.write("Ce graphique montre comment la corrélation entre les deux actifs a évolué. Si elle est proche de 1, l'actif guide est un excellent miroir.")
        st.line_chart(df['Corr'])

except Exception as e:
    st.error(f"Erreur lors du calcul : {e}")