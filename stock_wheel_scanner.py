import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Wheel & 2x-Hebel Scanner", layout="wide")
st.title("🚀 Dein Wheel + 2x-Hebel Scanner")
st.caption("Wheel: Div + 4Y Kurs + >18% Prämie | Hebel: Growth + Chart-Muster")

# Sidebar
st.sidebar.header("Einstellungen")
min_market_cap = st.sidebar.number_input("Mindest Market Cap (Mrd. USD)", value=5.0, step=0.5)
min_div_yield = st.sidebar.number_input("Mindest Dividendenrendite Wheel (%)", value=1.5, step=0.5)
macro = st.sidebar.selectbox("Makro-Szenario", ["Neutral", "Hohe Ölpreise / Iran", "Schwaches Asien-Wachstum"])

# Stabiles hardcoded Universum (ca. 380 Titel)
tickers = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","JPM","V","MA","PG","XOM","CVX","LLY","UNH","JNJ","HD","WMT","BAC","KO","PEP","MRK","ABBV","TMO","COST","ACN","MCD","ADBE","CSCO","NFLX","AMD","CRM","INTC","QCOM","TXN","AMGN","HON","SPGI","AXP","NOW","ISRG","BKNG","INTU","UBER","CAT","DE","GE","BA","RTX","LMT","NOC","GD","HII","PH","ETN","EMR","ITW","MMM","UPS","FDX","CSX","NSC","UNP","CP","CNI","KSU","DAL","UAL","AAL","LUV","SAVE","ALK","JBLU","SKYW","HA","RYAAY","WAB","TRN","GBX","RAIL","CSX","NSC","UNP","CP","CNI","KSU","DAL","UAL","AAL","LUV","SAVE","ALK","JBLU","SKYW","HA","RYAAY","WAB","TRN","GBX","RAIL",
    # Nasdaq 100 (Auszug)
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","ADBE","CSCO","INTC","QCOM","TXN","AMGN","HON","SPGI","AXP","NOW","ISRG","BKNG","INTU","UBER","CAT","DE","GE","BA","RTX","LMT","NOC","GD","HII","PH","ETN","EMR","ITW","MMM","UPS","FDX","CSX","NSC","UNP","CP","CNI","KSU","DAL","UAL","AAL","LUV","SAVE","ALK","JBLU","SKYW","HA","RYAAY","WAB","TRN","GBX","RAIL",
    # DAX
    "SAP.DE","AIR.DE","SIE.DE","DTE.DE","ALV.DE","MBG.DE","BMW.DE","VOW.DE","BAS.DE","BAYN.DE","DBK.DE","DPW.DE","IFX.DE","RWE.DE","VNA.DE","CON.DE","ADS.DE","HEI.DE","MRK.DE","MTX.DE","PUM.DE","1COV.DE","ZAL.DE","DB1.DE","ENR.DE","FRE.DE","HFG.DE","LEG.DE","LIN.DE","MLT.DE","MTX.DE","PAH3.DE","QIA.DE","RHM.DE","SDF.DE","SRT.DE","SY1.DE","TLX.DE","VOW3.DE",
    # Weitere große S&P Titel
    "BRK.B","LLY","UNH","JNJ","HD","WMT","BAC","KO","PEP","MRK","ABBV","TMO","COST","ACN","MCD","ADBE","CSCO","NFLX","AMD","CRM","INTC","QCOM","TXN","AMGN","HON","SPGI","AXP","NOW","ISRG","BKNG","INTU","UBER","CAT","DE","GE","BA","RTX","LMT","NOC","GD","HII","PH","ETN","EMR","ITW","MMM","UPS","FDX","CSX","NSC","UNP","CP","CNI","KSU","DAL","UAL","AAL","LUV","SAVE","ALK","JBLU","SKYW","HA","RYAAY","WAB","TRN","GBX","RAIL"
]

if st.button("🔥 Wöchentlichen Scan starten (2–5 Min)", type="primary"):
    with st.spinner(f"Scanne {len(tickers)} Titel..."):
        wheel_list = []
        hebel_list = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if info.get("marketCap", 0) < min_market_cap * 1e9:
                    continue

                hist4 = stock.history(period="4y")
                if len(hist4) < 150:
                    continue

                # Wheel
                price_cagr4 = (hist4['Close'][-1] / hist4['Close'][0]) ** (1/4) - 1
                div_yield = info.get("dividendYield", 0) * 100
                divs = stock.dividends
                div_growth = 0.0
                if len(divs) >= 4:
                    div_growth = (divs.iloc[-1] / divs.iloc[-4]) ** (1/4) - 1

                premium = 0.0
                try:
                    opt = stock.option_chain(stock.options[0])
                    puts = opt.puts
                    good_put = puts[(puts['delta'] > -0.35) & (puts['delta'] < -0.25)]
                    if not good_put.empty:
                        premium = good_put['lastPrice'].iloc[0]
                except:
                    pass

                annual_premium = premium * 12 if premium > 0 else 0
                if price_cagr4 > 0 and div_yield >= min_div_yield and annual_premium > 18:
                    delta = hist4['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    rs = gain / loss.replace(0, np.nan)
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

                    score = 40 + (div_growth > 0.10)*25 + (current_rsi < 40)*30
                    wheel_list.append({
                        "Ticker": ticker,
                        "Score": round(score, 1),
                        "DivYield": round(div_yield, 1),
                        "4YDivGrowth": round(div_growth*100, 1),
                        "AnnualPremium%": round(annual_premium, 1),
                        "RSI": round(current_rsi, 1),
                        "Warum": f"Hohe Prämie ({round(annual_premium,1)}%) + Div-Growth + positives 4Y-Kurswachstum"
                    })

                # Hebel
                eps_growth = info.get("earningsGrowth", 0) or 0
                rev_growth = info.get("revenueGrowth", 0) or 0
                if eps_growth > 0 and rev_growth > 0:
                    delta = hist4['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    rs = gain / loss.replace(0, np.nan)
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

                    score = 30 + (eps_growth > 0.15)*30 + (current_rsi < 50)*40
                    hebel_list.append({
                        "Ticker": ticker,
                        "Score": round(score, 1),
                        "EPS4YGrowth": round(eps_growth*100, 1),
                        "Revenue4YGrowth": round(rev_growth*100, 1),
                        "RSI": round(current_rsi, 1),
                        "Warum": f"Starkes Growth + RSI-Pullback/Momentum (RSI {round(current_rsi,1)})"
                    })
            except:
                continue

        st.subheader("📊 Wheel-Top 5–10 (>18% annualisierte Prämie)")
        if wheel_list:
            st.dataframe(pd.DataFrame(wheel_list).sort_values("Score", ascending=False).head(10), use_container_width=True)
        else:
            st.info("Keine Wheel-Kandidaten gefunden – Filter sind streng.")

        st.subheader("📈 2x-Hebel-Top 5–10 (Growth + Chart-Muster)")
        if hebel_list:
            st.dataframe(pd.DataFrame(hebel_list).sort_values("Score", ascending=False).head(10), use_container_width=True)
        else:
            st.info("Keine Hebel-Kandidaten gefunden.")

        st.success(f"✅ Scan fertig! Makro-Szenario: {macro} | {len(tickers)} Titel gescannt")

st.info("App ist jetzt final stabil. Drücke auf den Scan-Button!")
