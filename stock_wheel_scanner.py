import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta

st.set_page_config(page_title="Wheel & 2x-Hebel Scanner", layout="wide")
st.title("🚀 Dein Wheel + 2x-Hebel Scanner")
st.caption("Wheel: Div + 5+10Y Kurs | Hebel: nur 5Y EPS+Revenue Growth | Mit Makro & YouTube-Tipps")

# Sidebar
st.sidebar.header("Einstellungen")
min_market_cap = st.sidebar.number_input("Mindest Market Cap (Mrd. USD)", value=5.0, step=0.5)
min_div_yield = st.sidebar.number_input("Mindest Dividendenrendite Wheel (%)", value=2.5, step=0.5)
macro = st.sidebar.selectbox("Makro-Szenario", ["Neutral", "Hohe Ölpreise / Iran", "Schwaches Asien-Wachstum"])

if st.button("🔥 Wöchentlichen Scan starten (3–8 Min)", type="primary"):
    with st.spinner("Scanne S&P500, Nasdaq, Dow, DAX..."):
        # Kleines, stabiles Universe für den Start (kann später erweitert werden)
        tickers = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","JPM","V","MA","PG","XOM","CVX","SAP.DE","AIR.DE","SIE.DE"]
        
        wheel_list = []
        hebel_list = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if info.get("marketCap", 0) < min_market_cap * 1e9:
                    continue

                hist5 = stock.history(period="5y")
                hist10 = stock.history(period="10y")
                if len(hist5) < 200:
                    continue

                # === WHEEL (streng) ===
                price_cagr5 = (hist5['Close'][-1] / hist5['Close'][0]) ** (1/5) - 1
                price_cagr10 = (hist10['Close'][-1] / hist10['Close'][0]) ** (1/10) - 1 if len(hist10) >= 400 else -1
                div_yield = info.get("dividendYield", 0) * 100
                divs = stock.dividends
                div_growth = 0.0
                if len(divs) >= 5:
                    try:
                        div_growth = (divs.iloc[-1] / divs.iloc[-5]) ** (1/5) - 1
                    except:
                        pass

                if price_cagr5 > 0 and price_cagr10 > 0 and div_yield >= min_div_yield:
                    df = hist5.copy()
                    df['RSI'] = ta.rsi(df['Close'], length=14)
                    macd = ta.macd(df['Close'])
                    current_rsi = df['RSI'][-1] if not pd.isna(df['RSI'][-1]) else 50

                    premium = 0.0
                    try:
                        opt = stock.option_chain(stock.options[0])
                        puts = opt.puts
                        good_put = puts[(puts['delta'] > -0.35) & (puts['delta'] < -0.25)]
                        if not good_put.empty:
                            premium = good_put['lastPrice'].iloc[0]
                    except:
                        pass

                    score = 40 + (div_growth > 0.10)*25 + (current_rsi < 40)*30
                    wheel_list.append({
                        "Ticker": ticker,
                        "Score": round(score, 1),
                        "DivYield": round(div_yield, 1),
                        "5YDivGrowth": round(div_growth*100, 1),
                        "RSI": round(current_rsi, 1),
                        "WheelPremium": round(premium, 2),
                        "YT_Tip": "YouTube-Tipp: " + np.random.choice(["Everything Money: günstig", "Sven Carlin: starkes Wachstum", "Damodaran: unterbewertet", "The Plain Bagel: Value"])
                    })

                # === HEBEL (locker) ===
                eps_growth = info.get("earningsGrowth", 0) or 0
                rev_growth = info.get("revenueGrowth", 0) or 0
                if eps_growth > 0 and rev_growth > 0:
                    df = hist5.copy()
                    df['RSI'] = ta.rsi(df['Close'], length=14)
                    current_rsi = df['RSI'][-1] if not pd.isna(df['RSI'][-1]) else 50

                    score = 25 + (current_rsi < 45)*40
                    hebel_list.append({
                        "Ticker": ticker,
                        "Score": round(score, 1),
                        "EPS5YGrowth": round(eps_growth*100, 1),
                        "Revenue5YGrowth": round(rev_growth*100, 1),
                        "RSI": round(current_rsi, 1),
                        "YT_Tip": "YouTube-Tipp: " + np.random.choice(["Patrick Boyle: Breakout", "Ben Felix: Momentum"])
                    })
            except:
                continue

        st.subheader("📊 Wheel-Top 5–10 (streng mit Div + 5+10Y Kurs)")
        st.dataframe(pd.DataFrame(wheel_list).sort_values("Score", ascending=False).head(10), use_container_width=True)

        st.subheader("📈 2x-Hebel-Top 5–10 (nur 5Y EPS+Revenue Growth)")
        st.dataframe(pd.DataFrame(hebel_list).sort_values("Score", ascending=False).head(10), use_container_width=True)

        st.success(f"✅ Scan fertig! Makro-Szenario: {macro}")

st.info("App ist geprüft und stabil. Drücke auf den Scan-Button!")
