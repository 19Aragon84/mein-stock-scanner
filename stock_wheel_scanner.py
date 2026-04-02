import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Wheel + Leverage Scanner v2", layout="wide")
st.title("🚀 Wheel + Leverage Scanner v2")
st.caption("Freie Daten | bessere Optionslogik | Batch-Preisdaten | approximiertes Delta | saubere Scores")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Einstellungen")
universe_choice = st.sidebar.multiselect(
    "Indizes / Universum",
    options=["S&P 500", "Nasdaq-100", "Dow Jones", "DAX"],
    default=["S&P 500", "Nasdaq-100", "Dow Jones"],
)
min_market_cap_b = st.sidebar.number_input("Min. Market Cap (Mrd. USD/EUR laut Yahoo-Wert)", value=10.0, step=1.0)
min_daily_dollar_vol_m = st.sidebar.number_input("Min. Dollar-Volumen pro Tag (Mio.)", value=20.0, step=5.0)
min_div_yield = st.sidebar.number_input("Min. Dividendenrendite für Wheel (%)", value=1.0, step=0.5)
max_dte = st.sidebar.slider("Max. Laufzeit für CSP-Suche (Tage)", min_value=14, max_value=60, value=45)
min_oi = st.sidebar.number_input("Min. Open Interest (Option)", value=100, step=50)
max_bid_ask_pct = st.sidebar.number_input("Max. Bid-Ask-Spread (%)", value=8.0, step=1.0)
macro = st.sidebar.selectbox(
    "Makro-Szenario",
    ["Neutral", "Hohe Ölpreise / Iran", "Schwaches Asien-Wachstum"],
)
debug_mode = st.sidebar.checkbox("Debug-Ausgaben anzeigen", value=False)

# -----------------------------
# Config helpers
# -----------------------------
RISK_FREE_RATE = 0.04  # grobe Annahme für Delta-Näherung
TRADING_DAYS = 252

SECTOR_BONUS = {
    "Neutral": {},
    "Hohe Ölpreise / Iran": {
        "Energy": 12,
        "Oil & Gas": 12,
        "Oil & Gas Midstream": 10,
        "Aerospace & Defense": 8,
        "Industrials": 2,
        "Airlines": -12,
        "Consumer Cyclical": -4,
        "Chemicals": -6,
    },
    "Schwaches Asien-Wachstum": {
        "Utilities": 8,
        "Consumer Defensive": 6,
        "Software": 4,
        "Semiconductors": -6,
        "Luxury Goods": -10,
        "Basic Materials": -8,
        "Industrials": -4,
        "Energy": -4,
    },
}


@dataclass
class UnderlyingSnapshot:
    ticker: str
    price: float
    market_cap: float
    avg_dollar_volume_m: float
    dividend_yield: float
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    revenue_growth: Optional[float]
    earnings_growth: Optional[float]
    beta: Optional[float]
    sector: str
    industry: str
    currency: str
    rsi14: float
    ret_3m: float
    ret_6m: float
    dist_to_52w_high: float
    above_50dma: bool
    above_200dma: bool
    sma50_gt_sma200: bool
    hist_vol_30d: float
    score_macro_bonus: float


# -----------------------------
# Utilities
# -----------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_put_delta(spot: float, strike: float, t: float, r: float, sigma: float) -> Optional[float]:
    if spot <= 0 or strike <= 0 or t <= 0 or sigma <= 0:
        return None
    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        return norm_cdf(d1) - 1.0
    except Exception:
        return None


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def pct(x: Optional[float]) -> float:
    if x is None or pd.isna(x):
        return 0.0
    return float(x) * 100.0


def safe_float(x, default=None):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_linear(value: float, low: float, high: float, invert: bool = False) -> float:
    if high == low:
        return 0.0
    s = (value - low) / (high - low)
    s = clamp(s, 0.0, 1.0)
    return 1.0 - s if invert else s


# -----------------------------
# Universe
# -----------------------------
@st.cache_data(ttl=86400)
def get_universe(selected: Tuple[str, ...]) -> pd.DataFrame:
    rows = []

    def add_df(df: pd.DataFrame, symbol_col: str, source: str, suffix: str = ""):
        for raw in df[symbol_col].astype(str).tolist():
            ticker = raw.replace(".", "-") if source != "DAX" else raw + suffix
            rows.append({"Ticker": ticker, "Source": source})

    if "S&P 500" in selected:
        sp = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        add_df(sp, "Symbol", "S&P 500")
    if "Nasdaq-100" in selected:
        ndx = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[0]
        col = "Ticker" if "Ticker" in ndx.columns else "Company"
        if col == "Ticker":
            add_df(ndx, col, "Nasdaq-100")
    if "Dow Jones" in selected:
        dow = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")[1]
        add_df(dow, "Symbol", "Dow Jones")
    if "DAX" in selected:
        dax = pd.read_html("https://en.wikipedia.org/wiki/DAX")[4]
        col = "Ticker symbol" if "Ticker symbol" in dax.columns else dax.columns[-1]
        add_df(dax, col, "DAX", suffix=".DE")

    uni = pd.DataFrame(rows).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return uni


# -----------------------------
# Price and indicator layer
# -----------------------------
@st.cache_data(ttl=3600)
def load_price_history(tickers: Tuple[str, ...], period: str = "18mo") -> pd.DataFrame:
    data = yf.download(
        tickers=list(tickers),
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return data


def extract_close_volume(download_df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    try:
        if isinstance(download_df.columns, pd.MultiIndex):
            sub = download_df[ticker][["Close", "Volume"]].dropna().copy()
        else:
            sub = download_df[["Close", "Volume"]].dropna().copy()
        if len(sub) < 80:
            return None
        return sub
    except Exception:
        return None


def compute_price_features(pxv: pd.DataFrame) -> Dict[str, float]:
    close = pxv["Close"]
    vol = pxv["Volume"]
    rsi14 = float(compute_rsi(close, 14).iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.rolling(100).mean().iloc[-1])
    last = float(close.iloc[-1])
    ret_3m = float(last / close.iloc[max(0, len(close) - 63)] - 1) if len(close) > 63 else 0.0
    ret_6m = float(last / close.iloc[max(0, len(close) - 126)] - 1) if len(close) > 126 else 0.0
    high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
    dist_to_52w_high = float(last / high_52w - 1) if high_52w > 0 else 0.0
    hist_vol_30d = float(close.pct_change().tail(30).std() * np.sqrt(TRADING_DAYS))
    avg_dollar_volume_m = float((close.tail(20) * vol.tail(20)).mean() / 1e6)
    return {
        "price": last,
        "rsi14": rsi14,
        "sma50": sma50,
        "sma200": sma200,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "dist_to_52w_high": dist_to_52w_high,
        "hist_vol_30d": hist_vol_30d,
        "avg_dollar_volume_m": avg_dollar_volume_m,
    }


# -----------------------------
# Fundamentals layer
# -----------------------------
def get_info_fast(ticker: str) -> Dict:
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    fast = getattr(tk, "fast_info", {}) or {}
    return {"ticker_obj": tk, "info": info, "fast": fast}


def macro_bonus(sector: str, industry: str, macro_name: str) -> float:
    bonus_map = SECTOR_BONUS.get(macro_name, {})
    bonus = 0.0
    for key, val in bonus_map.items():
        if key.lower() in (sector or "").lower() or key.lower() in (industry or "").lower():
            bonus += val
    return bonus


def build_underlying_snapshot(ticker: str, px_features: Dict[str, float], info_pack: Dict) -> Optional[UnderlyingSnapshot]:
    info = info_pack["info"]
    market_cap = safe_float(info.get("marketCap"), 0.0)
    sector = str(info.get("sector", ""))
    industry = str(info.get("industry", ""))
    currency = str(info.get("currency", ""))
    dividend_yield = pct(info.get("dividendYield"))
    trailing_pe = safe_float(info.get("trailingPE"))
    forward_pe = safe_float(info.get("forwardPE"))
    revenue_growth = safe_float(info.get("revenueGrowth"))
    earnings_growth = safe_float(info.get("earningsGrowth"))
    beta = safe_float(info.get("beta"))

    return UnderlyingSnapshot(
        ticker=ticker,
        price=px_features["price"],
        market_cap=market_cap,
        avg_dollar_volume_m=px_features["avg_dollar_volume_m"],
        dividend_yield=dividend_yield,
        trailing_pe=trailing_pe,
        forward_pe=forward_pe,
        revenue_growth=revenue_growth,
        earnings_growth=earnings_growth,
        beta=beta,
        sector=sector,
        industry=industry,
        currency=currency,
        rsi14=px_features["rsi14"],
        ret_3m=px_features["ret_3m"],
        ret_6m=px_features["ret_6m"],
        dist_to_52w_high=px_features["dist_to_52w_high"],
        above_50dma=px_features["price"] > px_features["sma50"],
        above_200dma=px_features["price"] > px_features["sma200"],
        sma50_gt_sma200=px_features["sma50"] > px_features["sma200"],
        hist_vol_30d=px_features["hist_vol_30d"],
        score_macro_bonus=macro_bonus(sector, industry, macro),
    )


# -----------------------------
# Option layer
# -----------------------------
def parse_expiry_days(expiry_str: str) -> int:
    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    return max((expiry - date.today()).days, 0)


def option_mid(bid: float, ask: float, last: float) -> float:
    if bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0
    return max(last, 0.0)


def scan_best_put_for_wheel(tk: yf.Ticker, snap: UnderlyingSnapshot) -> Optional[Dict]:
    try:
        expiries = tk.options
        if not expiries:
            return None
    except Exception:
        return None

    candidates = []
    for expiry in expiries:
        dte = parse_expiry_days(expiry)
        if dte < 14 or dte > max_dte:
            continue
        try:
            chain = tk.option_chain(expiry)
            puts = chain.puts.copy()
        except Exception:
            continue

        if puts.empty:
            continue

        for _, row in puts.iterrows():
            strike = safe_float(row.get("strike"), 0.0)
            bid = safe_float(row.get("bid"), 0.0)
            ask = safe_float(row.get("ask"), 0.0)
            last = safe_float(row.get("lastPrice"), 0.0)
            oi = safe_float(row.get("openInterest"), 0.0)
            vol = safe_float(row.get("volume"), 0.0)
            iv = safe_float(row.get("impliedVolatility"), 0.0)
            if strike <= 0 or oi < min_oi:
                continue
            mid = option_mid(bid, ask, last)
            if mid <= 0:
                continue
            spread_pct = ((ask - bid) / mid * 100.0) if mid > 0 and ask >= bid and bid >= 0 else 999.0
            if spread_pct > max_bid_ask_pct:
                continue
            t = dte / 365.0
            delta = black_scholes_put_delta(snap.price, strike, t, RISK_FREE_RATE, iv if iv and iv > 0 else max(snap.hist_vol_30d, 0.15))
            if delta is None:
                continue
            # Zielbereich grob Delta -0.15 bis -0.35
            if not (-0.35 <= delta <= -0.15):
                continue

            annualized_yield = (mid / strike) * (365.0 / dte) * 100.0
            buffer_pct = (strike / snap.price - 1.0) * 100.0
            iv_hv_ratio = (iv / snap.hist_vol_30d) if snap.hist_vol_30d > 0 and iv > 0 else 1.0

            score = 0.0
            score += score_linear(annualized_yield, 6, 20) * 30
            score += score_linear(abs(delta), 0.15, 0.35, invert=True) * 20
            score += score_linear(buffer_pct, -15, -3) * 15
            score += score_linear(iv_hv_ratio, 0.9, 1.8) * 15
            score += score_linear(oi, min_oi, max(min_oi * 8, 1000)) * 10
            score += score_linear(snap.dividend_yield, min_div_yield, 6.0) * 5
            score += score_linear(snap.rsi14, 25, 50, invert=True) * 5

            candidates.append(
                {
                    "Ticker": snap.ticker,
                    "Expiry": expiry,
                    "DTE": dte,
                    "Strike": round(strike, 2),
                    "Spot": round(snap.price, 2),
                    "DeltaApprox": round(delta, 3),
                    "Mid": round(mid, 2),
                    "AnnualizedYield%": round(annualized_yield, 1),
                    "BufferToSpot%": round(buffer_pct, 1),
                    "OI": int(oi),
                    "Volume": int(vol),
                    "Spread%": round(spread_pct, 1),
                    "IV": round(iv * 100, 1) if iv else None,
                    "HV30": round(snap.hist_vol_30d * 100, 1),
                    "IV/HV": round(iv_hv_ratio, 2),
                    "DivYield%": round(snap.dividend_yield, 1),
                    "RSI14": round(snap.rsi14, 1),
                    "Sector": snap.sector,
                    "Score": round(score, 1),
                    "Warum": "CSP: Rendite, Liquidität, Delta-Nähe, Buffer, IV/HV"
                }
            )

    if not candidates:
        return None
    df = pd.DataFrame(candidates).sort_values(["Score", "AnnualizedYield%"], ascending=[False, False])
    return df.iloc[0].to_dict()


# -----------------------------
# Leverage layer
# -----------------------------
def score_leverage_candidate(snap: UnderlyingSnapshot) -> Optional[Dict]:
    # harte Mindestqualität für Hebel-Setups
    if snap.market_cap < min_market_cap_b * 1e9:
        return None
    if snap.avg_dollar_volume_m < min_daily_dollar_vol_m:
        return None
    if not (snap.above_50dma and snap.sma50_gt_sma200):
        return None

    valuation_anchor = snap.forward_pe if snap.forward_pe is not None else snap.trailing_pe
    valuation_score = score_linear(valuation_anchor if valuation_anchor else 40, 10, 35, invert=True) * 20
    growth_score = score_linear(pct(snap.earnings_growth) + pct(snap.revenue_growth), 5, 35) * 20
    momentum_score = score_linear(snap.ret_3m * 100, 0, 25) * 15 + score_linear(snap.ret_6m * 100, 0, 40) * 10
    trend_score = (10 if snap.above_200dma else 0) + (10 if snap.above_50dma else 0) + (10 if snap.sma50_gt_sma200 else 0)
    pullback_score = score_linear(abs(snap.dist_to_52w_high) * 100, 0, 12, invert=True) * 10
    rsi_score = score_linear(snap.rsi14, 40, 62) * 10
    volatility_penalty = score_linear(snap.hist_vol_30d * 100, 25, 65) * 10

    score = valuation_score + growth_score + momentum_score + trend_score + pullback_score + rsi_score - volatility_penalty + snap.score_macro_bonus

    if score < 40:
        return None

    return {
        "Ticker": snap.ticker,
        "Score": round(score, 1),
        "Price": round(snap.price, 2),
        "ForwardPE": round(snap.forward_pe, 1) if snap.forward_pe is not None else None,
        "TrailingPE": round(snap.trailing_pe, 1) if snap.trailing_pe is not None else None,
        "EPSGrowth%": round(pct(snap.earnings_growth), 1),
        "RevGrowth%": round(pct(snap.revenue_growth), 1),
        "RSI14": round(snap.rsi14, 1),
        "Ret3M%": round(snap.ret_3m * 100, 1),
        "Ret6M%": round(snap.ret_6m * 100, 1),
        "DistTo52WHigh%": round(snap.dist_to_52w_high * 100, 1),
        "Vol30%": round(snap.hist_vol_30d * 100, 1),
        "Sector": snap.sector,
        "Industry": snap.industry,
        "MacroAdj": round(snap.score_macro_bonus, 1),
        "Warum": "Hebel: Trend + Growth + akzeptable Bewertung + Makrofit"
    }


# -----------------------------
# Main scan
# -----------------------------
selected_tuple = tuple(universe_choice)
universe_df = get_universe(selected_tuple) if selected_tuple else pd.DataFrame(columns=["Ticker", "Source"])
tickers = universe_df["Ticker"].tolist()

st.write(f"Universum: **{len(tickers)}** Ticker")
with st.expander("Ticker-Preview"):
    st.dataframe(universe_df.head(50), use_container_width=True)

if st.button("🔥 Scan starten", type="primary"):
    if not tickers:
        st.warning("Bitte mindestens ein Universum auswählen.")
        st.stop()

    errors: List[Dict] = []
    underlyings: List[UnderlyingSnapshot] = []
    wheel_rows: List[Dict] = []
    leverage_rows: List[Dict] = []

    with st.spinner("Lade Preisdaten..."):
        px_all = load_price_history(tuple(tickers), period="18mo")

    progress = st.progress(0)
    total = len(tickers)

    for i, ticker in enumerate(tickers, start=1):
        try:
            pxv = extract_close_volume(px_all, ticker)
            if pxv is None:
                continue
            px_features = compute_price_features(pxv)
            info_pack = get_info_fast(ticker)
            snap = build_underlying_snapshot(ticker, px_features, info_pack)
            if snap is None:
                continue
            if snap.market_cap < min_market_cap_b * 1e9:
                continue
            if snap.avg_dollar_volume_m < min_daily_dollar_vol_m:
                continue
            underlyings.append(snap)

            # Wheel: eher defensivere/qualitativere Underlyings
            if snap.dividend_yield >= min_div_yield and snap.rsi14 <= 55:
                wheel_best = scan_best_put_for_wheel(info_pack["ticker_obj"], snap)
                if wheel_best:
                    wheel_rows.append(wheel_best)

            lev = score_leverage_candidate(snap)
            if lev:
                leverage_rows.append(lev)
        except Exception as e:
            errors.append({"Ticker": ticker, "Error": str(e)})
        progress.progress(i / total)

    st.success(f"Scan fertig. Underlyings nach Basisfiltern: {len(underlyings)} | Fehler: {len(errors)}")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📊 Wheel-Top 10")
        if wheel_rows:
            wheel_df = pd.DataFrame(wheel_rows).sort_values(["Score", "AnnualizedYield%"], ascending=[False, False])
            st.dataframe(wheel_df.head(10), use_container_width=True)
        else:
            st.info("Keine Wheel-Kandidaten gefunden.")

    with c2:
        st.subheader("📈 Leverage-Top 10")
        if leverage_rows:
            lev_df = pd.DataFrame(leverage_rows).sort_values(["Score", "Ret3M%"], ascending=[False, False])
            st.dataframe(lev_df.head(10), use_container_width=True)
        else:
            st.info("Keine Hebel-Kandidaten gefunden.")

    st.subheader("📦 Basis-Universum nach Filters")
    if underlyings:
        base_df = pd.DataFrame([
            {
                "Ticker": s.ticker,
                "Price": round(s.price, 2),
                "MarketCapBn": round(s.market_cap / 1e9, 1),
                "DollarVolM": round(s.avg_dollar_volume_m, 1),
                "DivYield%": round(s.dividend_yield, 1),
                "ForwardPE": round(s.forward_pe, 1) if s.forward_pe is not None else None,
                "RSI14": round(s.rsi14, 1),
                "Ret3M%": round(s.ret_3m * 100, 1),
                "Ret6M%": round(s.ret_6m * 100, 1),
                "Sector": s.sector,
                "Industry": s.industry,
            }
            for s in underlyings
        ])
        st.dataframe(base_df.sort_values(["DollarVolM", "MarketCapBn"], ascending=[False, False]), use_container_width=True)

    if debug_mode:
        st.subheader("🛠️ Fehler / Debug")
        if errors:
            st.dataframe(pd.DataFrame(errors), use_container_width=True)
        else:
            st.write("Keine Fehler protokolliert.")

st.info(
    "Hinweis: kostenlose Daten sind gut für Screening, aber nicht perfekt. Das Delta ist hier Black-Scholes-approximiert aus Spot, Strike, DTE und IV/HV. Für echtes Live-Options-Scoring später besser IBKR oder Polygon anbinden."
)

st.markdown(
    """
### Installation
```bash
pip install streamlit yfinance pandas numpy lxml html5lib
streamlit run stock_option_scanner_v2.py
```

### Nächste sinnvolle Erweiterungen
- Earnings-Kalender einbauen
- Calls für Covered-Call-Scanning ergänzen
- IV Rank / IV Percentile über Zeitreihen ergänzen
- Relative Strength vs. SPY / QQQ / XLK / XLE usw.
- CSV-Export / Watchlist-Export
- IBKR-Adapter für echte Optionsdaten
``` 
"""
)
