"""Baixa o histórico diário de BTC-USD (Yahoo Finance) dos últimos 3 anos
e salva em data/btc.csv com as colunas Date e Close."""

from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

TICKER = "BTC-USD"
PERIODO = "3y"
SAIDA = Path(__file__).parent / "data" / "btc.csv"


def baixar_yfinance() -> pd.DataFrame:
    df = yf.download(TICKER, period=PERIODO, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("yfinance retornou vazio")
    # yfinance recente devolve colunas MultiIndex (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Close"]]
    return df


def baixar_api_direta() -> pd.DataFrame:
    """Plano B: chama a API de chart do Yahoo diretamente."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}"
    resp = requests.get(
        url,
        params={"range": PERIODO, "interval": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    resp.raise_for_status()
    resultado = resp.json()["chart"]["result"][0]
    datas = pd.to_datetime(resultado["timestamp"], unit="s").normalize()
    fechamentos = resultado["indicators"]["quote"][0]["close"]
    return pd.DataFrame({"Date": datas, "Close": fechamentos})


def main() -> None:
    try:
        df = baixar_yfinance()
    except Exception as erro:
        print(f"yfinance falhou ({erro}); tentando API direta do Yahoo...")
        df = baixar_api_direta()

    df = df.dropna(subset=["Close"]).drop_duplicates(subset="Date").sort_values("Date")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA, index=False)
    print(f"{len(df)} linhas salvas em {SAIDA} ({df['Date'].iloc[0]} a {df['Date'].iloc[-1]})")


if __name__ == "__main__":
    main()
