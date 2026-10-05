"""Treina um modelo para prever o fechamento do BTC no dia seguinte a partir dos
últimos 7 fechamentos e salva o modelo em models/model.joblib.

O modelo exportado é um Pipeline que recebe os 7 fechamentos, converte em 6
log-retornos e prevê o log-retorno de amanhã. O preço previsto é
último_fechamento * exp(log_retorno_previsto), calculado pela API."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

DATA_PATH = Path(os.getenv("DATA_PATH", "/app/data/btc.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/model.joblib"))
N_LAGS = 7
TRAIN_FRAC = 0.8


def montar_dataset(closes: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    # Colunas do mais antigo (lag_7) ao mais recente (lag_1), na mesma ordem do /predict
    X = pd.DataFrame({f"lag_{i}": closes.shift(i - 1) for i in range(N_LAGS, 0, -1)})
    y = closes.shift(-1).rename("target")
    dataset = pd.concat([X, y], axis=1).dropna()
    return dataset.drop(columns="target"), dataset["target"]


def criar_modelo():
    # Funções do numpy (e não funções locais) para o Pipeline ser carregado na API
    # sem precisar deste arquivo
    return make_pipeline(
        FunctionTransformer(np.log),
        FunctionTransformer(np.diff, kw_args={"axis": 1}),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 4, 25), cv=TimeSeriesSplit(n_splits=5)),
    )


def alvo_log_retorno(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    return np.log(y.values / X["lag_1"].values)


def preco_previsto(model, X: pd.DataFrame) -> np.ndarray:
    return X["lag_1"].values * np.exp(model.predict(X.values))


def acuracia_direcao(X: pd.DataFrame, y: pd.Series, previsto: np.ndarray) -> float:
    real_subiu = y.values > X["lag_1"].values
    previsto_subiu = previsto > X["lag_1"].values
    return float(np.mean(real_subiu == previsto_subiu))


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date")
    X, y = montar_dataset(df["Close"].reset_index(drop=True))

    corte = int(len(X) * TRAIN_FRAC)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]

    # Modelo anterior (preço -> preço), mantido só para comparação
    antigo = LinearRegression().fit(X_train.values, y_train.values)
    prev_antigo = antigo.predict(X_test.values)

    model = criar_modelo().fit(X_train.values, alvo_log_retorno(X_train, y_train))
    prev_novo = preco_previsto(model, X_test)

    print(f"Amostras: {len(X)} (treino={len(X_train)}, teste={len(X_test)})")
    print(f"MAE baseline (amanhã = hoje):     {mean_absolute_error(y_test, X_test['lag_1']):,.2f} USD")
    print(f"MAE LinearRegression (preços):    {mean_absolute_error(y_test, prev_antigo):,.2f} USD"
          f" | direção: {acuracia_direcao(X_test, y_test, prev_antigo):.1%}")
    print(f"MAE Ridge (log-retornos):         {mean_absolute_error(y_test, prev_novo):,.2f} USD"
          f" | direção: {acuracia_direcao(X_test, y_test, prev_novo):.1%}")
    print(f"Alpha escolhido pelo RidgeCV: {model[-1].alpha_:.2f}")

    # Após avaliar, retreina com 100% dos dados para exportar o modelo mais atualizado
    model = criar_modelo().fit(X.values, alvo_log_retorno(X, y))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo (treinado com todos os {len(X)} exemplos) salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
