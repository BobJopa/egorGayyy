"""Stock Market Analysis Dashboard with Dash
Исправленная версия с поддержкой многоколоночных данных
"""

import datetime as dt
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ============== ИНДИКАТОРЫ ==============
# INDICATORS: SMA, EMA, RSI, MACD, BOLLINGER, ATR


def parse_tickers(raw_value: str) -> List[str]:
    return [item.strip().upper() for item in raw_value.split(",") if item.strip()]


@lru_cache(maxsize=64)
def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Загрузить данные одного тикера (обработка MultiIndex)"""
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
        
        if data.empty:
            print(f"⚠️ Нет данных для {ticker}")
            return pd.DataFrame()
        
        # Проверяем и убираем MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            # Берём только первый уровень (названия колонок)
            data.columns = data.columns.get_level_values(0)
        
        # Теперь безопасно применяем .str
        data.columns = data.columns.str.lower()
        data = data.rename(
            columns={
                "adj close": "Adj Close",
                "close": "Close",
                "high": "High",
                "low": "Low",
                "open": "Open",
                "volume": "Volume",
            }
        )
        
        print(f"✅ {ticker}: {len(data)} свечей загружено")
        return data
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить технические индикаторы (исправленная версия)"""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "Close" not in df.columns:
        return df

    # SMA / EMA
    df["Sma_20"] = df["Close"].rolling(window=20).mean()
    df["Ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["Rsi_14"] = 100 - (100 / (1 + rs))

    # EMA 12/26 (MACD)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["Macd"] = ema12 - ema26
    df["MacdSignal"] = df["Macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (исправлено)
    rolling_std = df["Close"].rolling(window=20).std()
    if isinstance(rolling_std, pd.DataFrame):
        rolling_std = rolling_std.iloc[:, 0]

    df["BollingerUpper"] = df["Sma_20"] + (rolling_std * 2)
    df["BollingerLower"] = df["Sma_20"] - (rolling_std * 2)

    # ATR
    if {"High", "Low", "Close"} <= set(df.columns):
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["Atr_14"] = true_range.rolling(window=14).mean()

    return df


def indicator_scores(df: pd.DataFrame) -> Dict[str, float]:
    """Оценить качество каждого индикатора"""
    scores: Dict[str, float] = {}

    if df is None or df.empty or "Close" not in df.columns:
        return scores

    # Целевой сигнал (цена выросла?)
    returns = df["Close"].pct_change().shift(-1) > 0

    # SMA
    if "Sma_20" in df:
        signal = np.where(df["Close"] > df["Sma_20"], 1, -1)
        scores["SMA"] = np.nanmean(signal == returns)

    # EMA
    if "Ema_20" in df:
        signal = np.where(df["Close"] > df["Ema_20"], 1, -1)
        scores["EMA"] = np.nanmean(signal == returns)

    # RSI
    if "Rsi_14" in df:
        signal = np.where(
            df["Rsi_14"] < 30,
            1,
            np.where(df["Rsi_14"] > 70, -1, 0),
        )
        scores["RSI"] = np.nanmean(signal == returns)

    # MACD
    if "Macd" in df and "MacdSignal" in df:
        signal = np.where(df["Macd"] > df["MacdSignal"], 1, -1)
        scores["MACD"] = np.nanmean(signal == returns)

    # Bollinger
    if "BollingerUpper" in df and "BollingerLower" in df:
        signal = np.where(
            df["Close"] < df["BollingerLower"],
            1,
            np.where(df["Close"] > df["BollingerUpper"], -1, 0),
        )
        scores["BOLLINGER"] = np.nanmean(signal == returns)

    # ATR
    if "Atr_14" in df:
        atr_median = df["Atr_14"].median()
        signal = np.where(df["Atr_14"] < atr_median, 1, -1)
        scores["ATR"] = np.nanmean(signal == returns)

    return {name: score for name, score in scores.items() if not np.isnan(score)}


def run_tournament(
    scores: Dict[str, float],
    winners: int = 4,
) -> List[Tuple[str, float]]:
    """Выбрать топ-N индикаторов"""
    if not scores:
        return []

    sorted_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return sorted_scores[:winners]


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Построить признаки для нейросети"""
    required_cols = [
        "Sma_20",
        "Ema_20",
        "Rsi_14",
        "Macd",
        "MacdSignal",
        "BollingerUpper",
        "BollingerLower",
        "Atr_14",
    ]
    if not set(required_cols) <= set(df.columns):
        return pd.DataFrame(), pd.Series(dtype=int)

    features = df[required_cols].copy()
    target = (df["Close"].shift(-1) > df["Close"]).astype(int)

    features = features.dropna()
    target = target.loc[features.index]

    return features, target


def predict_growth_probability(df: pd.DataFrame) -> float:
    """Предсказать вероятность роста цены с помощью MLP"""
    features, target = build_features(df)

    if features.empty or len(features) < 50:
        return float("nan")

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = target.values

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )

    model.fit(X[:-1], y[:-1])

    last_features = scaler.transform(features.iloc[-1:])
    probability = model.predict_proba(last_features)[0, 1]

    return float(probability)


def plot_indicators(df: pd.DataFrame) -> go.Figure:
    """Построить график индикаторов"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Close",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Sma_20"],
            name="SMA 20",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Ema_20"],
            name="EMA 20",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BollingerUpper"],
            name="Bollinger Upper",
            line=dict(dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BollingerLower"],
            name="Bollinger Lower",
            line=dict(dash="dot"),
        )
    )

    fig.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20))

    return fig


def plot_candles(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Построить свечной график"""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker,
            )
        ]
    )

    fig.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))

    return fig


def trading_view_iframe(symbol: str) -> html.Iframe:
    """Вставка TradingView-графика"""
    base = "https://s.tradingview.com/widget/embed"
    src = (
        f"{base}?symbol={symbol}"
        "&interval=D&hidesidetoolbar=1&symboledit=1&saveimage=1"
        "&toolbar=bg_f1f3f6&studies&theme=light"
    )

    return html.Iframe(
        src=src,
        style={"width": "100%", "height": "520px", "border": "0"},
    )


def datatable(df: pd.DataFrame) -> DataTable:
    """Создать JSON-сериализуемую таблицу данных для Dash"""
    df = df.reset_index()  # индекс Date в обычный столбец
    return DataTable(
        data=df.to_dict("records"),
        columns=[{"name": str(col), "id": str(col)} for col in df.columns],
        page_size=5,
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "bold"},
    )


# ============== DASH ПРИЛОЖЕНИЕ ==============

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(
                            "📊 Stock Market Analysis Dashboard",
                            className="mb-4",
                        ),
                        html.P("Технический анализ акций с нейросетевым предсказанием"),
                    ]
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Тикеры"),
                        dbc.Input(
                            id="tickers-input",
                            value="AAPL, MSFT",
                        ),
                    ]
                ),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Диапазон дат"),
                        dcc.DatePickerRange(
                            id="date-range",
                            start_date=dt.date.today() - dt.timedelta(days=365),
                            end_date=dt.date.today(),
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Label("TradingView"),
                        dbc.Input(id="tv-symbol", value="NASDAQ:AAPL"),
                    ]
                ),
            ],
            className="mb-3",
        ),
        dbc.Alert(
            "⚠ Выберите параметры выше",
            color="warning",
            id="alert",
        ),
        dcc.Tabs(
            id="tabs",
            value="data",
            children=[
                dcc.Tab(label="📈 Raw Data", value="data"),
                dcc.Tab(label="📊 Indicators", value="indicators"),
                dcc.Tab(label="🏆 Tournament", value="tournament"),
                dcc.Tab(label="🧠 Neural", value="neural"),
                dcc.Tab(label="📉 Candles", value="candles"),
            ],
        ),
        html.Div(id="tabs-content"),
    ],
    fluid=True,
)


@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value"),
    Input("tickers-input", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("tv-symbol", "value"),
)
def render_tab(tab, tickers_raw, start_date, end_date, tv_symbol):
    """Рендер содержимого вкладок"""
    tickers = parse_tickers(tickers_raw or "")

    if not tickers:
        return dbc.Alert(
            "⚠ Пожалуйста, введите хотя бы один тикер",
            color="warning",
        )

    content: List[html.Component] = []

    if tab in {"data", "indicators"}:
        for ticker in tickers:
            data = add_indicators(fetch_prices(ticker, start_date, end_date))
            content.append(html.H5(f"{ticker}"))

            if not data.empty:
                content.append(dcc.Graph(figure=plot_indicators(data)))
                content.append(datatable(data.tail()))
            else:
                content.append(
                    dbc.Alert(f"⚠ Нет данных для {ticker}", color="warning")
                )

    elif tab == "tournament":
        for ticker in tickers:
            data = add_indicators(fetch_prices(ticker, start_date, end_date))
            scores = indicator_scores(data)
            winners = run_tournament(scores, winners=4)

            content.append(html.H5(f"{ticker}"))

            if winners:
                winner_df = pd.DataFrame(winners, columns=["Indicator", "Score"])
                content.append(datatable(winner_df))
            else:
                content.append(
                    dbc.Alert(
                        f"⚠ Нет результатов для {ticker}",
                        color="warning",
                    )
                )

    elif tab == "neural":
        for ticker in tickers:
            data = add_indicators(fetch_prices(ticker, start_date, end_date))

            content.append(html.H5(f"{ticker}"))

            if not data.empty:
                probability = predict_growth_probability(data)

                if not np.isnan(probability):
                    content.append(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H6("Growth Probability"),
                                        html.H3(f"{probability * 100:.2f}%"),
                                        html.P(
                                            "Predicted growth probability by neural network"
                                        ),
                                    ]
                                )
                            ]
                        )
                    )
                else:
                    content.append(
                        dbc.Alert(
                            f"⚠ Недостаточно данных для {ticker}",
                            color="warning",
                        )
                    )
            else:
                content.append(
                    dbc.Alert(f"⚠ Нет данных для {ticker}", color="warning")
                )

    elif tab == "candles":
        for ticker in tickers:
            data = fetch_prices(ticker, start_date, end_date)
            content.append(html.H5(f"{ticker}"))

            if not data.empty:
                content.append(dcc.Graph(figure=plot_candles(data, ticker)))
                content.append(html.H5("TradingView"))
                content.append(trading_view_iframe(tv_symbol))
            else:
                content.append(
                    dbc.Alert(f"⚠ Нет данных для {ticker}", color="warning")
                )

    return html.Div(content)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)