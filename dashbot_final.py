import pandas as pd
import numpy as np
import dash
from dash import html, dcc, Input, Output
import plotly.graph_objs as go
import joblib
import os
import random
from sklearn.ensemble import RandomForestClassifier
import dash_bootstrap_components as dbc
import xgboost as xgb  # Added XGBoost for better performance

# === GLOBAL CONFIG ===
DATA_FILE = "yourdata.csv"
MODEL_RF = "model_rf.pkl"
MODEL_XGB = "model_xgb.pkl"  # New model for XGBoost
SCALER_FILE = "scaler.pkl"
MAX_HISTORY = 100
ACTIONS = [1.2, 1.5, 1.8, 2.0, 2.5]
q_table = {a: 0 for a in ACTIONS}
initial_data = []

# === Load Data with 20 History Rounds ===
def load_data():
    global initial_data
    # Here we are feeding the bot with 20 rounds of data
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        initial_data = df["multiplier"].tolist()[:20]  # Only keep 20 rounds of history
    else:
        # Simulating 20 rounds if no history is present
        initial_data = [round(random.expovariate(1/2.0) + 1, 2) for _ in range(20)]
        pd.DataFrame(initial_data, columns=["multiplier"]).to_csv(DATA_FILE, index=False)

# === Feature Engineering Function ===
def create_features(df):
    # Adding additional features for better prediction
    df['previous'] = df['multiplier'].shift(1)
    df['change'] = df['multiplier'] - df['previous']
    df['rolling_mean'] = df['multiplier'].rolling(window=5).mean()
    df['rolling_std'] = df['multiplier'].rolling(window=5).std()
    df['volatility'] = (df['multiplier'] - df['rolling_mean']) / df['rolling_std']
    df = df.dropna()
    return df

# === Train RandomForest and XGBoost Models ===
def train_models():
    df = pd.read_csv(DATA_FILE)
    df = create_features(df)

    X = df[['previous', 'change', 'rolling_mean', 'rolling_std', 'volatility']]
    y = (df["multiplier"].shift(-1) >= 2).astype(int)  # Binary target: 1 if multiplier >= 2, else 0

    # Train RandomForest
    model_rf = RandomForestClassifier(n_estimators=100)
    model_rf.fit(X, y)
    joblib.dump(model_rf, MODEL_RF)

    # Train XGBoost
    model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
    model_xgb.fit(X, y)
    joblib.dump(model_xgb, MODEL_XGB)

# === Predict Functions ===
def predict_rf():
    model_rf = joblib.load(MODEL_RF)
    X = np.array(initial_data[-6:-1]).reshape(1, -1)
    prob = model_rf.predict_proba(X)[0][1]
    label = "🟢 GO" if prob > 0.7 else "⚪ WAIT" if prob > 0.4 else "🔴 NO"
    return label, round(prob * 100, 1)

def predict_xgb():
    model_xgb = joblib.load(MODEL_XGB)
    X = np.array(initial_data[-6:-1]).reshape(1, -1)
    prob = model_xgb.predict_proba(X)[0][1]
    label = "🟢 GO" if prob > 0.7 else "⚪ WAIT" if prob > 0.4 else "🔴 NO"
    return label, round(prob * 100, 1)

# === Simulate Round Function ===
def simulate_round():
    # Simulate the "Crash" game
    crash_point = round(random.expovariate(1/2.0) + 1, 2)  # The point at which the game crashes
    multiplier = 1.0  # The current multiplier
    while multiplier < crash_point:
        multiplier += random.uniform(0.05, 0.1)  # Increment multiplier over time (simulating crash)
    return crash_point, round(multiplier, 2)

# === Dash App ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Aviator AI DashBot"

app.layout = dbc.Container([
    html.H1("✈️ Aviator AI DashBot – Final Boss Edition", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            html.Button("📡 Load Data", id="btn-load", n_clicks=0, className="btn btn-primary w-100 mb-2"),
            html.Button("🎓 Train Models", id="btn-train", n_clicks=0, className="btn btn-secondary w-100 mb-2"),
            html.Button("🎯 Simulate Round", id="btn-sim", n_clicks=0, className="btn btn-success w-100 mb-2"),
        ], md=3),

        dbc.Col([
            dcc.Graph(id="live-graph"),
            html.Div(id="signals", className="text-center fs-4 mt-4"),
            html.Div(id="round-info", className="text-center mt-4"),
        ], md=9)
    ]),
    dcc.Interval(id="interval-update", interval=1000, n_intervals=0)  # 1 second interval for graph update
], fluid=True)

@app.callback(
    [Output("live-graph", "figure"),
     Output("signals", "children"),
     Output("round-info", "children")],
    [Input("btn-load", "n_clicks"),
     Input("btn-train", "n_clicks"),
     Input("btn-sim", "n_clicks"),
     Input("interval-update", "n_intervals")]
)
def update_graph(n1, n2, n3, n_intervals):
    global initial_data
    ctx = dash.callback_context.triggered_id
    if not ctx:
        return dash.no_update  # Prevents an error if no button is pressed
    
    crash_point = None
    multiplier = 1.0

    if ctx == "btn-load":
        load_data()
    elif ctx == "btn-train":
        train_models()
    elif ctx == "btn-sim":
        crash_point, multiplier = simulate_round()
        initial_data.append(multiplier)
        initial_data = initial_data[-MAX_HISTORY:]
        pd.DataFrame(initial_data, columns=["multiplier"]).to_csv(DATA_FILE, index=False)

    # Update the graph with new multipliers
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(initial_data))), y=initial_data,
                            mode='lines+markers', name='Crash Multipliers'))
    fig.update_layout(title="Crash History", xaxis_title="Round", yaxis_title="Multiplier")

    # Display the hybrid decision, Random Forest, and Q-learning outputs
    rf_label, rf_prob = predict_rf()
    xgb_label, xgb_prob = predict_xgb()

    hybrid_decision = f"🧠 Hybrid Decision: {rf_label} | XGBoost: {xgb_label}"

    round_info = f"💥 Round ended with crash point at {crash_point}x, current multiplier: {multiplier}x"
    
    return fig, hybrid_decision, round_info

# ✅ Final Correct Run Method for Dash 3.x+
if __name__ == "__main__":
    app.run(debug=True)
