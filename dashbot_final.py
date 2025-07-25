import pandas as pd
import numpy as np
import dash
from dash import html, dcc, Input, Output
import plotly.graph_objs as go
import joblib
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import dash_bootstrap_components as dbc

# === GLOBAL CONFIG ===
DATA_FILE = "yourdata.csv"
MODEL_RF = "model_rf.pkl"
SCALER_FILE = "scaler.pkl"
MAX_HISTORY = 100
ACTIONS = [1.2, 1.5, 1.8, 2.0, 2.5]
q_table = {a: 0 for a in ACTIONS}
initial_data = []

# === Load or Simulate Data ===
def load_data():
    global initial_data
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        initial_data = df["multiplier"].tolist()
    else:
        initial_data = [round(random.expovariate(1/2.0) + 1, 2) for _ in range(MAX_HISTORY)]
        pd.DataFrame(initial_data, columns=["multiplier"]).to_csv(DATA_FILE, index=False)

# === Train RandomForest ===
def train_rf_model():
    df = pd.read_csv(DATA_FILE)
    df["target"] = df["multiplier"].shift(-1).apply(lambda x: 1 if x >= 2 else 0)
    for i in range(1, 6):
        df[f"prev_{i}"] = df["multiplier"].shift(i)
    df.dropna(inplace=True)
    X = df[[f"prev_{i}" for i in range(1, 6)]]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, MODEL_RF)

# === Train Reinforcement Learning (Q-learning) ===
def train_reinforcement():
    global q_table
    q_table = {a: 0 for a in ACTIONS}
    data = pd.read_csv(DATA_FILE)["multiplier"].tolist()

    for _ in range(300):
        for i in range(5, len(data)):
            outcome = data[i]
            action = random.choice(ACTIONS)
            reward = 1 if outcome >= action else -1
            q_table[action] += 0.1 * (reward + 0.9 * max(q_table.values()) - q_table[action])

# === Predict Functions ===
def predict_rf():
    if len(initial_data) < 6:
        return "⚠️ Not enough data", 0.0
    model = joblib.load(MODEL_RF)
    X = np.array(initial_data[-6:-1]).reshape(1, -1)
    prob = model.predict_proba(X)[0][1]
    label = "🟢 GO" if prob > 0.7 else "⚪ WAIT" if prob > 0.4 else "🔴 NO"
    return label, round(prob * 100, 1)

def hybrid_decision():
    rf_label, rf_prob = predict_rf()
    rl_target = max(q_table, key=q_table.get) if q_table else 2  # Default value if q_table is empty
    hybrid_score = (0.85 * (rf_prob / 100)) + (0.15 * (1 if rl_target <= 2 else 0.5))

    decision = "🟢 STRONG ENTRY" if hybrid_score > 0.75 else "⚪ CAUTION" if hybrid_score > 0.55 else "🔴 SKIP ROUND"
    return decision, rf_label, rl_target

# === Load Data Initially ===
load_data()

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
        ], md=9)
    ])
], fluid=True)

@app.callback(
    [Output("live-graph", "figure"),
     Output("signals", "children")],
    [Input("btn-load", "n_clicks"),
     Input("btn-train", "n_clicks"),
     Input("btn-sim", "n_clicks")]
)
def update_graph(n1, n2, n3):
    global initial_data
    ctx = dash.callback_context.triggered_id

    if ctx == "btn-load":
        load_data()
    elif ctx == "btn-train":
        train_rf_model()
        train_reinforcement()
    elif ctx == "btn-sim":
        next_val = round(random.expovariate(1/2.0) + 1, 2)
        initial_data.append(next_val)
        initial_data = initial_data[-MAX_HISTORY:]
        pd.DataFrame(initial_data, columns=["multiplier"]).to_csv(DATA_FILE, index=False)

    # Ensure the graph renders properly even if there's no data yet
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(initial_data))), y=initial_data,
                            mode='lines+markers', name='Crash Multipliers'))
    fig.update_layout(title="Crash History", xaxis_title="Round", yaxis_title="Multiplier")

    hybrid, rf, rl = hybrid_decision()
    text = f"""
    🧠 Signal: {hybrid}   
    🌲 RandomForest: {rf}   
    🎯 Cashout Target (RL): {rl}x   
    """
    return fig, text.replace("  ", "<br>")

# ✅ Final Correct Run Method for Dash 3.x+
if __name__ == "__main__":
    app.run(debug=True)
