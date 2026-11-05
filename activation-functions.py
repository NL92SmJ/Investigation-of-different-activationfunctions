# Module importieren
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# Ordner für die Plots erstellen, falls er nicht existiert
os.makedirs("activation_plots", exist_ok=True)

# Daten einlesen & Vorverarbeitung (einmalig vor den Tests)
path = "../Data/iris.csv"
data = pd.read_csv(path, delimiter=',')

# Korrekturen / Spaltenanpassung
data = data.drop(['sepal.width'], axis=1)
col = data['sepal.length']
data = data.drop(['sepal.length'], axis=1)

conv_ohe = ['species']
data = pd.get_dummies(data, columns=conv_ohe, dtype=float)

s_scaler = StandardScaler()
cols_to_s_scale = ['petal.length', 'petal.width']
data[cols_to_s_scale] = s_scaler.fit_transform(data[cols_to_s_scale])

# Train-Test-Split
train_data, test_data, train_col, test_col = train_test_split(data, col, test_size=0.2, random_state=42)

# Liste der zu testenden Aktivierungsfunktionen
# (Leaky ReLU: Nutzung eines Layer-Objekts, für den Rest Strings)
activations_to_test = [
    ("ReLU", "relu"),
    ("Leaky_ReLU", lambda: tf.keras.layers.LeakyReLU(alpha=0.1)),
    ("ELU", "elu"),
    ("SeLU", "selu"),
    ("GELU", "gelu"),
    ("Swish", "swish"),
    ("Tanh", "tanh"),
    ("Sigmoid", "sigmoid")
]

# Dictionary zum Sammeln der Endergebnisse
results = {}

# Schleife über alle Aktivierungsfunktionen
for name, act in activations_to_test:
    print(f"\n---> Starte Training mit Aktivierungsfunktion: {name}")
    
    # 1. Modell frisch aufbauen
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(data.shape[1],)))
    
    # Unterscheidung zwischen klassischem String und speziellem Layer (wie Leaky ReLU)
    if callable(act):
        model.add(tf.keras.layers.Dense(32))
        model.add(act())
        model.add(tf.keras.layers.Dense(64))
        model.add(act())
    else:
        model.add(tf.keras.layers.Dense(32, activation=act))
        model.add(tf.keras.layers.Dense(64, activation=act))
        
    model.add(tf.keras.layers.Dense(1))

    # 2. Kompilieren
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])

    # 3. Training (Genau einmal pro Funktion, 70 Epochen)
    history = model.fit(
        train_data, train_col,
        epochs=70,
        batch_size=32,
        validation_split=0.2,
        verbose=0  # verbose=0 hält das Terminal sauber, auf 1 stellen für Live-Fortschritt
    )

    # 4. Vorhersagen & Metriken berechnen
    predictions = model.predict(test_data)
    mae = mean_absolute_error(test_col, predictions)
    r2 = r2_score(test_col, predictions)
    results[name] = {"mae": mae, "r2": r2}
    errors = predictions.flatten() - test_col.values

    # 5. Plot 1: Loss-Verlauf (Training vs Validation) speichern
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Verlauf ({name})')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(f"activation_plots/loss_{name.lower()}.png")
    plt.close()

    # 6. Plot 2: Prediction vs Reality speichern
    plt.figure(figsize=(6, 6))
    plt.scatter(test_col, predictions, alpha=0.7)
    plt.plot([test_col.min(), test_col.max()], [test_col.min(), test_col.max()], 'r--')
    plt.xlabel("Echte Werte")
    plt.ylabel("Vorhersage")
    plt.title(f"Prediction vs Reality ({name})")
    plt.savefig(f"activation_plots/pred_{name.lower()}.png")
    plt.close()



    # 7. Plot 3: Fehlerverteilung (Histogramm)
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=20, edgecolor='black')
    plt.title(f"Fehlerverteilung ({name})")
    plt.xlabel("Vorhersagefehler")
    plt.ylabel("Anzahl")
    plt.savefig(f"activation_plots/error_hist_{name.lower()}.png")
    plt.close()

    # 8. Plot 4: Residual Plot
    plt.figure(figsize=(8, 4))
    plt.scatter(predictions.flatten(), errors, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Vorhersage")
    plt.ylabel("Fehler")
    plt.title(f"Residual Plot ({name})")
    plt.savefig(f"activation_plots/residual_{name.lower()}.png")
    plt.close()

# R² Score
# 1.0 = perfekt; 0.9 = sehr gut; 0.8 gut; 0.5 = mittel; 0.0 = kaum besser als Raten

# Zusammenfassung am Ende ausgeben
print("\n" + "="*30)
print("ERGEBNIS-ÜBERSICHT")
print("="*30)
for name, res in results.items():
    print(f"{name:12} -> MAE: {res['mae']:.3f} | R² Score: {res['r2']:.3f}")