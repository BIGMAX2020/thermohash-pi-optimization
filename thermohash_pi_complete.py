import sqlite3
import numpy as np
from sklearn.linear_model import SGDRegressor

class LightweightPowerOptimizer:
    def __init__(self):
        self.model = SGDRegressor()
        self.X = []  # Features
        self.y = []  # Targets

    def fit(self, X, y):
        self.model.partial_fit(X, y)
        self.X.extend(X)
        self.y.extend(y)

    def predict(self, X):
        return self.model.predict(X)

    def update_model(self, new_data):
        self.fit(*zip(*new_data))

class EfficientDataLogger:
    def __init__(self, db_name='thermo_data.db'):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            temperature REAL,
            power_usage REAL
        )''')
        self.connection.commit()

    def log_data(self, temperature, power_usage):
        self.cursor.execute('''INSERT INTO readings (temperature, power_usage) VALUES (?, ?)''', (temperature, power_usage))
        self.connection.commit()

    def close(self):
        self.connection.close()

class ThermoHash:
    def __init__(self):
        self.optimizer = LightweightPowerOptimizer()
        self.logger = EfficientDataLogger()

    def process_data(self, temperature, power_usage):
        self.logger.log_data(temperature, power_usage)
        self.optimizer.update_model([(temperature, power_usage)])

    def continuous_learning(self, new_data):
        self.optimizer.update_model(new_data)

if __name__ == '__main__':
    # Example usage
    thermo_hash = ThermoHash()
    thermo_hash.process_data(24.5, 150)
    print(thermo_hash.optimizer.predict([[25.0]]))  # Predict power usage for 25.0 degrees

    # Remember to close the database connection
    thermo_hash.logger.close()