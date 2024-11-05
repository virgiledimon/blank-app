from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

class InterpretabilityAgent:
    def __init__(self):
        self.scaler_minmax = MinMaxScaler()
        self.scaler_standard = StandardScaler()
        self.model_decision = DecisionTreeClassifier()  # Modèle pour les décisions (arbre de décision)

    # Méthode pour normaliser les données de décisions
    def normalize_decision_data(self, X_train, X_test):
        X_train_normalized = self.scaler_standard.fit_transform(X_train)
        X_test_normalized = self.scaler_standard.transform(X_test)
        return X_train_normalized, X_test_normalized

    # Méthode pour entraîner le modèle de décisions
    def train_decision_model(self, X_train, y_train):
        self.model_decision.fit(X_train, y_train)
        print("Modèle de décisions entraîné avec succès.")

    # Méthode pour faire des prédictions sur les décisions
    def predict_decision(self, X_test):
        return self.model_decision.predict(X_test)

    # Méthode pour évaluer les prédictions
    def evaluate_model(self, X_test, y_test):
        y_pred = self.predict_decision(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Précision du modèle : {accuracy:.2f}")
        return accuracy

    # Méthode pour afficher l'arbre de décision
    def plot_decision_tree(self):
        if hasattr(self.model_decision, 'tree_'):
            plt.figure(figsize=(20, 10))
            plot_tree(self.model_decision, filled=True,
                      feature_names=['lambda_threshold', 'l_value'],
                      class_names=['Reject', 'Accept'])
            plt.show()
        else:
            print("Le modèle n'est pas encore entraîné. Entraînez d'abord le modèle avant de l'afficher.")


# Exemple d'utilisation
if __name__ == "__main__":
    # Supposons que db_agent est déjà initialisé et connecté à la base de données
    db_agent = PostgreSQLAgent(db_name='', user='postgres', password='postgres')

    # Récupérer les données de décisions via l'agent PostgreSQL
    decision_data = db_agent.fetch_decisions_data()

    # Sauvegarder le tableau NumPy en fichier CSV
    #np.savetxt('fc_impersonation_detection_dataset.csv', decision_data, delimiter=',', header="lambda_threshold,L_value,decision,v_channel_vector,r_channel_record,reward,legit_transmitter", comments='')

    # Séparer les caractéristiques (lambda_threshold, l_value) et la cible (decision_made)
    X = decision_data[:, [0, 1]]  # Caractéristiques : lambda_threshold, l_value
    y = decision_data[:, 2]  # Cible : decision_made

    # Séparation des données en ensembles d'entraînement et de test (50/50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Création de l'agent d'interprétabilité
    interpret_agent = InterpretabilityAgent()

    # Normalisation des données
    X_train_normalized, X_test_normalized = interpret_agent.normalize_decision_data(X_train, X_test)

    # Entraîner le modèle de décision sur les données d'entraînement
    interpret_agent.train_decision_model(X_train_normalized, y_train)

    # Évaluer le modèle sur les données de test
    interpret_agent.evaluate_model(X_test_normalized, y_test)

    # Afficher l'arbre de décision entraîné
    interpret_agent.plot_decision_tree()