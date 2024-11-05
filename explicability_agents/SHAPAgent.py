import shap
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SHAPAgent:
    def __init__(self, model, db_agent):
        self.model = model
        self.db_agent = db_agent
        self.scaler = StandardScaler()
        self.X_test = None  # Initialisation des données de décision normalisées

    # Méthode pour récupérer et normaliser les données de décisions depuis la base
    def get_normalized_decision_data(self):
        # Récupérer les données via la méthode existante
        decision_data = self.db_agent.fetch_decisions_data_min()

        if decision_data is not None and len(decision_data) > 0:
            print("Données de décisions récupérées avec succès.")

            # Séparer les caractéristiques (lambda_threshold, l_value)
            X_decision = decision_data[:, [0, 1]]

            # Normaliser les données
            X_decision_normalized = self.scaler.fit_transform(X_decision)
            self.X_test = X_decision_normalized  # Stocker les données normalisées dans la classe
        else:
            print("Erreur: Aucune donnée de décision disponible.")

    # Méthode pour expliquer une instance spécifique avec SHAP
    def explain_instance(self):
        # Sélection de la première instance de test
        X_instance = self.X_test.iloc[0]
        print(X_instance)
        # Créer l'explainer SHAP basé sur le modèle (assurez-vous que c'est un modèle d'arbre)
        explainer = shap.TreeExplainer(self.model)
        # Calcul des valeurs SHAP pour l'instance
        shap_values = explainer.shap_values(X_instance)
        print(shap_values)
        print(explainer.expected_value)
        print("\n")
        print(f"Base value: {explainer.expected_value[1]}")
        # Initialisation de JavaScript pour les graphiques interactifs
        shap.force_plot(explainer.expected_value[1], shap_values[..., 1], feature_names=self.X_test.columns, matplotlib=True, show=True, figsize=(9,4))
        plt.show()


    # Méthode pour expliquer le modèle entier avec SHAP
    def explain_model(self):

        if self.X_test is None:
          print("Les données ne sont pas encore disponibles. Veuillez appeler 'get_normalized_decision_data' avant.")
          return

        if not isinstance(self.X_test, pd.DataFrame):
          feature_names = ['lambda_threshold', 'L_value']
          self.X_test = pd.DataFrame(self.X_test, columns=feature_names)

        # Créer l'explainer SHAP basé sur le modèle
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test, approximate=True)

        # Affichage des valeurs SHAP sous forme de tableau
        self.display_shap_values_table(shap_values, self.X_test)

        # Afficher le summary plot et le bar plot
        accept_shap_values = shap_values[..., 1]
        shap.summary_plot(accept_shap_values, self.X_test)
        print("\n")
        shap.summary_plot(accept_shap_values, self.X_test, plot_type="bar")
        print("\n")
        # Afficher les contributions pour chaque classe individuellement
        shap.dependence_plot("lambda_threshold", accept_shap_values, self.X_test)
        print("\n")
        shap.dependence_plot("L_value", accept_shap_values, self.X_test)
        print("\n")

    # Méthode pour afficher les valeurs SHAP dans un tableau
    def display_shap_values_table(self, shap_values, X_test):
        # Sélectionner les valeurs SHAP pour la classe d'acceptation
        accept_shap_values = shap_values[...,1]  # Utilisation de la classe 'Accept'
        # Créer un DataFrame pour afficher les valeurs SHAP
        shap_df = pd.DataFrame(accept_shap_values, columns=X_test.columns)
        print("\nTableau des valeurs SHAP pour les décisions 'Accept':")
        print(shap_df.head())  # Afficher les premières lignes du tableau


# Exemple d'utilisation
# if __name__ == "__main__":
#     # Supposons que db_agent est déjà initialisé et connecté à la base de données
#     db_agent = PostgreSQLAgent(db_name='', user='postgres', password='postgres')

#     # Créer l'agent SHAP en utilisant le modèle déjà entraîné
#     shap_agent = SHAPAgent(model=interpret_agent.model_decision, db_agent=db_agent)

#     # Récupérer et normaliser les données de décisions depuis la base
#     shap_agent.get_normalized_decision_data()

#     # Expliquer le modèle entier
#     shap_agent.explain_model()

#     # Instance
#     shap_agent.explain_instance()