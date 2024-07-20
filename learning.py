import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import json

class Learning:
    def __init__(self):
        self.data: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.pca = PCA(n_components=2)
        self.kmeans = KMeans(n_clusters=5)
        self.classifier = RandomForestClassifier()
        self.model_path = "learning_model.joblib"

    def process(self, new_data: Dict[str, Any]) -> None:
        self.data.append(new_data)
        self._cluster_data()
        self._train_classifier()
        self._save_model()

    def _cluster_data(self) -> None:
        if len(self.data) < 10:  # Wait until we have enough data
            return
        
        texts = [item['text'] for item in self.data if 'text' in item]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        pca_result = self.pca.fit_transform(tfidf_matrix.toarray())
        self.kmeans.fit(pca_result)

        for i, item in enumerate(self.data):
            if 'text' in item:
                item['cluster'] = int(self.kmeans.labels_[i])

    def _train_classifier(self) -> None:
        labeled_data = [item for item in self.data if 'label' in item and 'text' in item]
        if len(labeled_data) < 10:  # Wait until we have enough labeled data
            return

        X = [item['text'] for item in labeled_data]
        y = [item['label'] for item in labeled_data]

        X = self.vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"Classifier Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)

    def _save_model(self) -> None:
        model_data = {
            'vectorizer': self.vectorizer,
            'pca': self.pca,
            'kmeans': self.kmeans,
            'classifier': self.classifier
        }
        joblib.dump(model_data, self.model_path)

    def load_model(self) -> None:
        model_data = joblib.load(self.model_path)
        self.vectorizer = model_data['vectorizer']
        self.pca = model_data['pca']
        self.kmeans = model_data['kmeans']
        self.classifier = model_data['classifier']

    def predict(self, text: str) -> Tuple[int, str]:
        tfidf_vector = self.vectorizer.transform([text])
        pca_result = self.pca.transform(tfidf_vector.toarray())
        cluster = int(self.kmeans.predict(pca_result)[0])
        label = self.classifier.predict(tfidf_vector)[0]
        return cluster, label

    def get_insights(self) -> Dict[str, Any]:
        if len(self.data) < 10:
            return {"message": "Not enough data for insights"}

        cluster_counts = {}
        for item in self.data:
            if 'cluster' in item:
                cluster_counts[item['cluster']] = cluster_counts.get(item['cluster'], 0) + 1

        label_counts = {}
        for item in self.data:
            if 'label' in item:
                label_counts[item['label']] = label_counts.get(item['label'], 0) + 1

        return {
            "total_data_points": len(self.data),
            "cluster_distribution": cluster_counts,
            "label_distribution": label_counts
        }

    def export_data(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(self.data, f)

    def import_data(self, file_path: str) -> None:
        with open(file_path, 'r') as f:
            self.data = json.load(f)

    def __str__(self) -> str:
        return f"Learning module with {len(self.data)} data points"