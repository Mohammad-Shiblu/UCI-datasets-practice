from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

class Models:
    def __init__(self, model_type = 'rf', **kwargs):
        self.model = self.initialize_model(model_type, **kwargs)

    def initialize_model(self, model_type, **kwargs):
        if model_type == 'rf':
            return RandomForestClassifier(**kwargs)
        elif model_type == 'svm':
            return SVC(**kwargs)
        elif model_type == 'logistic_regression':
            return LogisticRegression(**kwargs)
        elif model_type == 'nb':
            return GaussianNB(**kwargs)
        else:
            raise ValueError(f"Model type {model_type} is not supported.")
        
    
    def 