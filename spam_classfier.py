import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# This class is responsible for converting text data into numerical format using CountVectorizer.
class Preprocessor:
    def __init__(self):
        self.vectorizer = CountVectorizer()
    
    def fit_transform(self, X_train):
        #fit on training data and tranform it into features vectors 
        return self.vectorizer.fit_transform(X_train)
    
    def transform(self, X_test):
        #transforms test data into feature vectors using the fitted vectorizer
        return self.vectorizer.transform(X_test)

#This class defines the overall classifier logic using Naive Bayes 
class SpamClassifier:
    def __init__(self):
        self.model = MultinomialNB()  # Naive Bayes model for text
        self.preprocessor = Preprocessor()  # Use our custom preprocessor
    
    def load_data(self, url):
        # Load and label SMS spam dataset from URL
        df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0/1
        return df

    def train(self, messages, labels):
        # Split into training and test data
        X_train, X_test, y_train, y_test = train_test_split(
            messages, labels, test_size=0.2, random_state=42
        )
        # Preprocess text
        X_train_vec = self.preprocessor.fit_transform(X_train)
        X_test_vec = self.preprocessor.transform(X_test)

        # Train model
        self.model.fit(X_train_vec, y_train)

        # Evaluate performance
        y_pred = self.model.predict(X_test_vec)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def predict(self, message):
        # Predict if a single message is spam or not
        vec = self.preprocessor.transform([message])
        return "Spam" if self.model.predict(vec)[0] else "Not Spam"
    
# This block runs when the script is executed directly
if __name__ == "__main__":
    url = "sms.tsv"
    classifier = SpamClassifier()  # Create a classifier object
    df = classifier.load_data(url)  # Load and prepare data
    classifier.train(df['message'], df['label'])  # Train and evaluate model

    # Try predicting a custom message
    print(classifier.predict("You have won a $1000 Walmart gift card! Click now!"))