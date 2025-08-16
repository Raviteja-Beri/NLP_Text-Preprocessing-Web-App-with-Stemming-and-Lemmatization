import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from flask import Flask, request, render_template

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

def process_text(content, method):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    wordpunct_tokenizer = WordPunctTokenizer()
    
    if method == 'sentence':
        return {'result': nltk.sent_tokenize(content), 'type': 'Sentences'}
    elif method == 'word':
        return {'result': nltk.word_tokenize(content), 'type': 'Words'}
    elif method == 'wordpunct':
        return {'result': wordpunct_tokenizer.tokenize(content), 'type': 'WordPunct Words'}
    elif method == 'stem':
        sentences = nltk.sent_tokenize(content)
        processed = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
            processed.append(' '.join(words))
        return {'result': processed, 'type': 'Stemmed Sentences'}
    elif method == 'lemmatize':
        sentences = nltk.sent_tokenize(content)
        processed = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
            processed.append(' '.join(words))
        return {'result': processed, 'type': 'Lemmatized Sentences'}
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    input_text = ""
    method = ""
    if request.method == 'POST':
        input_text = request.form['text']
        method = request.form['method']
        if input_text and method:
            result = process_text(input_text, method)
    return render_template('index.html', result=result, input_text=input_text, method=method)

if __name__ == '__main__':
    app.run(debug=True)