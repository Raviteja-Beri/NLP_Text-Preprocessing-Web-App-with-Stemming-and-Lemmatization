import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def process_text(content):
    # Initialize tools
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    wordpunct_tokenizer = WordPunctTokenizer()

    # Tokenization
    sentences = nltk.sent_tokenize(content)
    words = nltk.word_tokenize(content)
    
    # WordPunct Tokenization
    wordpunct_words = wordpunct_tokenizer.tokenize(content)
    
    # Stemming with stopwords removal
    stemmed_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        stemmed_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
        stemmed_sentences.append(' '.join(stemmed_words))
    
    # Lemmatization with stopwords removal
    lemmatized_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
        lemmatized_sentences.append(' '.join(lemmatized_words))
    
    return {
        'sentences': sentences,
        'words': words,
        'wordpunct_words': wordpunct_words,
        'stemmed_sentences': stemmed_sentences,
        'lemmatized_sentences': lemmatized_sentences,
        'last_stemmed_words': stemmed_words,
        'last_lemmatized_words': lemmatized_words
    }

def main():
    # Get user input
    content = input("Enter the text to process: ")
    if not content.strip():
        print("No input provided.")
        return
    
    results = process_text(content)
    
    # Print results
    print("\n=== Sentence Tokenization ===")
    for sentence in results['sentences']:
        print(sentence)
    
    print("\n=== Word Tokenization ===")
    print(results['words'])
    
    print("\n=== WordPunct Tokenization ===")
    print(results['wordpunct_words'])
    
    print("\n=== Stemmed Sentences (Stopwords Removed) ===")
    for sentence in results['stemmed_sentences']:
        print(sentence)
    
    print("\n=== Lemmatized Sentences (Stopwords Removed) ===")
    for sentence in results['lemmatized_sentences']:
        print(sentence)
    
    print("\n=== Last Sentence Stemmed Words ===")
    print(results['last_stemmed_words'])
    
    print("\n=== Last Sentence Lemmatized Words ===")
    print(results['last_lemmatized_words'])

if __name__ == "__main__":
    main()