import re # Import the regex module
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle

## tokenization
nltk.download('punkt_tab')

##stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
### exclude not
stop_words.remove('not')
stop_words.remove('no')

## vectorization
tf = pickle.load(open('artifacts/tf.pkl', 'rb'))

###lemmetization 
# Download the necessary resources
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng') # Download the missing resource

lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    lemmatized_words = []
    for word in text:
        pos_tag = nltk.pos_tag([word])[0][1] # Get POS tag for each word
        pos = get_wordnet_pos(pos_tag)
        if pos:
          lemmatized_words.append(lemmatizer.lemmatize(word, pos=pos)) # Lemmatize with POS tag
        else:
          lemmatized_words.append(lemmatizer.lemmatize(word)) # Default lemmatization if no valid POS tag
    return lemmatized_words

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return None



### data preprocessing 
def text_preprocessing(text):
  ## lower case
  text = text.lower()
  ## special charcter
  text = re.sub('[^a-zA-z]', ' ', text)
  ## Tokinzation
  text = word_tokenize(text)
  ## stopwords
  text = [word for word in text if word not in stop_words]
  ## Lemmatization using your custom function
  text = lemmatize_text(text) # Call your lemmatize_text function
  text = ' '.join(text)
  text = tf.transform([text])
  return text
