import nltk
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, reuters
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import spacy
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('reuters', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load SpaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    # If model not found, download it
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = nlp
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return processed_tokens
    
    def extract_entities(self, text):
        """Extract named entities using SpaCy"""
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities
    
    def extract_with_regex(self, text, pattern, label):
        """Extract information using regex patterns"""
        matches = re.findall(pattern, text)
        return {label: matches}
    
    def extract_key_phrases(self, text):
        """Extract key phrases using POS patterns"""
        doc = self.nlp(text)
        
        # Get noun chunks (noun phrases)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Get verb phrases (simplified approach)
        verb_phrases = []
        for token in doc:
            if token.pos_ == "VERB":
                phrase = token.text
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        phrase += " " + child.text
                verb_phrases.append(phrase)
        
        return {
            "noun_phrases": noun_phrases,
            "verb_phrases": verb_phrases
        }
    
    def analyze_sentiment(self, text):
        """Basic rule-based sentiment analysis"""
        doc = self.nlp(text)
        
        # Simple approach using polarity scores of adjectives
        positive_adj = ["good", "great", "excellent", "positive", "wonderful", "fantastic", "amazing"]
        negative_adj = ["bad", "poor", "negative", "terrible", "awful", "horrible", "disappointing"]
        
        sentiment_score = 0
        for token in doc:
            if token.lemma_.lower() in positive_adj:
                sentiment_score += 1
            elif token.lemma_.lower() in negative_adj:
                sentiment_score -= 1
        
        if sentiment_score > 0:
            return "positive"
        elif sentiment_score < 0:
            return "negative"
        else:
            return "neutral"
    
    def extract_features(self, text):
        """Extract all features from text"""
        features = {}
        
        # Named entities
        features["entities"] = self.extract_entities(text)
        
        # Key phrases
        key_phrases = self.extract_key_phrases(text)
        features["noun_phrases"] = key_phrases["noun_phrases"]
        features["verb_phrases"] = key_phrases["verb_phrases"]
        
        # Sentiment
        features["sentiment"] = self.analyze_sentiment(text)
        
        # Dates (basic regex)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        dates = self.extract_with_regex(text, date_pattern, "dates")
        features.update(dates)
        
        return features


class TextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def extractive_summarize_tfidf(self, text, num_sentences=3):
        """Generate extractive summary using TF-IDF"""
        sentences = sent_tokenize(text)
        
        # If too few sentences, return original text
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF matrix
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        
        # Get sentence similarity matrix
        sentence_similarity = cosine_similarity(tfidf_matrix)
        
        # Create a graph and get scores
        nx_graph = nx.from_numpy_array(sentence_similarity)
        print(nx_graph)
        scores = nx.pagerank(nx_graph)
        
        # Rank sentences by score and select top ones
        ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
        
        # Get the indices of the top sentences in their original order
        top_indices = sorted([idx for _, idx, _ in ranked_sentences[:num_sentences]])
        
        # Join the top sentences to form the summary
        summary = ' '.join([sentences[i] for i in top_indices])
        
        return summary
    
    def extractive_summarize_centroid(self, text, num_sentences=3):
        """Generate extractive summary using centroid method"""
        sentences = sent_tokenize(text)
        
        # If too few sentences, return original text
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF matrix
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        
        # Calculate centroid vector (average of all sentence vectors)
        centroid = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Calculate similarity of each sentence to the centroid
        similarities = []
        for i in range(len(sentences)):
            sentence_vector = tfidf_matrix[i].toarray()[0]
            similarity = cosine_similarity([centroid], [sentence_vector])[0][0]
            similarities.append((i, similarity))
        
        # Rank sentences by similarity to centroid
        ranked_sentences = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Get the indices of the top sentences in their original order
        top_indices = sorted([idx for idx, _ in ranked_sentences[:num_sentences]])
        
        # Join the top sentences to form the summary
        summary = ' '.join([sentences[i] for i in top_indices])
        
        return summary


class DocumentExplorer:
    def __init__(self):
        self.processor = TextProcessor()
    
    def load_reuters_corpus(self, max_documents=100):
        """Load documents from the Reuters corpus"""
        file_ids = reuters.fileids()[:max_documents]
        documents = []
        
        for file_id in file_ids:
            text = reuters.raw(file_id)
            category = reuters.categories(file_id)
            documents.append({
                'id': file_id,
                'text': text,
                'category': category
            })
        
        return documents
    
    def explore_corpus(self, documents):
        """Perform exploratory analysis on the corpus"""
        # Document lengths
        doc_lengths = [len(doc['text'].split()) for doc in documents]
        
        # Word frequencies
        all_words = []
        for doc in documents:
            processed_tokens = self.processor.preprocess_text(doc['text'])
            all_words.extend(processed_tokens)
        
        word_freq = Counter(all_words)
        most_common = word_freq.most_common(20)
        
        # Category distribution (for Reuters)
        categories = []
        for doc in documents:
            categories.extend(doc['category'])
        category_freq = Counter(categories)
        
        results = {
            'doc_count': len(documents),
            'avg_doc_length': np.mean(doc_lengths),
            'median_doc_length': np.median(doc_lengths),
            'min_doc_length': min(doc_lengths),
            'max_doc_length': max(doc_lengths),
            'most_common_words': most_common,
            'category_distribution': category_freq.most_common(10)
        }
        
        return results

    def visualize_corpus(self, results):
        """Visualize corpus statistics"""
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Word frequency plot
        words, counts = zip(*results['most_common_words'])
        sns.barplot(x=list(counts), y=list(words), ax=ax1)
        ax1.set_title('Top 20 Words by Frequency')
        ax1.set_xlabel('Count')
        
        # Category distribution plot
        categories, cat_counts = zip(*results['category_distribution'])
        sns.barplot(x=list(cat_counts), y=list(categories), ax=ax2)
        ax2.set_title('Top 10 Categories')
        ax2.set_xlabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        # Print additional stats
        print(f"Corpus Statistics:")
        print(f"- Number of documents: {results['doc_count']}")
        print(f"- Average document length: {results['avg_doc_length']:.2f} words")
        print(f"- Median document length: {results['median_doc_length']} words")
        print(f"- Document length range: {results['min_doc_length']} to {results['max_doc_length']} words")


class InformationAgent:
    def __init__(self):
        self.processor = TextProcessor()
        self.summarizer = TextSummarizer()
        self.document_store = {}
        self.document_index = {}
        self.processed_features = {}
    
    def add_documents(self, documents):
        """Add documents to the agent's knowledge base"""
        for doc in tqdm(documents, desc="Processing documents"):
            doc_id = doc['id'] if 'id' in doc else len(self.document_store)
            
            # Store original document
            self.document_store[doc_id] = doc
            
            # Process and extract features
            features = self.processor.extract_features(doc['text'])
            self.processed_features[doc_id] = features
            
            # Create simple inverted index for entities and key phrases
            self._update_index(doc_id, features)
            
            # Generate summary
            summary = self.summarizer.extractive_summarize_tfidf(doc['text'])
            self.document_store[doc_id]['summary'] = summary
    
    def _update_index(self, doc_id, features):
        """Update the inverted index with extracted features"""
        # Index entities
        for entity_type, entities in features['entities'].items():
            for entity in entities:
                if entity not in self.document_index:
                    self.document_index[entity] = set()
                self.document_index[entity].add(doc_id)
        
        # Index noun phrases
        for phrase in features['noun_phrases']:
            if phrase not in self.document_index:
                self.document_index[phrase] = set()
            self.document_index[phrase].add(doc_id)
    
    def search(self, query, top_k=5):
        """Search for relevant documents based on query"""
        # Process query to extract key terms
        query_features = self.processor.extract_features(query)
        
        # Get all entities and phrases from query
        search_terms = []
        for entity_type, entities in query_features['entities'].items():
            search_terms.extend(entities)
        search_terms.extend(query_features['noun_phrases'])
        
        # Find matching documents
        matching_docs = {}
        for term in search_terms:
            if term in self.document_index:
                for doc_id in self.document_index[term]:
                    if doc_id not in matching_docs:
                        matching_docs[doc_id] = 0
                    matching_docs[doc_id] += 1
        
        # Rank documents by match count
        ranked_docs = sorted(matching_docs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [(doc_id, self.document_store[doc_id]) for doc_id, score in ranked_docs]
    
    def answer_question(self, question):
        """Answer a question using the document knowledge base"""
        # Find relevant documents
        relevant_docs = self.search(question)
        
        if not relevant_docs:
            return "I couldn't find any relevant information to answer your question."
        
        # Extract information directly relevant to the question
        question_features = self.processor.extract_features(question)
        answer_components = []
        
        for doc_id, doc in relevant_docs:
            # Add document summary
            answer_components.append(f"From document {doc_id}: {doc['summary']}")
            
            # Add specific entities if they match the question
            for entity_type, entities in question_features['entities'].items():
                if entity_type in self.processed_features[doc_id]['entities']:
                    matching_entities = set(entities) & set(self.processed_features[doc_id]['entities'][entity_type])
                    if matching_entities:
                        answer_components.append(f"Found {entity_type}: {', '.join(matching_entities)}")
        
        # Combine components into an answer
        answer = "\n\n".join(answer_components)
        return answer


# Main function to demonstrate the system
def main():
    print("Initializing Text Processing System...")
    explorer = DocumentExplorer()
    agent = InformationAgent()
    
    # 1. Data Preparation & Exploration
    print("\nLoading Reuters corpus...")
    documents = explorer.load_reuters_corpus(max_documents=100)
    
    print(f"Loaded {len(documents)} documents from Reuters corpus")
    print("Sample document:")
    print("-" * 50)
    print(documents[0]['text'][:500] + "...")
    print("-" * 50)
    
    # Exploratory analysis
    print("\nPerforming exploratory analysis...")
    results = explorer.explore_corpus(documents)
    explorer.visualize_corpus(results)
    
    # 2. Information Extraction & Summarization
    print("\nProcessing documents and building agent knowledge base...")
    agent.add_documents(documents)
    
    # 3. Demo extraction on a sample document
    sample_doc = documents[0]['text']
    
    print("\nSample information extraction:")
    print("-" * 50)
    processor = TextProcessor()
    features = processor.extract_features(sample_doc)
    
    print("Named Entities:")
    for entity_type, entities in features['entities'].items():
        if entities:
            print(f"- {entity_type}: {', '.join(entities[:5])}")
    
    print("\nSentiment:", features['sentiment'])
    
    if 'dates' in features and features['dates']:
        print("\nDates:", ', '.join(features['dates'][:5]))
    
    print("\nNoun Phrases (sample):", ', '.join(features['noun_phrases'][:5]))
    
    # 4. Demo summarization
    print("\nSample summarization:")
    print("-" * 50)
    summarizer = TextSummarizer()
    summary = summarizer.extractive_summarize_tfidf(sample_doc)
    print(summary)
    
    # 5. Demo agent answering a question
    print("\nAsking the agent a question:")
    print("-" * 50)
    sample_question = "What information is there about oil prices?"
    print(f"Question: {sample_question}")
    print("\nAnswer:")
    answer = agent.answer_question(sample_question)
    print(answer)
    
    print("\nText Processing System demonstration complete!")


if __name__ == "__main__":
    main()