# src/modules/nlp.py

from typing import Dict, Any, List
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
import speech_recognition as sr
import pyttsx3

class NLPModule:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')

        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        
        # Load the AllenNLP coreference resolution model
        self.coref_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
        self.spacy_tokenizer = SpacyTokenizer()

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()

        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()

    def listen(self) -> str:
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, I didn't catch that.")
                return ""
            except sr.RequestError:
                print("Sorry, there was an error with the speech recognition service.")
                return ""

    def speak(self, text: str) -> None:
        print(f"AI: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def parse(self, input_data: str) -> Dict[str, Any]:
        tokens = word_tokenize(input_data)
        pos_tags = pos_tag(tokens)
        named_entities = self.spacy_nlp(input_data)

        return {
            'tokens': tokens,
            'pos_tags': pos_tags,
            'named_entities': [(ent.text, ent.label_) for ent in named_entities.ents],
            'intent': self.detect_intent(input_data),
            'sentiment': self.analyze_sentiment(input_data)
        }

    def detect_intent(self, input_data: str) -> str:
        inputs = self.tokenizer(input_data, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        intent_labels = ["query", "task", "conversation"]  # Example labels
        return intent_labels[predicted_class_id]

    def analyze_sentiment(self, input_data: str) -> str:
        result = self.sentiment_analyzer(input_data)[0]
        return f"{result['label']} with a score of {result['score']}"

    def generate_response(self, results: List[Dict[str, Any]]) -> str:
        response_parts = []
        for result in results:
            if result['status'] == 'completed':
                response_parts.append(f"Task '{result['task']['description']}' completed. Result: {result['result']}")
            else:
                response_parts.append(f"Task '{result['task']['description']}' failed. Error: {result['result']}")
        return ' '.join(response_parts)

    def summarize_text(self, input_data: str) -> str:
        summarizer = pipeline("summarization")
        summary = summarizer(input_data, max_length=50, min_length=25, do_sample=False)
        return summary[0]['summary_text']

    def translate_text(self, input_data: str, target_language: str = "fr") -> str:
        translator = pipeline("translation_en_to_fr")
        translation = translator(input_data)
        return translation[0]['translation_text']

    def extract_keywords(self, input_data: str) -> List[str]:
        doc = self.spacy_nlp(input_data)
        keywords = [chunk.text for chunk in doc.noun_chunks]
        return keywords

    def named_entity_recognition(self, input_data: str) -> List[Dict[str, str]]:
        doc = self.spacy_nlp(input_data)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        return entities

    def part_of_speech_tagging(self, input_data: str) -> List[Dict[str, str]]:
        tokens = word_tokenize(input_data)
        pos_tags = pos_tag(tokens)
        return [{"word": word, "pos": pos} for word, pos in pos_tags]

    def dependency_parsing(self, input_data: str) -> List[Dict[str, str]]:
        doc = self.spacy_nlp(input_data)
        dependencies = [{"text": token.text, "dep": token.dep_, "head": token.head.text} for token in doc]
        return dependencies

    def coreference_resolution(self, input_data: str) -> List[Dict[str, Any]]:
        result = self.coref_predictor.predict(document=input_data)
        document = result['document']
        resolved_corefs = []
        for cluster in result['clusters']:
            main_mention = self.get_span_text(document, cluster[0])
            for mention in cluster:
                mention_text = self.get_span_text(document, mention)
                if mention_text != main_mention:
                    resolved_corefs.append({
                        "mention": mention_text,
                        "resolved": main_mention,
                        "start": mention[0],
                        "end": mention[1]
                    })
        return resolved_corefs

    def get_span_text(self, document: List[str], span: List[int]) -> str:
        return " ".join(document[span[0]:span[1]+1])

    def resolve_text(self, input_data: str) -> str:
        doc = self.spacy_nlp(input_data)
        tokens = [token.text for token in doc]
        coref_results = self.coreference_resolution(input_data)
        coref_results.sort(key=lambda x: x['start'], reverse=True)
        for coref in coref_results:
            tokens[coref['start']:coref['end']+1] = [coref['resolved']]
        resolved_text = ' '.join(tokens)
        return resolved_text

    def question_answering(self, question: str, context: str) -> str:
        qa_pipeline = pipeline("question-answering")
        result = qa_pipeline(question=question, context=context)
        return result['answer']

    def text_classification(self, input_data: str) -> str:
        classifier = pipeline("zero-shot-classification")
        labels = ["positive", "negative", "neutral"]
        result = classifier(input_data, candidate_labels=labels)
        return result['labels'][0]

    def __str__(self) -> str:
        return f"NLPModule with configuration: {self.config}"

# Example usage
if __name__ == "__main__":
    config = {"language": "en"}
    nlp_module = NLPModule(config)

    while True:
        # Listen for voice input
        user_input = nlp_module.listen()
        if not user_input:
            continue

        # Process the input
        intent = nlp_module.detect_intent(user_input)
        sentiment = nlp_module.analyze_sentiment(user_input)

        # Generate a response based on intent and sentiment
        if intent == "query":
            response = f"I detected a question. The sentiment is {sentiment}."
        elif intent == "task":
            response = f"I'll get right on that task. By the way, you sound {sentiment}."
        else:
            response = f"Let's continue our {sentiment} conversation."

        # Speak the response
        nlp_module.speak(response)

        # Exit condition
        if "goodbye" in user_input.lower():
            nlp_module.speak("Goodbye!")
            break