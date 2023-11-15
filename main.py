# Author: Hari Balaji Baskar
# Description: This script defines a Flask web application for similarity search and text summarization.

import os
from flask import Flask, render_template, request, jsonify
from similarity_search import extract_text_from_pdf, search, preprocess_text
from summarize_text import summarize_text

# Initialize the Flask application
app = Flask(__name__, template_folder='templates')


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Route for the Similarity search page
@app.route('/search', methods=['GET', 'POST'])
def search_api():
    if request.method == 'POST':
        search_query = request.form.get('query')
        text = request.form.get('text')
        pdf_file = request.files.get('pdf')

        if text:
            # Preprocess the text and tokenize into sentences
            sentences = preprocess_text(text)

            # Perform search based on the query
            results = search(search_query, sentences)

            # Return the top search results
            top_results = results[:100]  # Change the number as per your preference
            search_results = []
            for index, score in top_results:
                if score >= 0.0:
                    search_results.append((sentences[index], score))
            return jsonify({'results': search_results, 'Confidence Score': score})

        elif pdf_file:
            # Extract text from PDF file
            text = extract_text_from_pdf(pdf_file)
            if text:
                # Preprocess the text and tokenize into sentences
                sentences = preprocess_text(text)

                # Perform search based on the query
                results = search(search_query, sentences)

                # Return the top search results
                top_results = results[:5]  # Change the number as per your preference
                search_results = []
                for index, score in top_results:
                    if score >= 0.0:
                        search_results.append((sentences[index], score))
                return jsonify({'results': search_results, 'Confidence Score': score})

        return jsonify({'error': 'Invalid input.'})

    else:
        # Handle GET request
        return render_template('similarity_search.html')


# Route for summarizing text
@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']
        summary = summarize_text(text)
        return render_template('summarization.html', text=text, summary=summary)
    return render_template('summarization.html')


# Run the Flask application
if __name__ == '__main__':
    app.run()
