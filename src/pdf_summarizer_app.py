import PyPDF2
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import streamlit as st
from io import BytesIO


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


# Function to summarize text using BART model
def summarize_text(text, model, tokenizer, max_chunk_length=1024, max_output_length=2000, min_output_length=500):
    inputs = tokenizer.batch_encode_plus(
        [text],
        max_length=max_chunk_length,
        truncation=True,
        return_tensors='pt'
    )
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,  # Reduced the number of beams for more diversity
        max_length=max_output_length,  # Increased max length for the summary
        min_length=min_output_length,  # Added a minimum length to force longer summaries
        length_penalty=1.5,  # Encouraging longer summaries
        temperature=1.0,  # Set to 1.0 for more diversity
        num_return_sequences=1,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Streamlit app layout
st.title(" PDF Summarizer using BART")
st.write("Upload a PDF file to get its summary using Facebook's BART model.")

# File uploader
pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

# Check if a file is uploaded
if pdf_file is not None:
    # Extract text from PDF
    st.write(" Extracting text from the PDF...")

    # Convert the uploaded file to a file-like object
    pdf_file = BytesIO(pdf_file.read())

    # Extract the text from the PDF
    text = extract_text_from_pdf(pdf_file)

    if text:
        # Load BART model and tokenizer
        st.write(" Loading BART model and tokenizer...")
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

        # Summarize the text
        st.write(" Summarizing...")
        summary = summarize_text(text, model, tokenizer)

        # Display the summary
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.error(" No text found in the PDF.")
