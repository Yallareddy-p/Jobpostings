import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from datetime import datetime, timedelta

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Sample data (replace with actual data loading logic)
def load_data():
    # Simulated job postings data
    data = {
        'job_title': ['ML Engineer', 'Machine Learning Engineer', 'Data Scientist', 'AI Engineer'],
        'company': ['Company A', 'Company B', 'Company C', 'Company D'],
        'location': ['New York', 'San Francisco', 'Boston', 'Seattle'],
        'job_description': [
            'Looking for an ML Engineer with Python and TensorFlow experience.',
            'Machine Learning Engineer needed. Skills: Python, PyTorch, NLP.',
            'Data Scientist role. Required: Python, R, SQL.',
            'AI Engineer position. Must know Python, TensorFlow, and NLP.'
        ]
    }
    return pd.DataFrame(data)

# Clean job titles (normalize)
def clean_job_titles(df):
    df['job_title'] = df['job_title'].str.lower().str.replace(' ', '_')
    return df

# Extract skills from job descriptions
def extract_skills(df):
    skills = []
    for desc in df['job_description']:
        doc = nlp(desc)
        # Simple extraction: words after 'Skills:' or 'Required:'
        skill_text = desc.split('Skills:')[-1].split('Required:')[-1]
        skill_doc = nlp(skill_text)
        skill_list = [token.text for token in skill_doc if token.is_alpha]
        skills.append(skill_list)
    df['skills'] = skills
    return df

# Generate basic analytics
def generate_analytics(df):
    # Count job titles
    title_counts = df['job_title'].value_counts()
    # Count skills
    all_skills = [skill for skills_list in df['skills'] for skill in skills_list]
    skill_counts = pd.Series(all_skills).value_counts()
    # Count jobs by location
    location_counts = df['location'].value_counts()
    # Count jobs by company
    company_counts = df['company'].value_counts()
    return title_counts, skill_counts, location_counts, company_counts

# Simulate time series analysis for job postings over the last 12 months
def generate_time_series():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    # Simulate job posting counts (replace with actual data)
    job_counts = np.random.randint(10, 100, size=len(date_range))
    time_series_df = pd.DataFrame({'Month': date_range, 'Job Count': job_counts})
    return time_series_df 