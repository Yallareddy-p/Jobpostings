"""
Data analysis module for ML Job Analytics Dashboard.

This module provides functions for processing and analyzing job posting data,
including skill extraction, co-occurrence analysis, and time series analysis.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from datetime import datetime, timedelta
import networkx as nx

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def load_data():
    """
    Load and return sample job posting data.
    
    Returns:
        pandas.DataFrame: DataFrame containing job posting data with columns:
            - job_title: Title of the job posting
            - company: Company name
            - location: Job location
            - job_description: Full job description text
    """
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

def clean_job_titles(df):
    """
    Normalize job titles by converting to lowercase and replacing spaces with underscores.
    
    Args:
        df (pandas.DataFrame): DataFrame containing job posting data
        
    Returns:
        pandas.DataFrame: DataFrame with normalized job titles
    """
    df['job_title'] = df['job_title'].str.lower().str.replace(' ', '_')
    return df

def extract_skills(df):
    """
    Extract skills from job descriptions using NLP.
    
    Args:
        df (pandas.DataFrame): DataFrame containing job posting data
        
    Returns:
        pandas.DataFrame: DataFrame with added 'skills' column containing lists of extracted skills
    """
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

def generate_analytics(df):
    """
    Generate basic analytics from job posting data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing job posting data
        
    Returns:
        tuple: Four pandas.Series containing:
            - title_counts: Count of each job title
            - skill_counts: Count of each skill
            - location_counts: Count of jobs by location
            - company_counts: Count of jobs by company
    """
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

def generate_skill_network(df):
    """
    Generate a skill co-occurrence network from job posting data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing job posting data
        
    Returns:
        tuple: Two dictionaries containing:
            - edge_trace: List of dictionaries for network edges
            - node_trace: Dictionary for network nodes
    """
    # Create a list of all skill pairs
    skill_pairs = []
    for skills in df['skills']:
        for i in range(len(skills)):
            for j in range(i + 1, len(skills)):
                skill_pairs.append((skills[i], skills[j]))
    
    # Count co-occurrences
    co_occurrence = pd.Series(skill_pairs).value_counts()
    
    # Create network graph
    G = nx.Graph()
    
    # Add edges with weights
    for (skill1, skill2), weight in co_occurrence.items():
        G.add_edge(skill1, skill2, weight=weight)
    
    # Calculate node positions using spring layout
    pos = nx.spring_layout(G)
    
    # Prepare data for visualization
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append({
            'x': [x0, x1, None],
            'y': [y0, y1, None],
            'line': {'width': edge[2]['weight']},
            'hoverinfo': 'text',
            'text': f"{edge[0]} - {edge[1]}: {edge[2]['weight']} co-occurrences"
        })
    
    node_trace = {
        'x': [],
        'y': [],
        'text': [],
        'mode': 'markers',
        'hoverinfo': 'text',
        'marker': {
            'size': [],
            'color': []
        }
    }
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node])
        node_trace['marker']['size'] += tuple([G.degree(node) * 5])
        node_trace['marker']['color'] += tuple([G.degree(node)])
    
    return edge_trace, node_trace

def generate_time_series():
    """
    Generate time series data for job posting trends.
    
    Returns:
        pandas.DataFrame: DataFrame containing monthly job posting counts with columns:
            - Month: Date of the month
            - Job Count: Number of job postings for that month
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    # Simulate job posting counts (replace with actual data)
    job_counts = np.random.randint(10, 100, size=len(date_range))
    time_series_df = pd.DataFrame({'Month': date_range, 'Job Count': job_counts})
    return time_series_df 