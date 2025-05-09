# ML Job Analytics Dashboard

An intelligent analytics dashboard for Machine Learning job postings in the U.S., providing insights into job trends, required skills, and market dynamics.

## Features

- **Interactive Dashboard**: Visualize job market trends and patterns
- **Skill Analysis**: 
  - Skill distribution across job postings
  - Skill co-occurrence network visualization
  - Required skill sets by job title
- **Geographic Analysis**: Job distribution across different locations
- **Company Insights**: Hiring patterns and trends by company
- **Time Series Analysis**: Job posting trends over the last 12 months
- **Interactive Filters**: Filter data by job title, location, company, and date range
- **Data Export**: Download filtered data for further analysis

## Tech Stack

- **Backend**: Python, Flask
- **Data Processing**: Pandas, NumPy, spaCy
- **Visualization**: Dash, Plotly
- **Machine Learning**: scikit-learn
- **Deployment**: PythonAnywhere

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Yallareddy-p/Jobpostings.git
cd Jobpostings
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Start the development server:
```bash
python dashboard.py
```

2. Open your browser and navigate to:
```
http://localhost:8050
```

## Project Structure

```
Jobpostings/
├── app.py                 # Flask application
├── dashboard.py           # Dash dashboard implementation
├── data_analysis.py       # Data processing and analysis functions
├── requirements.txt       # Project dependencies
├── Procfile              # Heroku deployment configuration
└── README.md             # Project documentation
```

## Data Analysis Features

### Job Title Analysis
- Normalization of job titles
- Distribution analysis
- Title clustering

### Skill Analysis
- Skill extraction from job descriptions
- Co-occurrence network analysis
- Skill demand trends

### Geographic Analysis
- Location-based job distribution
- Regional skill demand patterns

### Time Series Analysis
- Monthly job posting trends
- Seasonal patterns
- Growth rate analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data sources and APIs used
- Contributing developers
- Open-source libraries and tools

## Contact

For questions and support, please open an issue in the GitHub repository.