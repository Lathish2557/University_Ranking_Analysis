import streamlit as st
import pandas as pd
import sqlite3
import openai
import plotly.express as px

# Define constants
csv_file_path = "data/timesData.csv"  # Replace with your dataset path
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Replace with your OpenAI API key

# Set page configuration
st.set_page_config(
    page_title="University Ranking Chatbot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add a sidebar for navigation
with st.sidebar:
    st.title("🎓 University Ranking Chatbot")
    st.markdown("This application helps to analyze university rankings and identify top-performing institutions from 2011 to 2016.")
    st.image("data/logo.jpg", use_container_width=True)
    st.write("Developed by Lathish (https://github.com/Lathish2557)")
# Apply custom CSS for styling
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 15px;
        font-size: 16px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stTextInput > div > div > input {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px;
    }
    .title-text {
        color: #2E86C1;
        font-weight: bold;
        font-size: 28px;
        margin-bottom: 10px;
    }
    .stMarkdown {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="title-text">University Ranking Analyser </div>', unsafe_allow_html=True)
st.markdown(
    " **Ask a question** about university rankings, scores, or demographics, and this chatbot will give the relavant data and the analysis for it. "
)

@st.cache_data
def load_data(csv_path):
    try:
        data = pd.read_csv(csv_path)
        return data
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

# Load data
df = load_data(csv_file_path)

# Dynamic mapping of user-friendly values to dataset column values
value_mapping = {
    "country": {
        "united states": "United States of America",
        "uk": "United Kingdom",
        "china": "China",
        "india": "India"
    },
    "year": {
        "2011": "2011",
        "2012": "2012",
        "2013": "2013"
    }
}

# Synonyms dictionary for understanding user input
synonyms = {
    "rank": ["world rank", "ranking", "position", "place"],
    "university": ["university", "institution", "college"],
    "country": ["country", "location", "nation"],
    "teaching": ["teaching", "education quality", "academic quality"],
    "research": ["research", "research score", "research quality"],
    "citations": ["citations", "citation impact", "citation score"],
    "income": ["income", "funding", "revenue"],
    "total score": ["total score", "overall score", "aggregate score"],
    "students": ["students", "number of students", "enrollment"],
    "staff ratio": ["student-staff ratio", "staff ratio"],
    "international": ["international students", "diversity", "international percentage"],
    "gender ratio": ["female to male ratio", "gender ratio", "gender balance"],
    "year": ["year", "academic year"]
}

# Parameter definitions for dynamic extraction
parameter_definitions = [
    {"name": "country", "patterns": ["United States", "UK", "China", "India"], "mapping": value_mapping.get("country", {})},
    {"name": "year", "patterns": ["2011", "2012", "2013"], "mapping": value_mapping.get("year", {})},
    {"name": "metrics", "patterns": ["teaching", "research", "citations", "income", "total score"], "mapping": None},
    {"name": "rank", "patterns": ["top 10", "below 50", "highest", "lowest"], "mapping": None},
    {"name": "students", "patterns": ["student-staff ratio", "international students", "gender ratio"], "mapping": None}
]

# Few-shot examples for OpenAI
few_shot_examples = [
    {
        "question": "Show me the top 10 universities in 2011.",
        "sql": "SELECT * FROM df WHERE year = 2011 ORDER BY world_rank LIMIT 10;"
    },
    {
        "question": "List universities in the United States with a research score above 90.",
        "sql": "SELECT * FROM df WHERE country = 'United States of America' AND research > 90;"
    },
    {
        "question": "Find universities with a student-staff ratio below 10 in 2012.",
        "sql": "SELECT * FROM df WHERE year = 2012 AND student_staff_ratio < 10;"
    },
    {
        "question": "Which universities have the highest income in 2013?",
        "sql": "SELECT * FROM df WHERE year = 2013 ORDER BY income DESC LIMIT 1;"
    },
    {
        "question": "Show universities in China with more than 20% international students.",
        "sql": "SELECT * FROM df WHERE country = 'China' AND international_students > '20%';"
    }
]

# Function to preprocess user input
def preprocess_user_input(user_input):
    for canonical_term, synonyms_list in synonyms.items():
        for synonym in synonyms_list:
            if synonym.lower() in user_input.lower():
                user_input = user_input.replace(synonym, canonical_term)
    for column, mapping in value_mapping.items():
        for friendly_value, actual_value in mapping.items():
            if friendly_value.lower() in user_input.lower():
                user_input = user_input.replace(friendly_value, actual_value)
    return user_input

# Function to extract dynamic parameters
def extract_parameters(user_input):
    """Extract parameters dynamically based on defined parameter patterns."""
    params = {}
    for param_def in parameter_definitions:
        for pattern in param_def["patterns"]:
            if pattern.lower() in user_input.lower():
                if param_def["mapping"] and pattern in param_def["mapping"]:
                    params[param_def["name"]] = param_def["mapping"][pattern]
                else:
                    params[param_def["name"]] = pattern.capitalize()
    return params

# Function to generate SQL
def get_sql_from_model(user_input):
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        few_shot_prompt = "\n".join(
            [f"User Question: {ex['question']}\nSQL Query: {ex['sql']}" for ex in few_shot_examples]
        )
        few_shot_prompt += f"\nUser Question: {user_input}\nSQL Query:"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert SQL generator for structured university datasets."},
                {"role": "user", "content": few_shot_prompt},
            ],
            max_tokens=150,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return None

# Function to fetch data using SQLite
def fetch_data_from_csv(query, df):
    try:
        conn = sqlite3.connect(":memory:")
        df.to_sql("df", conn, index=False, if_exists="replace")
        result_df = pd.read_sql_query(query, conn)
        conn.close()
        return result_df
    except Exception as e:
        st.error(f"Error executing SQL: {e}")
        return None

def generate_analysis(df):
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        prompt = (
            "Here is a dataset:\n\n" + df.head(10).to_string(index=False) +
            "\n\nGenerate a concise analysis of key insights from this data."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a data analyst expert. Just provide an overall analysis for the data."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating analysis: {e}")
        return None

# User Input Section
st.markdown("### Ask a Question")
user_input = st.text_input(
    "Example: 'Show me the top university in USA in 2011.'",
    placeholder="Type your question here...",
)

# Button to generate SQL and fetch data
if st.button("Answer 💾"):
    if user_input:
        # Preprocess input and extract parameters
        preprocessed_input = preprocess_user_input(user_input)
        params = extract_parameters(preprocessed_input)

        # Generate SQL query
        sql_query = get_sql_from_model(preprocessed_input)
        if sql_query:
            #st.markdown("#### 📝 Generated SQL Query")
            #st.code(sql_query, language="sql")

            result_df = fetch_data_from_csv(sql_query, df)
            if result_df is not None and not result_df.empty:
                st.markdown("### 📊 Query Results")
                st.dataframe(result_df)

                # Generate analysis
                analysis = generate_analysis(result_df)
                if analysis:
                    st.markdown("### 🧠 Data Analysis")
                    st.markdown(analysis)
            else:
                st.warning("No results found for your query.")
