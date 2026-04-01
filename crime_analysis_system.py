# ============================================
# COMPLETE CRIME ANALYSIS SYSTEM WITH DASHBOARD
# ============================================

# ============================================
# 1. IMPORTS & SETUP
# ============================================
from IPython import get_ipython
from IPython.display import display
from google.colab import drive
import os
import string
import pandas as pd
import numpy as np
import warnings
import nltk
from collections import Counter
from nltk import word_tokenize, pos_tag
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, hamming_loss
import gradio as gr

warnings.filterwarnings("ignore")

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Mount Google Drive
drive.mount('/content/drive')

# ============================================
# 2. CONSTANTS & CONFIGURATION
# ============================================
# Base paths
base_path = '/content/drive/MyDrive/Crime/'
csv_path = os.path.join(base_path, 'output.csv')
text_folder = os.path.join(base_path, 'data_for_tfidf/')
clean_csv_path = '/content/clean_crime_dataset.csv'

# Crime tools list (single source of truth)
CRIME_TOOLS = [
    "Baseball bat", "Bomb", "Knife", "Razor", "Machete", "Sword", "Dagger", "Club", "Mace",
    "Chains", "Ropes", "Shackles", "Grenade", "Scanner", "Printer", "Hidden camera", "Syringe",
    "Car", "Drugs", "Gun"
]

CRIME_TOOLS_LOWER = [tool.lower() for tool in CRIME_TOOLS]
CRIME_TOOLS_SET = set(CRIME_TOOLS_LOWER)

# ============================================
# 3. DATA LOADING FUNCTIONS
# ============================================
def load_text_files(folder_path, use_df_order=False, df=None):
    """Load text files from folder - flexible loading"""
    text_data = []
    if use_df_order and df is not None and 'File' in df.columns:
        # Load in order matching CSV
        for filename in df['File']:
            filepath = os.path.join(folder_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text_data.append(f.read())
            else:
                text_data.append("")
    else:
        # Load all files in directory
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text_data.append(f.read())
    return text_data

def load_all_text_combined(folder_path):
    """Load all text files into a single string"""
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                all_text += f.read() + " "
    return all_text

# ============================================
# 4. TEXT PROCESSING FUNCTIONS
# ============================================
def tokenize(text, remove_stopwords=True, min_word_length=2):
    """Clean and tokenize text"""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    if remove_stopwords:
        words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > min_word_length]
    else:
        words = [w for w in words if len(w) > min_word_length]
    return words

def extract_nouns(text):
    """Extract nouns from text using NLTK"""
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    return [(word, tag) for word, tag in tagged_tokens if tag in ['NN', 'NNS', 'NNP', 'NNPS']]

# ============================================
# 5. CRIME TOOL ANALYSIS FUNCTIONS
# ============================================
def find_new_tool_candidates(tokens, known_tools_set, min_frequency=5, top_n=30):
    """Find frequent unknown words as potential new crime tools"""
    word_counts = Counter(tokens)
    candidates = [
        (word, count) for word, count in word_counts.items()
        if word not in known_tools_set and count > min_frequency
    ]
    return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]

def analyze_crime_tool_occurrences(file_contents, file_names, crime_tools):
    """Analyze crime tool occurrences across files"""
    crime_tool_data = {tool: {'total_count': 0, 'files': {}} for tool in crime_tools}
    
    for file_name, content in zip(file_names, file_contents):
        for tool in crime_tools:
            occurrences = content.lower().count(tool.lower())
            if occurrences > 0:
                crime_tool_data[tool]['total_count'] += occurrences
                crime_tool_data[tool]['files'][file_name] = occurrences
    
    return crime_tool_data

def extract_potential_crime_tools_from_files(folder_path):
    """Extract potential crime tools using NLTK"""
    potential_crime_tools = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
                nouns = extract_nouns(text)
                for word, tag in nouns:
                    potential_crime_tools.append((filename, word))
    return pd.DataFrame(potential_crime_tools, columns=['File', 'Potential Crime Tool'])

# ============================================
# 6. MODEL TRAINING FUNCTIONS
# ============================================
def prepare_tfidf_features(text_data, max_features=5000):
    """Prepare TF-IDF features"""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    X = tfidf.fit_transform(text_data)
    return X, tfidf

def train_models(X_train, y_train, X_test, y_test, crime_tools_list):
    """Train and evaluate multiple models"""
    models = {
        'KNN': OneVsRestClassifier(KNeighborsClassifier(n_neighbors=7)),
        'SVM': OneVsRestClassifier(SVC()),
        'Logistic Regression': OneVsRestClassifier(LogisticRegression(max_iter=1000)),
        'Neural Network': OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)),
        'Decision Tree': OneVsRestClassifier(DecisionTreeClassifier()),
        'Random Forest': OneVsRestClassifier(RandomForestClassifier(n_estimators=100, max_depth=5))
    }
    
    results = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='micro', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='micro', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='micro', zero_division=0),
            'Model': model
        }
    
    return results

def predict_crime_tools(model, tfidf, text, crime_tools_list):
    """Predict crime tools from new text"""
    text_vectorized = tfidf.transform([text])
    prediction = model.predict(text_vectorized)
    return [tool for i, tool in enumerate(crime_tools_list) if prediction[0][i] == 1]

# ============================================
# 7. GRADIO DASHBOARD FUNCTIONS
# ============================================
def filter_crime_data(location=None, start_date=None, end_date=None, crime_type=None):
    """Filter crime data for Gradio dashboard"""
    try:
        df = pd.read_csv(clean_csv_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        if location:
            df = df[df["Location"].str.lower() == location.lower()]
        if start_date:
            start_date = pd.to_datetime(start_date, errors='coerce')
            df = df[df["Date"] >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date, errors='coerce')
            df = df[df["Date"] <= end_date]
        if crime_type:
            df = df[df["Crime_Type"].str.lower() == crime_type.lower()]
        
        return f"✅ {len(df)} record(s) found.", df
    except Exception as e:
        return f"❌ Error: {str(e)}", pd.DataFrame()

def create_dashboard():
    """Create and launch the Gradio dashboard"""
    try:
        # Load data for dashboard
        crime_df = pd.read_csv(clean_csv_path)
        crime_df['Date'] = pd.to_datetime(crime_df['Date'], errors='coerce')
        unique_locations = sorted(crime_df["Location"].dropna().unique())
        unique_crime_types = sorted(crime_df["Crime_Type"].dropna().unique())
        
        with gr.Blocks(theme=gr.themes.Soft(), title="Crime Analysis Dashboard") as demo:
            gr.Markdown("""
            # 🔎 Crime Analysis Dashboard
            ### Filter and Explore Crime Data
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Filter Options")
                    location = gr.Dropdown(
                        label="📍 Select Area", 
                        choices=unique_locations, 
                        interactive=True
                    )
                    start_date = gr.Textbox(
                        label="📅 Start Date (YYYY-MM-DD)", 
                        placeholder="Optional"
                    )
                    end_date = gr.Textbox(
                        label="📅 End Date (YYYY-MM-DD)", 
                        placeholder="Optional"
                    )
                    crime_type = gr.Dropdown(
                        label="🔪 Crime Type", 
                        choices=[""] + unique_crime_types, 
                        interactive=True
                    )
                    filter_button = gr.Button("🔍 Apply Filter", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Results")
                    status = gr.Textbox(label="Status", interactive=False)
                    filtered_output = gr.Dataframe(
                        label="Filtered Crime Records",
                        interactive=False,
                        wrap=True
                    )
            
            filter_button.click(
                fn=filter_crime_data,
                inputs=[location, start_date, end_date, crime_type],
                outputs=[status, filtered_output]
            )
            
            gr.Markdown("---")
            gr.Markdown("📊 *Dashboard shows filtered crime data based on your selections*")
        
        return demo
    except Exception as e:
        print(f"⚠️ Error creating dashboard: {e}")
        return None

# ============================================
# 8. MAIN EXECUTION
# ============================================
def main():
    print("=" * 60)
    print("CRIME ANALYSIS SYSTEM")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded CSV with {len(df)} records")
    
    # Load text content
    text_data = load_text_files(text_folder, use_df_order=True, df=df)
    df['content'] = text_data[:len(df)]
    
    # Filter empty content
    df = df[df['content'].str.strip().astype(bool)]
    print(f"✅ Loaded {len(df)} text files with content")
    
    # Prepare labels
    y = df[[col for col in df.columns if col in CRIME_TOOLS]]
    y = y.loc[:, y.nunique() > 1]
    active_tools = y.columns.tolist()
    print(f"✅ Using {len(active_tools)} active crime tool labels")
    
    # Prepare features
    X, tfidf = prepare_tfidf_features(df['content'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train models
    print("\n🤖 Training Models...")
    results = train_models(X_train, y_train, X_test, y_test, active_tools)
    
    # Display results
    print("\n📈 Model Performance Comparison:")
    results_df = pd.DataFrame({name: {k: v for k, v in res.items() if k != 'Model'} 
                               for name, res in results.items()}).T
    print(results_df.round(4))
    
    # Get best model
    best_model_name = results_df['F1-Score'].idxmax()
    best_model = results[best_model_name]['Model']
    print(f"\n🏆 Best Model: {best_model_name} with F1-Score: {results_df.loc[best_model_name, 'F1-Score']:.4f}")
    
    # Find new tool candidates
    print("\n🔍 Finding new crime tool candidates...")
    all_text = load_all_text_combined(text_folder)
    tokens = tokenize(all_text)
    candidates = find_new_tool_candidates(tokens, CRIME_TOOLS_SET)
    
    print("\n🔍 Possible New Crime Tools (not in known list):")
    for word, count in candidates[:20]:
        print(f"  {word} - {count} times")
    
    # Extract potential crime tools using NLTK
    print("\n🔍 Extracting potential crime tools with NLTK...")
    potential_tools_df = extract_potential_crime_tools_from_files(text_folder)
    print(f"✅ Found {len(potential_tools_df)} potential tool mentions")
    print(potential_tools_df.head(10))
    
    # Analyze crime tool occurrences
    print("\n📊 Analyzing crime tool occurrences...")
    file_names = [f for f in os.listdir(text_folder) if f.endswith('.txt')]
    file_contents = load_text_files(text_folder)
    occurrences = analyze_crime_tool_occurrences(file_contents, file_names, CRIME_TOOLS_LOWER)
    
    print("\nTop crime tools by occurrence:")
    for tool, data in sorted(occurrences.items(), key=lambda x: x[1]['total_count'], reverse=True)[:10]:
        if data['total_count'] > 0:
            print(f"  {tool}: {data['total_count']} occurrences across {len(data['files'])} files")
    
    # Example predictions
    print("\n🔮 Testing Predictions with Best Model:")
    test_texts = [
        "The suspect used a knife and there were drugs found at the crime scene. Also, a car was used to flee.",
        "Razor was used to cut the victim. Drugs were also found at the crime scene",
        "A bomb threat was called in and the suspect fled in a car"
    ]
    
    for test_text in test_texts:
        predicted = predict_crime_tools(best_model, tfidf, test_text, active_tools)
        print(f"\n📝 Text: {test_text[:60]}...")
        print(f"   Predicted Tools: {predicted if predicted else 'None'}")
    
    # Launch Gradio dashboard
    print("\n" + "=" * 60)
    print("🎨 Launching Gradio Dashboard...")
    print("=" * 60)
    
    demo = create_dashboard()
    if demo:
        demo.launch(share=True, debug=False)
    else:
        print("⚠️ Could not launch dashboard")

# Run the main function
if __name__ == "__main__":
    main()
