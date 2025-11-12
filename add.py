import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    accuracy_score, 
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
import os 
import webbrowser

# --- Page Config ---
st.set_page_config(
    page_title="NYC Collision Injury Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
RANDOM_SEED = 42

# =============================================================================
# CACHED FUNCTIONS (Data Loading & Model Training)
# =============================================================================

@st.cache_data
def load_data():
    """Loads 50k rows from the NYC OpenData API and aggressively cleans for caching."""
    st.write("Cache miss: Loading 50,000 rows from NYC OpenData API...")
    data_url = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
    params = {
        "$limit": 50000,  # 50k sample
        "$order": "crash_date DESC"
    }
    try:
        response = requests.get(data_url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            
            # --- AGGRESSIVE FIX: Convert all columns to string to guarantee hashability ---
            for col in df.columns:
                df[col] = df[col].astype(str)
            # --- AGGRESSIVE FIX END ---
            
            return df
        else:
            st.error(f"Failed to load data: {response.status_code}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request failed: {e}")
        return pd.DataFrame()

@st.cache_data
def clean_and_engineer(df_raw):
    """Cleans data, engineers features, and creates X and Y."""
    st.write("Cache miss: Cleaning data and engineering features...")
    df = df_raw.copy()
    
    # Since load_data converts everything to str, we skip the object-to-str check here
    
    # --- 1. Clean ALL Numeric Columns (Converting from string now) ---
    numeric_cols = [
        'number_of_persons_injured', 'number_of_persons_killed',
        'number_of_pedestrians_injured', 'number_of_pedestrians_killed',
        'number_of_cyclist_injured', 'number_of_cyclist_killed',
        'number_of_motorist_injured', 'number_of_motorist_killed',
        'latitude', 'longitude'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # --- 2. Clean Temporal/Categorical Columns ---
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    # Need to handle potential NaN values after string conversion for crash_time
    df['crash_hour'] = pd.to_datetime(df['crash_time'].str.strip(), format='%H:%M', errors='coerce').dt.hour.fillna(-1).astype(int)
    df['day_of_week'] = df['crash_date'].dt.day_name().fillna("Unspecified")
    df['month'] = df['crash_date'].dt.month
    
    df['borough'] = df['borough'].fillna("Unspecified").replace(["Unknown", ""], "Unspecified")
    df['contributing_factor_vehicle_1'] = df['contributing_factor_vehicle_1'].fillna("Unspecified").replace(["Unknown", ""], "Unspecified")
    df['vehicle_type_code1'] = df['vehicle_type_code1'].fillna("Unspecified").replace(["Unknown", ""], "Unspecified")
    df['vehicle_type_code2'] = df['vehicle_type_code2'].fillna("Unspecified").replace(["Unknown", ""], "Unspecified")

    # --- 3. Create Target Variable (Y_class) ---
    df['total_injured'] = (
        df['number_of_pedestrians_injured'] + 
        df['number_of_cyclist_injured'] + 
        df['number_of_motorist_injured']
    )
    Y_class = (df['total_injured'] > 0).astype(int)
    
    # --- 4. Create Predictor Variables (X_features) ---
    X_features = pd.DataFrame(index=df.index)
    
    X_features['latitude'] = df['latitude'].replace(0, np.nan).fillna(0)
    X_features['longitude'] = df['longitude'].replace(0, np.nan).fillna(0)
    X_features = X_features.join(pd.get_dummies(df['borough'], prefix='borough', drop_first=True, dtype=int))
    X_features = X_features.join(pd.get_dummies(df['day_of_week'], prefix='day', drop_first=True, dtype=int))
    X_features = X_features.join(pd.get_dummies(df['crash_hour'], prefix='hour', drop_first=True, dtype=int))

    top_10_factors = df['contributing_factor_vehicle_1'].value_counts().nlargest(10).index
    df['factor_top10'] = df['contributing_factor_vehicle_1'].apply(lambda x: x if x in top_10_factors else 'Other')
    X_features = X_features.join(pd.get_dummies(df['factor_top10'], prefix='factor', drop_first=True, dtype=int))

    top_10_vehicles = df['vehicle_type_code1'].value_counts().nlargest(10).index
    df['vehicle_top10'] = df['vehicle_type_code1'].apply(lambda x: x if x in top_10_vehicles else 'Other')
    X_features = X_features.join(pd.get_dummies(df['vehicle_top10'], prefix='vehicle', drop_first=True, dtype=int))
    
    return df, X_features, Y_class

@st.cache_data
def create_hotspot_map(_df):
    """Creates the Folium hotspot map object."""
    st.write("Cache miss: Generating hotspot map...")
    # Filter data
    map_data = _df.dropna(subset=['latitude', 'longitude'])
    map_data = map_data[(map_data['latitude'].abs() > 0.01) & (map_data['longitude'].abs() > 0.01)]
    
    # Sample for performance
    if len(map_data) > 20000:
        map_data = map_data.sample(20000, random_state=RANDOM_SEED)
    
    heat_data = list(zip(map_data['latitude'], map_data['longitude']))
    
    center_lat = map_data['latitude'].mean()
    center_lon = map_data['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbdarkmatter")
    HeatMap(heat_data).add_to(m)
    return m

@st.cache_data
def get_splits_and_scaler(X, Y):
    """Splits data and creates a scaler object."""
    st.write("Cache miss: Splitting data and defining scaler...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y
    )
    preprocessor = StandardScaler()
    return X_train, X_test, y_train, y_test, preprocessor

@st.cache_resource
def train_baseline_models(_X_train, _y_train, _preprocessor):
    """Trains all 6 baseline models and returns them in a dict."""
    st.write("Cache miss: Training baseline models (this may take a moment)...")
    
    linear_svc = LinearSVC(random_state=RANDOM_SEED, dual="auto", max_iter=2000)
    calibrated_svc = CalibratedClassifierCV(linear_svc, cv=3)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED),
        "KNN": KNeighborsClassifier(),
        "LinearSVC (Calibrated)": calibrated_svc, 
        "Neural Network": MLPClassifier(random_state=RANDOM_SEED, max_iter=1000, early_stopping=True)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        st.write(f"Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', _preprocessor),
                                    ('classifier', model)])
        pipeline.fit(_X_train, _y_train)
        trained_models[name] = pipeline
        
    return trained_models

@st.cache_resource
def train_tuned_models(_X_train, _y_train, _preprocessor):
    """Runs GridSearchCV for the two NN models on a sampled dataset."""
    st.write("Cache miss: Tuning Neural Networks (this may take several minutes)...")
    
    # --- CRITICAL FIX FOR MEMORY LIMITS START ---
    # Apply a 50% stratified sample of the training data to reduce the load on GridSearchCV/SMOTE
    if len(_X_train) > 10000:
        X_train_sampled, _, y_train_sampled, _ = train_test_split(
            _X_train, _y_train, train_size=0.5, random_state=RANDOM_SEED, stratify=_y_train
        )
        st.write(f"Tuning on a reduced sample size: {len(X_train_sampled)} rows.")
    else:
        X_train_sampled = _X_train
        y_train_sampled = _y_train
    # --- CRITICAL FIX FOR MEMORY LIMITS END ---
    
    param_grid = {
        'classifier__hidden_layer_sizes': [(50,), (100,)], # Smaller grid for app speed
        'classifier__alpha': [0.0001, 0.001],
    }

    # --- 1. "No SMOTE" Grid Search ---
    pipeline_no_smote = Pipeline(steps=[
        ('preprocessor', _preprocessor),
        ('classifier', MLPClassifier(random_state=RANDOM_SEED, max_iter=1000, early_stopping=True))
    ])
    grid_search_no_smote = GridSearchCV(
        pipeline_no_smote, param_grid, cv=2, scoring='roc_auc', n_jobs=-1, verbose=1 # Reduced CV for stability
    )
    # Fit on the sampled data
    grid_search_no_smote.fit(X_train_sampled, y_train_sampled)
    st.write("Tuning (No SMOTE) complete.")
    
    # --- 2. "WITH SMOTE" Grid Search ---
    pipeline_with_smote = ImbPipeline(steps=[
        ('preprocessor', _preprocessor),
        ('smote', SMOTE(random_state=RANDOM_SEED)), # <-- Add SMOTE step
        ('classifier', MLPClassifier(random_state=RANDOM_SEED, max_iter=1000, early_stopping=True))
    ])
    grid_search_with_smote = GridSearchCV(
        pipeline_with_smote, param_grid, cv=2, scoring='roc_auc', n_jobs=-1, verbose=1 # Reduced CV for stability
    )
    # Fit on the sampled data
    grid_search_with_smote.fit(X_train_sampled, y_train_sampled)
    st.write("Tuning (WITH SMOTE) complete.")
    
    return grid_search_no_smote, grid_search_with_smote

# =============================================================================
# HELPER FUNCTIONS (Plotting & Analysis)
# =============================================================================

def get_baseline_results(trained_models, X_test, y_test):
    """Generates predictions and results from trained baseline models."""
    results = []
    roc_data = {}
    
    for name, pipeline in trained_models.items():
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results.append({
            "Model": name, "Accuracy": accuracy, "ROC-AUC": auc,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        })
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
        
    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    return results_df, roc_data

def get_tuned_results(baseline_results_df, model_no_smote, model_with_smote, X_test, y_test):
    """Generates predictions and results from tuned models."""
    
    baseline_nn_row = baseline_results_df[baseline_results_df['Model'] == 'Neural Network'].iloc[0].to_dict()
    
    # 1. Evaluate "Tuned (No SMOTE)"
    y_pred_no_smote = model_no_smote.predict(X_test)
    y_pred_proba_no_smote = model_no_smote.predict_proba(X_test)[:, 1]
    cm_no_smote = confusion_matrix(y_test, y_pred_no_smote)
    tn_no_smote, fp_no_smote, fn_no_smote, tp_no_smote = cm_no_smote.ravel()
    tuned_no_smote_results = {
        "Model": "NN (Tuned, No SMOTE)",
        "Accuracy": accuracy_score(y_test, y_pred_no_smote),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba_no_smote),
        "TP": tp_no_smote, "TN": tn_no_smote, "FP": fp_no_smote, "FN": fn_no_smote
    }
    
    # 2. Evaluate "Tuned (WITH SMOTE)"
    y_pred_with_smote = model_with_smote.predict(X_test)
    y_pred_proba_with_smote = model_with_smote.predict_proba(X_test)[:, 1]
    cm_with_smote = confusion_matrix(y_test, y_pred_with_smote)
    tn_with_smote, fp_with_smote, fn_with_smote, tp_with_smote = cm_with_smote.ravel()
    tuned_with_smote_results = {
        "Model": "NN (Tuned + SMOTE)",
        "Accuracy": accuracy_score(y_test, y_pred_with_smote),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba_with_smote),
        "TP": tp_with_smote, "TN": tn_with_smote, "FP": fp_with_smote, "FN": fn_with_smote
    }
    
    comparison_df = pd.DataFrame([
        baseline_nn_row,  
        tuned_no_smote_results, 
        tuned_with_smote_results
    ]).sort_values(by="ROC-AUC", ascending=False)
    
    return comparison_df, cm_no_smote, cm_with_smote

def create_sankey_fig(df):
    """Generates the Sankey Diagram Plotly figure."""
    sankey_data = df.groupby(['vehicle_type_code1', 'vehicle_type_code2']).size().reset_index(name='count')
    sankey_data = sankey_data[
        (sankey_data['vehicle_type_code1'] != 'Unspecified') &
        (sankey_data['vehicle_type_code2'] != 'Unspecified')
    ]
    sankey_data = sankey_data.nlargest(25, 'count')

    all_nodes = pd.concat([sankey_data['vehicle_type_code1'], sankey_data['vehicle_type_code2']]).unique()
    node_dict = {node: i for i, node in enumerate(all_nodes)}

    links = pd.DataFrame()
    links['source'] = sankey_data['vehicle_type_code1'].map(node_dict)
    links['target'] = sankey_data['vehicle_type_code2'].map(node_dict)
    links['value'] = sankey_data['count']

    # Color Logic
    color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
                     '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    node_colors = []
    node_color_map = {}
    for i, node_name in enumerate(all_nodes):
        color = color_palette[i % len(color_palette)]
        node_colors.append(color)
        node_color_map[node_name] = color

    link_colors = []
    for source_node_name in sankey_data['vehicle_type_code1']:
        hex_color = node_color_map[source_node_name].lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        link_colors.append(f'rgba({r},{g},{b},0.4)')

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                  label=all_nodes, color=node_colors),
        link=dict(source=links['source'], target=links['target'],
                  value=links['value'], color=link_colors)
    )])
    fig.update_layout(
        title_text="Top 25 Most Common Vehicle Collision Combinations", 
        font_size=12,
        width=1400,
        height=1100,
        margin=dict(t=300, b=50, l=50, r=50) 
    )
    return fig

# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================

# --- 1. Load, Clean, and Split Data (using cache) ---
with st.spinner('Loading and cleaning data (first load may take a minute)...'):
    df_raw = load_data()
    if df_raw.empty:
        st.stop()
    df, X_features, Y_class = clean_and_engineer(df_raw)
    X_train, X_test, y_train, y_test, preprocessor = get_splits_and_scaler(X_features, Y_class)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ("Introduction & Data", 
     "Descriptive Analysis", 
     "Diagnostic Analysis", 
     "Predictive Modeling (Baseline)", 
     "Predictive Modeling (Tuning & SMOTE)", 
     "Conclusion & Recommendations")
)

# --- 2. Page Navigation ---

if page == "Introduction & Data":
    st.title("Beyond the Crash: A Multi-Dimensional Analysis of Injury Risk")
    st.markdown("""
    Motor vehicle collisions are a leading cause of unintentional injury in urban environments. 
    This application analyzes the NYC OpenData "Motor Vehicle Collisions" dataset to answer a critical question: 
    *What factors determine if a collision results in an injury?*
    """)
    st.subheader("Raw Data Sample (Most Recent 50,000 Collisions)")
    st.dataframe(df_raw.head())
    
    with st.expander("View Full Cleaned Data & Feature Engineering"):
        st.dataframe(df.head())
        st.dataframe(X_features.head())

    st.subheader("Data Dictionary")
    st.markdown("""
    * **`is_injury` (Target):** Binary. `1` if `NUMBER_OF_PERSONS_INJURED` > 0, else `0`.
    * **`latitude`, `longitude` (Numeric):** WGS 1984 coordinates.
    * **`borough` (Categorical):** One-hot encoded (Brooklyn, Queens, Manhattan, etc.).
    * **`day_of_week` (Categorical):** One-hot encoded (Monday, Tuesday, etc.).
    * **`crash_hour` (Categorical):** One-hot encoded (0-23).
    * **`factor_top10` (Categorical):** One-hot encoded. Top 10 factors (e.g., "Driver Inattention," "Unsafe Speed") or "Other."
    * **`vehicle_top10` (Categorical):** One-hot encoded. Top 10 types (e.g., "Sedan," "Station Wagon/Sport Utility Vehicle") or "Other."
    """)

elif page == "Descriptive Analysis":
    st.header("Descriptive Analysis: What, When, and Where")
    st.markdown("This section explores the basic patterns of collisions.")
    
    st.subheader("When do collisions occur?")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        # Filter out the placeholder -1 hour
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sns.countplot(x='crash_hour', data=df[df['crash_hour'] != -1], ax=ax1, palette='viridis')
        ax1.set_title('Collisions by Hour of Day')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sns.countplot(x='day_of_week', data=df, ax=ax2, palette='plasma',
                          order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        ax2.set_title('Collisions by Day of Week')
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

    # --- Time-Series and Heatmap ---
    st.write("---")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    with warnings.catch_warnings():
         warnings.simplefilter("ignore", FutureWarning)
         df.set_index('crash_date').resample('ME').size().plot(ax=ax3) 
    ax3.set_title('Collisions Over Time (Monthly Trend)')
    st.pyplot(fig3)
    
    st.write("---")
    fig4, ax4 = plt.subplots(figsize=(14, 8))
    df_heatmap = df.groupby(['day_of_week', 'crash_hour']).size().unstack(fill_value=0)
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_heatmap = df_heatmap.reindex(days_order)
    sns.heatmap(df_heatmap, cmap='Reds', linewidths=.5, ax=ax4)
    ax4.set_title('Collisions Heatmap: Hour of Day vs. Day of Week')
    st.pyplot(fig4)
    st.write("---")

    st.subheader("Where do collisions occur?")
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    with warnings.catch_warnings():
         warnings.simplefilter("ignore", FutureWarning)
         sns.countplot(y='borough', data=df, ax=ax5, palette='coolwarm',
                      order=df['borough'].value_counts().index)
    ax5.set_title('Total Collisions by Borough')
    st.pyplot(fig5)
    
    # --- Hotspot Map ---
    st.subheader("Collision Hotspot Map")
    st.markdown("This map shows the highest concentrations of the 50,000 most recent crashes.")
    with st.spinner("Generating interactive hotspot map..."):
        m = create_hotspot_map(df)
        st_folium(m, width='100%', height=500)
    st.write("---")

    
    st.subheader("What is the human impact?")
    # --- Victim Plots ---
    col1, col2 = st.columns(2)
    with col1:
        fig6, ax6 = plt.subplots()
        total_injured = {
            'Pedestrians': df['number_of_pedestrians_injured'].sum(),
            'Cyclists': df['number_of_cyclist_injured'].sum(),
            'Motorists': df['number_of_motorist_injured'].sum()
        }
        ax6.pie(total_injured.values(), labels=total_injured.keys(), autopct='%1.1f%%',
                startangle=140, colors=sns.color_palette('muted'))
        ax6.set_title('Breakdown of Total Persons Injured by Type')
        st.pyplot(fig6)
    
    with col2:
        fig7, ax7 = plt.subplots(figsize=(12, 7))
        borough_injuries = df.groupby('borough')[['number_of_pedestrians_injured', 'number_of_cyclist_injured', 'number_of_motorist_injured']].sum()
        borough_injuries = borough_injuries.rename(columns={
            'number_of_pedestrians_injured': 'Pedestrians',
            'number_of_cyclist_injured': 'Cyclists',
            'number_of_motorist_injured': 'Motorists'
        })
        borough_injuries.drop('Unspecified', errors='ignore', inplace=True)
        borough_injuries['total'] = borough_injuries.sum(axis=1)
        borough_injuries = borough_injuries.sort_values('total', ascending=False).drop('total', axis=1)
        borough_injuries.plot(kind='bar', stacked=True, ax=ax7, colormap='viridis')
        ax7.set_title('Injuries by Type and Borough')
        st.pyplot(fig7)
    
    st.write("---")
    fig8, ax8 = plt.subplots(figsize=(12, 7))
    with warnings.catch_warnings():
         warnings.simplefilter("ignore", FutureWarning)
         victim_trends = df.set_index('crash_date').resample('ME')[['number_of_pedestrians_injured', 'number_of_cyclist_injured', 'number_of_motorist_injured']].sum().rename(columns={
             'number_of_pedestrians_injured': 'Pedestrians Injured',
             'number_of_cyclist_injured': 'Cyclists Injured',
             'number_of_motorist_injured': 'Motorists Injured'
         })
    victim_trends.plot(ax=ax8, colormap='Dark2')
    ax8.set_title('Monthly Injuries by Victim Type')
    st.pyplot(fig8)


elif page == "Diagnostic Analysis":
    st.header("Diagnostic Analysis: Why Do Injuries Happen?")
    st.markdown("This section explores the *causes* and *relationships* behind collisions.")
    
    # --- OLS Regression ---
    st.subheader("Statistical Factor Analysis (OLS Regression)")
    with st.spinner("Running OLS Regression..."):
        Y_ols = df['total_injured']
        X_ols = sm.add_constant(X_features, has_constant='add').astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ols_model = sm.OLS(Y_ols, X_ols).fit()
    
    st.text(ols_model.summary())
    st.markdown(f"""
    * **Interpretation:** The **R-squared ({ols_model.rsquared:.3f})** is very low, which is a key finding. It means our model can only explain {ols_model.rsquared:.1%} of the variation in injuries. This proves that predicting the *exact number* of injuries is extremely difficult and dominated by random chance and unmeasured factors (like speed at impact, seatbelt use, etc.).
    * **Key Insight:** However, the **P>|t|** column (p-value) is `0.000` for many factors, proving they are **highly statistically significant**. Factors like "Failure to Yield Right-of-Way" and "Traffic Control Disregarded" are clearly linked to more injuries.
    """)
    st.write("---")

    # --- Top Factors Plots ---
    st.subheader("What are the most common causes?")
    fig_factor, ax_factor = plt.subplots(figsize=(12, 8))
    factor_data = df[df['contributing_factor_vehicle_1'] != 'Unspecified']['contributing_factor_vehicle_1']
    factor_counts = factor_data.value_counts().nlargest(15)
    with warnings.catch_warnings():
         warnings.simplefilter("ignore", FutureWarning)
         sns.barplot(x=factor_counts.values, y=factor_counts.index, palette='rocket', ax=ax_factor)
    ax_factor.set_title('Top 15 Contributing Factors for Collisions (Vehicle 1)')
    st.pyplot(fig_factor)
    
    st.subheader("What causes are the most *fatal*?")
    fig_fatal, ax_fatal = plt.subplots(figsize=(12, 8))
    fatal_crashes = df[df['number_of_persons_killed'] > 0]
    if not fatal_crashes.empty:
        fatal_factor_data = fatal_crashes[fatal_crashes['contributing_factor_vehicle_1'] != 'Unspecified']['contributing_factor_vehicle_1']
        fatal_factor_counts = fatal_factor_data.value_counts().nlargest(15)
        if not fatal_factor_counts.empty:
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", FutureWarning)
                 sns.barplot(x=fatal_factor_counts.values, y=fatal_factor_counts.index, palette='Reds_r', ax=ax_fatal)
            ax_fatal.set_title('Top 15 Contributing Factors in *Fatal* Collisions (Vehicle 1)')
        else:
            st.write("No fatal crashes with specified factors in this sample.")
    else:
        st.write("No fatal crashes in this sample.")
    st.pyplot(fig_fatal)
    st.write("---")

    # --- Vehicle Type Plots ---
    st.subheader("What vehicle types are most involved?")
    fig_vehicle, ax_vehicle = plt.subplots(figsize=(12, 8))
    vehicle_data = df[df['vehicle_type_code1'] != 'Unspecified']['vehicle_type_code1']
    vehicle_counts = vehicle_data.value_counts().nlargest(15)
    with warnings.catch_warnings():
         warnings.simplefilter("ignore", FutureWarning)
         sns.barplot(x=vehicle_counts.values, y=vehicle_counts.index, palette='Spectral', ax=ax_vehicle)
    ax_vehicle.set_title('Top 15 Vehicle Types Involved in Collisions (Vehicle 1)')
    st.pyplot(fig_vehicle)

    # --- Sankey Diagram ---
    st.subheader("What are the most common collision types?")
    with st.spinner("Building Sankey diagram..."):
        sankey_fig = create_sankey_fig(df)
        # Suppress future warning regarding use_container_width
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module='streamlit')
            st.plotly_chart(sankey_fig, use_container_width=True)
        
        # Save the Sankey diagram to HTML
        sankey_filename = '5_2_vehicle_sankey_diagram.html'
        sankey_fig.write_html(sankey_filename)
        st.caption(f"Interactive Sankey diagram also saved to: {sankey_filename}")


elif page == "Predictive Modeling (Baseline)":
    st.header("Predictive Modeling: Baseline Performance")
    
    # Calculate percentages first
    pct_0 = y_test.value_counts(normalize=True)[0]
    pct_1 = y_test.value_counts(normalize=True)[1]
    
    st.markdown(f"""
    Can we predict the binary outcome: **Injury (1)** vs. **No Injury (0)**?
    
    Test Set Class Balance:
    * No Injury (0): {pct_0:.1%}
    * Injury (1): {pct_1:.1%}
    
    This **{pct_0/pct_1:.1f}-to-1 class imbalance** is the central challenge. 
    Accuracy is a misleading metric, as a model just guessing "No Injury" would be {pct_0:.1%} accurate.
    """)

    with st.spinner("Loading trained baseline models..."):
        trained_models = train_baseline_models(X_train, y_train, preprocessor)
    
    # Generate results
    results_df, roc_data = get_baseline_results(trained_models, X_test, y_test)
    
    # --- Plot Confusion Matrices ---
    st.subheader("Baseline Confusion Matrices")
    class_labels = ['No Injury (0)', 'Injury (1)']
    
    # Create a 2x3 grid for the plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten() # Make it easy to iterate
    
    for i, (name, pipeline) in enumerate(trained_models.items()):
        cm = confusion_matrix(y_test, pipeline.predict(X_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues, ax=axes[i])
        axes[i].set_title(name)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    st.pyplot(fig)

    # --- Plot ROC Curve ---
    st.subheader("Combined ROC Curve (Baseline)")
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.500)')
    for name, data in roc_data.items():
        ax_roc.plot(data['fpr'], data['tpr'], label=f"{name} (AUC = {data['auc']:.3f})")
    ax_roc.set_xlabel('False Positive Rate (FPR)')
    ax_roc.set_ylabel('True Positive Rate (TPR)')
    ax_roc.set_title('Combined ROC Curve for All Models', fontsize=16)
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True)
    st.pyplot(fig_roc)
    
    # --- Show Final Table ---
    st.subheader("Final Summary Table (Sorted by ROC-AUC)")
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'ROC-AUC']))

elif page == "Predictive Modeling (Tuning & SMOTE)":
    st.header("Tuning the Neural Network & Fixing Imbalance")
    
    # --- UI EXPLANATION FOR SAMPLING FIX ---
    st.info("""
    **Note on Stability (Resource Management):**
    The process of hyperparameter tuning (`GridSearchCV`) combined with synthetic data generation (`SMOTE`) is extremely memory-intensive. To prevent the application from crashing due to resource limits in the deployment environment, the tuning process is now performed on a **stratified 50% sample** of the original training data. All final results (metrics, confusion matrices) are calculated against the **full 20% test set** to ensure the final model evaluation remains accurate.
    """)
    # --- END UI EXPLANATION ---

    st.markdown("""
    The baseline models failed because they could not find the rare "Injury" class. 
    We will now try two strategies:
    1.  **Tuning:** Tune the best baseline model (Neural Network) to optimize its parameters.
    2.  **Tuning + SMOTE:** Tune the Neural Network *after* using SMOTE to create a balanced training dataset.
    """)

    with st.spinner("Tuning Neural Networks (this may take 10+ minutes)..."):
        model_no_smote, model_with_smote = train_tuned_models(X_train, y_train, preprocessor)
    
    # Get baseline results to compare
    baseline_models = train_baseline_models(X_train, y_train, preprocessor)
    baseline_results_df, _ = get_baseline_results(baseline_models, X_test, y_test)
    
    # Get tuned results
    comparison_df, cm_no_smote, cm_with_smote = get_tuned_results(
        baseline_results_df, model_no_smote, model_with_smote, X_test, y_test
    )
    
    # --- Plot Tuned Confusion Matrices ---
    st.subheader("Tuned vs. SMOTE Confusion Matrices")
    class_labels = ['No Injury (0)', 'Injury (1)']
    col1, col2 = st.columns(2)
    
    with col1:
        fig_no_smote, ax1 = plt.subplots()
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_no_smote, display_labels=class_labels)
        disp1.plot(cmap=plt.cm.Blues, ax=ax1)
        ax1.set_title("NN (Tuned, No SMOTE)")
        st.pyplot(fig_no_smote)
        
    with col2:
        fig_with_smote, ax2 = plt.subplots()
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_with_smote, display_labels=class_labels)
        disp2.plot(cmap=plt.cm.Blues, ax=ax2)
        ax2.set_title("NN (Tuned + SMOTE)")
        st.pyplot(fig_with_smote)
        
    st.subheader("Final Model Comparison (Sorted by ROC-AUC)")
    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'ROC-AUC', 'TP']))
    
    st.subheader("Analysis")
    # --- START: DYNAMIC CONCLUSION ---
    try:
        baseline_fn = comparison_df.loc[comparison_df['Model'] == 'Neural Network', 'FN'].values[0]
        smote_fn = comparison_df.loc[comparison_df['Model'] == 'NN (Tuned + SMOTE)', 'FN'].values[0]
        baseline_tp = comparison_df.loc[comparison_df['Model'] == 'Neural Network', 'TP'].values[0]
        smote_tp = comparison_df.loc[comparison_df['Model'] == 'NN (Tuned + SMOTE)', 'TP'].values[0]
        
        st.markdown(f"""
        The results are clear:
        1.  **Tuning Alone Failed:** The `NN (Tuned, No SMOTE)` model performed identically to the baseline. This proves the issue was not the model's parameters, but the imbalanced data.
        2.  **SMOTE Succeeded:** The `NN (Tuned + SMOTE)` model **fundamentally fixed the model's behavior.**
            * **False Negatives** (missed injuries) plummeted from **{baseline_fn:,}** to **{smote_fn:,}**.
            * **True Positives** (found injuries) skyrocketed from **{baseline_tp:,}** to **{smote_tp:,}**.
            
        The SMOTE model is the only one that achieves the primary goal: **finding the crashes that result in injury.**
        """)
    except Exception as e:
        st.error(f"Could not generate dynamic conclusion. {e}")
    # --- END: DYNAMIC CONCLUSION ---


elif page == "Conclusion & Recommendations":
    st.header("Conclusion & Recommendations")
    
    # --- START: DYNAMIC CONCLUSION ---
    # We need to re-run the models here to get the data for the conclusion
    # This is fast because the models are cached
    with st.spinner("Loading models for final report..."):
        baseline_models = train_baseline_models(X_train, y_train, preprocessor)
        baseline_results_df, _ = get_baseline_results(baseline_models, X_test, y_test)
        
        model_no_smote, model_with_smote = train_tuned_models(X_train, y_train, preprocessor)
        
        comparison_df, _, _ = get_tuned_results(
            baseline_results_df, model_no_smote, model_with_smote, X_test, y_test
        )
    
    try:
        # Get the numbers from the final DataFrame
        pct_0 = y_test.value_counts(normalize=True)[0]
        pct_1 = y_test.value_counts(normalize=True)[1]
        
        baseline_acc = comparison_df.loc[comparison_df['Model'] == 'Neural Network', 'Accuracy'].values[0]
        baseline_fn = comparison_df.loc[comparison_df['Model'] == 'Neural Network', 'FN'].values[0]
        baseline_tp = comparison_df.loc[comparison_df['Model'] == 'Neural Network', 'TP'].values[0]
        baseline_fn_pct = baseline_fn / (baseline_fn + baseline_tp)

        smote_acc = comparison_df.loc[comparison_df['Model'] == 'NN (Tuned + SMOTE)', 'Accuracy'].values[0]
        smote_auc = comparison_df.loc[comparison_df['Model'] == 'NN (Tuned + SMOTE)', 'ROC-AUC'].values[0]
        smote_fn = comparison_df.loc[comparison_df['Model'] == 'NN (Tuned + SMOTE)', 'FN'].values[0]
        smote_tp = comparison_df.loc[comparison_df['Model'] == 'NN (Tuned + SMOTE)', 'TP'].values[0]
        smote_fp = comparison_df.loc[comparison_df['Model'] == 'NN (Tuned + SMOTE)', 'FP'].values[0]

        st.markdown(f"""
        ### The Problem: Accuracy vs. Reality
        Our baseline models, including the top-performing **`Neural Network`**, all achieved a high and misleading **Accuracy of {baseline_acc:.1%}**. 
        A closer look at the results showed this was a failure, not a success. The models were simply guessing the majority "No Injury" class ({pct_0:.1%} of the data) and failed to learn the patterns for the "Injury" class.
        
        This is proven by the **{baseline_fn:,} False Negatives**—our best baseline model *failed to identify {baseline_fn_pct:.1%} of all actual injury crashes.*
        
        ### The Solution: SMOTE's Impact
        Our goal was to fix this imbalance. The comparison table clearly shows the effect of our strategies:
        
        * **1. `NN (Tuned, No SMOTE)`:** Hyperparameter tuning alone provided **no meaningful improvement**. The metrics are identical to the baseline, proving the model's weakness was not its parameters, but the imbalanced data.
        
        * **2. `NN (Tuned + SMOTE)`:** Applying SMOTE to the training data **fundamentally fixed the model's behavior.**
            * **False Negatives** (missed injuries) plummeted from **{baseline_fn:,}** to **{smote_fn:,}**.
            * **True Positives** (found injuries) skyrocketed from **{baseline_tp:,}** to **{smote_tp:,}**.
        
        This model did see its `Accuracy` drop to **{smote_acc:.1%}** and its `ROC-AUC` (distinguishing power) stay flat at **{smote_auc:.3f}**. 
        This is an expected trade-off: in forcing the model to find the rare "Injury" class, it made more mistakes on the "No Injury" class (**{smote_fp:,}** False Positives).
        
        ### Final Recommendation
        
        **The `NN (Tuned + SMOTE)` model is the *only* useful model for this problem.**
        
        For a public safety analysis, the cost of a **False Negative** (missing an injury) is far higher than the cost of a **False Positive** (flagging a safe crash for review). 
        The SMOTE-trained model was the *only one* capable of achieving the primary goal: finding the crashes that result in injury. It successfully identified **{smote_tp - baseline_tp:,} more injury crashes** than the baseline, making it the superior choice.
        
        The low final `ROC-AUC` (**{smote_auc:.3f}**) suggests we have reached the limit of what our current features (borough, time, cause) can predict. 
        Future work must focus on **engineering new features** (e.g., weather data, road type, speed limits) to improve the model's ability to distinguish between a crash and an *injurious* crash.
        """)
    except Exception as e:
        st.error(f"Could not generate final report. {e}")
    # --- END: DYNAMIC CONCLUSION ---
