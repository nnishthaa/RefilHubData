"""
RefillHub Market Analysis Dashboard
Author: Data Science Team
Description: Comprehensive market analysis dashboard with ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             silhouette_score, davies_bouldin_score, r2_score, mean_squared_error)
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="RefillHub Market Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 100%;
    }
    h1 {
        color: #2c5f2d;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #97be5a, #2c5f2d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2 {
        color: #2c5f2d;
        border-bottom: 3px solid #97be5a;
        padding-bottom: 10px;
    }
    h3 {
        color: #4a7c59;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #2c5f2d;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #97be5a;
        color: #2c5f2d;
    }
    .sidebar .sidebar-content {
        background-color: #e8f5e9;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("<h1>🌱 RefillHub Market Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: white; border-radius: 10px; margin-bottom: 30px;'>
    <h3 style='color: #2c5f2d;'>Data-Driven Decision Making for Sustainable Business Growth</h3>
    <p style='font-size: 16px; color: #666;'>
        Leveraging Classification, Clustering, Association Rule Mining, and Regression 
        to understand customer behavior and optimize refill service offerings.
    </p>
</div>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load the RefillHub survey data"""
    try:
        df = pd.read_csv('RefillHub_SyntheticSurvey.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Data preprocessing function
@st.cache_data
def preprocess_data(df):
    """Preprocess data for machine learning"""
    df_processed = df.copy()
    
    # Handle categorical variables
    le_dict = {}
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
        le_dict[col] = le
    
    return df_processed, le_dict

# Load data
df = load_data()

if df is not None:
    df_processed, le_dict = preprocess_data(df)
    
    # Sidebar navigation
    st.sidebar.image("https://img.icons8.com/color/96/000000/leaf.png", width=100)
    st.sidebar.title("📊 Navigation")
    
    menu_options = [
        "🏠 Home & Overview",
        "📈 Exploratory Data Analysis",
        "🎯 Classification Models",
        "🔍 Customer Segmentation (Clustering)",
        "🔗 Association Rule Mining",
        "💰 Willingness to Pay (Regression)",
        "📊 Interactive Insights",
        "🎯 Business Recommendations"
    ]
    
    choice = st.sidebar.radio("Select Analysis", menu_options)
    
    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Filters")
    
    age_groups = st.sidebar.multiselect(
        "Age Group",
        options=df['Age_Group'].unique(),
        default=df['Age_Group'].unique()
    )
    
    emirates = st.sidebar.multiselect(
        "Emirate",
        options=df['Emirate'].unique(),
        default=df['Emirate'].unique()
    )
    
    # Apply filters
    df_filtered = df[
        (df['Age_Group'].isin(age_groups)) &
        (df['Emirate'].isin(emirates))
    ]
    
    # ============================================================================
    # 1. HOME & OVERVIEW
    # ============================================================================
    if choice == "🏠 Home & Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h2 style='color: #2c5f2d; text-align: center;'>{}</h2>
                <p style='text-align: center; color: #666;'>Total Respondents</p>
            </div>
            """.format(len(df_filtered)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h2 style='color: #2c5f2d; text-align: center;'>{}</h2>
                <p style='text-align: center; color: #666;'>Features</p>
            </div>
            """.format(len(df.columns)), unsafe_allow_html=True)
        
        with col3:
            likely_users = len(df_filtered[df_filtered['Likely_to_Use_ReFillHub'] == 'Yes'])
            percentage = (likely_users / len(df_filtered) * 100)
            st.markdown("""
            <div class='metric-card'>
                <h2 style='color: #2c5f2d; text-align: center;'>{:.1f}%</h2>
                <p style='text-align: center; color: #666;'>Likely Users</p>
            </div>
            """.format(percentage), unsafe_allow_html=True)
        
        with col4:
            avg_willingness = df_filtered['Willingness_to_Pay_AED'].mean()
            st.markdown("""
            <div class='metric-card'>
                <h2 style='color: #2c5f2d; text-align: center;'>AED {:.2f}</h2>
                <p style='text-align: center; color: #666;'>Avg Willingness to Pay</p>
            </div>
            """.format(avg_willingness), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Data Preview")
            st.dataframe(df_filtered.head(10), use_container_width=True)
        
        with col2:
            st.subheader("📊 Data Statistics")
            st.write("**Dataset Shape:**", df_filtered.shape)
            st.write("**Missing Values:**", df_filtered.isnull().sum().sum())
            st.write("**Duplicate Rows:**", df_filtered.duplicated().sum())
            
            st.subheader("📈 Target Distribution")
            target_counts = df_filtered['Likely_to_Use_ReFillHub'].value_counts()
            fig = px.pie(
                values=target_counts.values,
                names=target_counts.index,
                title="Likely to Use RefillHub",
                color_discrete_sequence=['#2c5f2d', '#97be5a', '#e8f5e9']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data Quality Report
        st.markdown("---")
        st.subheader("🔍 Data Quality Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Features:**")
            numerical_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            st.write(f"Count: {len(numerical_cols)}")
            st.write(numerical_cols)
        
        with col2:
            st.write("**Categorical Features:**")
            categorical_cols = df_filtered.select_dtypes(include=['object']).columns.tolist()
            st.write(f"Count: {len(categorical_cols)}")
            st.write(categorical_cols)
    
    # ============================================================================
    # 2. EXPLORATORY DATA ANALYSIS
    # ============================================================================
    elif choice == "📈 Exploratory Data Analysis":
        st.header("📈 Exploratory Data Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Demographic Analysis",
            "🛍️ Purchase Behavior",
            "🌱 Sustainability Metrics",
            "💡 Correlation Analysis"
        ])
        
        with tab1:
            st.subheader("Demographic Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Age Group distribution
                fig = px.histogram(
                    df_filtered,
                    x='Age_Group',
                    color='Gender',
                    title="Age Group Distribution by Gender",
                    barmode='group',
                    color_discrete_sequence=['#2c5f2d', '#97be5a', '#c5e1a5']
                )
                fig.update_layout(xaxis_title="Age Group", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
                
                # Income distribution
                fig = px.box(
                    df_filtered,
                    x='Income',
                    y='Willingness_to_Pay_AED',
                    color='Income',
                    title="Willingness to Pay by Income Level",
                    color_discrete_sequence=px.colors.sequential.Greens
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Emirate distribution
                emirate_counts = df_filtered['Emirate'].value_counts()
                fig = px.bar(
                    x=emirate_counts.index,
                    y=emirate_counts.values,
                    title="Distribution Across Emirates",
                    labels={'x': 'Emirate', 'y': 'Count'},
                    color=emirate_counts.values,
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Education level
                fig = px.pie(
                    df_filtered,
                    names='Education',
                    title="Education Level Distribution",
                    color_discrete_sequence=px.colors.sequential.Greens
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Purchase Behavior Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Purchase frequency
                fig = px.histogram(
                    df_filtered,
                    x='Purchase_Frequency',
                    color='Likely_to_Use_ReFillHub',
                    title="Purchase Frequency vs Refill Likelihood",
                    barmode='group',
                    color_discrete_sequence=['#e74c3c', '#2c5f2d']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Monthly spending
                fig = px.violin(
                    df_filtered,
                    x='Monthly_Spending_Range',
                    y='Willingness_to_Pay_AED',
                    color='Monthly_Spending_Range',
                    title="Willingness to Pay by Monthly Spending",
                    box=True,
                    color_discrete_sequence=px.colors.sequential.Greens
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Purchase location
                location_counts = df_filtered['Purchase_Location'].value_counts()
                fig = px.bar(
                    x=location_counts.index,
                    y=location_counts.values,
                    title="Preferred Purchase Locations",
                    labels={'x': 'Location', 'y': 'Count'},
                    color=location_counts.values,
                    color_continuous_scale='Greens'
                )
                fig.update_xaxis(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Usage frequency
                fig = px.histogram(
                    df_filtered,
                    x='Usage_Frequency',
                    color='Likely_to_Use_ReFillHub',
                    title="Usage Frequency Distribution",
                    barmode='stack',
                    color_discrete_sequence=['#e74c3c', '#2c5f2d']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Sustainability Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Eco product usage
                eco_counts = df_filtered['Uses_Eco_Products'].value_counts()
                fig = px.pie(
                    values=eco_counts.values,
                    names=eco_counts.index,
                    title="Eco-Product Usage",
                    color_discrete_sequence=['#2c5f2d', '#97be5a', '#e8f5e9']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Importance scores
                importance_df = pd.DataFrame({
                    'Factor': ['Convenience', 'Price', 'Sustainability'],
                    'Average Score': [
                        df_filtered['Importance_Convenience'].mean(),
                        df_filtered['Importance_Price'].mean(),
                        df_filtered['Importance_Sustainability'].mean()
                    ]
                })
                fig = px.bar(
                    importance_df,
                    x='Factor',
                    y='Average Score',
                    title="Average Importance Scores",
                    color='Average Score',
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Plastic ban awareness
                awareness_counts = df_filtered['Aware_Plastic_Ban'].value_counts()
                fig = px.pie(
                    values=awareness_counts.values,
                    names=awareness_counts.index,
                    title="Plastic Ban Awareness",
                    color_discrete_sequence=['#2c5f2d', '#97be5a', '#e8f5e9']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Waste reduction score
                fig = px.histogram(
                    df_filtered,
                    x='Reduce_Waste_Score',
                    nbins=5,
                    title="Waste Reduction Score Distribution",
                    color_discrete_sequence=['#2c5f2d']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Correlation Analysis")
            
            # Select numerical columns for correlation
            numerical_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            
            # Calculate correlation matrix
            corr_matrix = df_filtered[numerical_cols].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                color_continuous_scale='RdYlGn',
                aspect='auto',
                labels=dict(color="Correlation")
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations with target variable
            st.subheader("Top Correlations with Willingness to Pay")
            if 'Willingness_to_Pay_AED' in numerical_cols:
                target_corr = corr_matrix['Willingness_to_Pay_AED'].sort_values(ascending=False)[1:11]
                
                fig = px.bar(
                    x=target_corr.values,
                    y=target_corr.index,
                    orientation='h',
                    title="Top 10 Features Correlated with Willingness to Pay",
                    labels={'x': 'Correlation Coefficient', 'y': 'Feature'},
                    color=target_corr.values,
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================================
    # 3. CLASSIFICATION MODELS
    # ============================================================================
    elif choice == "🎯 Classification Models":
        st.header("🎯 Classification: Predicting RefillHub Usage")
        
        st.markdown("""
        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h4 style='color: #2c5f2d;'>🎯 Business Question:</h4>
            <p style='font-size: 16px;'>
                Which customers are most likely to use RefillHub services? 
                Understanding this helps us target marketing efforts and optimize resource allocation.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare data for classification
        target_col = 'Likely_to_Use_ReFillHub'
        
        if target_col in df.columns:
            # Select features for modeling
            feature_cols = [col for col in df_processed.columns if col.endswith('_encoded')]
            feature_cols = [col for col in feature_cols if col != target_col + '_encoded']
            
            X = df_processed[feature_cols]
            y = df_processed[target_col + '_encoded']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Model selection
            st.subheader("🔧 Model Configuration")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                model_choice = st.selectbox(
                    "Select Classification Model",
                    ["Decision Tree", "Random Forest", "Gradient Boosting", "All Models"]
                )
                
                if st.button("🚀 Train Model(s)", key="train_classification"):
                    with st.spinner("Training models..."):
                        models = {}
                        results = {}
                        
                        if model_choice == "Decision Tree" or model_choice == "All Models":
                            dt = DecisionTreeClassifier(max_depth=10, random_state=42)
                            dt.fit(X_train, y_train)
                            models['Decision Tree'] = dt
                            
                            y_pred = dt.predict(X_test)
                            results['Decision Tree'] = {
                                'accuracy': accuracy_score(y_test, y_pred),
                                'predictions': y_pred,
                                'model': dt
                            }
                        
                        if model_choice == "Random Forest" or model_choice == "All Models":
                            rf = RandomForestClassifier(n_estimators=100, random_state=42)
                            rf.fit(X_train, y_train)
                            models['Random Forest'] = rf
                            
                            y_pred = rf.predict(X_test)
                            results['Random Forest'] = {
                                'accuracy': accuracy_score(y_test, y_pred),
                                'predictions': y_pred,
                                'model': rf
                            }
                        
                        if model_choice == "Gradient Boosting" or model_choice == "All Models":
                            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
                            gb.fit(X_train, y_train)
                            models['Gradient Boosting'] = gb
                            
                            y_pred = gb.predict(X_test)
                            results['Gradient Boosting'] = {
                                'accuracy': accuracy_score(y_test, y_pred),
                                'predictions': y_pred,
                                'model': gb
                            }
                        
                        # Store in session state
                        st.session_state['classification_results'] = results
                        st.session_state['classification_models'] = models
                        st.session_state['X_test'] = X_test
                        st.session_state['y_test'] = y_test
                        
                        st.success("✅ Models trained successfully!")
            
            with col2:
                if 'classification_results' in st.session_state:
                    results = st.session_state['classification_results']
                    
                    # Display accuracies
                    st.subheader("📊 Model Performance")
                    
                    for model_name, result in results.items():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric(model_name, f"{result['accuracy']:.2%}")
                        with col2:
                            st.progress(result['accuracy'])
            
            # Detailed results
            if 'classification_results' in st.session_state:
                st.markdown("---")
                st.subheader("📈 Detailed Results")
                
                results = st.session_state['classification_results']
                y_test = st.session_state['y_test']
                
                tab1, tab2, tab3 = st.tabs([
                    "Confusion Matrices",
                    "Classification Reports",
                    "Feature Importance"
                ])
                
                with tab1:
                    cols = st.columns(len(results))
                    
                    for idx, (model_name, result) in enumerate(results.items()):
                        with cols[idx]:
                            st.write(f"**{model_name}**")
                            
                            cm = confusion_matrix(y_test, result['predictions'])
                            
                            fig = px.imshow(
                                cm,
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['No', 'Yes'],
                                y=['No', 'Yes'],
                                text_auto=True,
                                color_continuous_scale='Greens'
                            )
                            fig.update_layout(
                                title=f"{model_name}",
                                width=300,
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    for model_name, result in results.items():
                        st.write(f"**{model_name} Classification Report:**")
                        
                        report = classification_report(
                            y_test,
                            result['predictions'],
                            target_names=['No', 'Yes'],
                            output_dict=True
                        )
                        
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.background_gradient(cmap='Greens'), use_container_width=True)
                        st.markdown("---")
                
                with tab3:
                    for model_name, result in results.items():
                        model = result['model']
                        
                        if hasattr(model, 'feature_importances_'):
                            st.write(f"**{model_name} Feature Importance:**")
                            
                            importance_df = pd.DataFrame({
                                'Feature': [col.replace('_encoded', '') for col in X.columns],
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False).head(15)
                            
                            fig = px.bar(
                                importance_df,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title=f"Top 15 Features - {model_name}",
                                color='Importance',
                                color_continuous_scale='Greens'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown("---")
    
    # ============================================================================
    # 4. CUSTOMER SEGMENTATION (CLUSTERING)
    # ============================================================================
    elif choice == "🔍 Customer Segmentation (Clustering)":
        st.header("🔍 Customer Segmentation using Clustering")
        
        st.markdown("""
        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h4 style='color: #2c5f2d;'>🎯 Business Question:</h4>
            <p style='font-size: 16px;'>
                Can we identify distinct customer segments based on their behavior, preferences, and demographics?
                This helps in creating targeted marketing strategies for each segment.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Select features for clustering
        cluster_features = st.multiselect(
            "Select features for clustering:",
            options=[col for col in df_processed.columns if col.endswith('_encoded')],
            default=[col for col in df_processed.columns if col.endswith('_encoded')][:10]
        )
        
        if cluster_features:
            X_cluster = df_processed[cluster_features].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🔧 Clustering Configuration")
                
                n_clusters = st.slider("Number of Clusters", 2, 10, 4)
                
                if st.button("🚀 Perform Clustering", key="run_clustering"):
                    with st.spinner("Performing clustering analysis..."):
                        # K-Means clustering
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(X_scaled)
                        
                        # Calculate metrics
                        silhouette = silhouette_score(X_scaled, clusters)
                        davies_bouldin = davies_bouldin_score(X_scaled, clusters)
                        
                        # Store results
                        st.session_state['clusters'] = clusters
                        st.session_state['kmeans_model'] = kmeans
                        st.session_state['X_scaled'] = X_scaled
                        st.session_state['silhouette'] = silhouette
                        st.session_state['davies_bouldin'] = davies_bouldin
                        
                        st.success("✅ Clustering completed!")
            
            with col2:
                if 'clusters' in st.session_state:
                    st.subheader("📊 Clustering Metrics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Silhouette Score",
                            f"{st.session_state['silhouette']:.3f}",
                            help="Higher is better (range: -1 to 1)"
                        )
                    
                    with col2:
                        st.metric(
                            "Davies-Bouldin Index",
                            f"{st.session_state['davies_bouldin']:.3f}",
                            help="Lower is better"
                        )
            
            # Visualizations
            if 'clusters' in st.session_state:
                st.markdown("---")
                st.subheader("📈 Cluster Visualizations")
                
                clusters = st.session_state['clusters']
                df_clustered = df_filtered.copy()
                df_clustered['Cluster'] = clusters
                
                tab1, tab2, tab3 = st.tabs([
                    "2D Projection",
                    "Cluster Profiles",
                    "Business Insights"
                ])
                
                with tab1:
                    # PCA for visualization
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(st.session_state['X_scaled'])
                    
                    pca_df = pd.DataFrame(
                        X_pca,
                        columns=['PC1', 'PC2']
                    )
                    pca_df['Cluster'] = clusters
                    
                    fig = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster',
                        title="Customer Segments (PCA Projection)",
                        color_continuous_scale='Greens',
                        labels={'Cluster': 'Segment'}
                    )
                    fig.update_traces(marker=dict(size=10, opacity=0.7))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"""
                    **Explained Variance:** 
                    PC1: {pca.explained_variance_ratio_[0]:.2%} | 
                    PC2: {pca.explained_variance_ratio_[1]:.2%}
                    """)
                
                with tab2:
                    st.subheader("Cluster Characteristics")
                    
                    # Cluster size
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    
                    fig = px.bar(
                        x=cluster_counts.index,
                        y=cluster_counts.values,
                        title="Segment Sizes",
                        labels={'x': 'Cluster', 'y': 'Number of Customers'},
                        color=cluster_counts.values,
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Profile each cluster
                    st.subheader("Segment Profiles")
                    
                    for cluster_id in sorted(df_clustered['Cluster'].unique()):
                        with st.expander(f"📊 Segment {cluster_id} Profile"):
                            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**Demographics:**")
                                st.write(f"Size: {len(cluster_data)} customers")
                                st.write(f"Dominant Age: {cluster_data['Age_Group'].mode()[0]}")
                                st.write(f"Dominant Gender: {cluster_data['Gender'].mode()[0]}")
                                st.write(f"Main Emirate: {cluster_data['Emirate'].mode()[0]}")
                            
                            with col2:
                                st.write("**Behavior:**")
                                st.write(f"Avg WTP: AED {cluster_data['Willingness_to_Pay_AED'].mean():.2f}")
                                st.write(f"Likely Users: {(cluster_data['Likely_to_Use_ReFillHub']=='Yes').sum()}")
                                st.write(f"Main Purchase Loc: {cluster_data['Purchase_Location'].mode()[0]}")
                            
                            with col3:
                                st.write("**Preferences:**")
                                st.write(f"Eco-friendly: {(cluster_data['Uses_Eco_Products']=='Yes').sum()}")
                                st.write(f"Avg Sustainability Score: {cluster_data['Importance_Sustainability'].mean():.1f}")
                                st.write(f"Main Payment: {cluster_data['Preferred_Payment_Mode'].mode()[0]}")
                
                with tab3:
                    st.subheader("🎯 Actionable Business Insights")
                    
                    for cluster_id in sorted(df_clustered['Cluster'].unique()):
                        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
                        
                        likely_pct = (cluster_data['Likely_to_Use_ReFillHub']=='Yes').sum() / len(cluster_data) * 100
                        avg_wtp = cluster_data['Willingness_to_Pay_AED'].mean()
                        main_location = cluster_data['Purchase_Location'].mode()[0]
                        
                        st.markdown(f"""
                        <div style='background-color: white; padding: 15px; border-left: 5px solid #2c5f2d; margin: 10px 0; border-radius: 5px;'>
                            <h4 style='color: #2c5f2d;'>Segment {cluster_id} Strategy</h4>
                            <ul>
                                <li><b>Target Likelihood:</b> {likely_pct:.1f}% likely to use RefillHub</li>
                                <li><b>Price Point:</b> Average WTP is AED {avg_wtp:.2f}</li>
                                <li><b>Marketing Channel:</b> Focus on {main_location}</li>
                                <li><b>Size:</b> {len(cluster_data)} customers ({len(cluster_data)/len(df_clustered)*100:.1f}% of total)</li>
                            </ul>
                            <p><b>Recommendation:</b> 
                            {"High priority segment - invest in targeted campaigns" if likely_pct > 60 else 
                             "Medium priority - nurture with educational content" if likely_pct > 40 else
                             "Low priority - require significant awareness building"}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # ============================================================================
    # 5. ASSOCIATION RULE MINING
    # ============================================================================
    elif choice == "🔗 Association Rule Mining":
        st.header("🔗 Association Rule Mining: Product Combinations")
        
        st.markdown("""
        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h4 style='color: #2c5f2d;'>🎯 Business Question:</h4>
            <p style='font-size: 16px;'>
                Which products are frequently purchased together? 
                This helps in creating product bundles, optimizing inventory, and cross-selling strategies.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Parse products bought
        if 'Products_Bought' in df.columns:
            # Create transaction data
            transactions = []
            for products in df['Products_Bought'].dropna():
                items = [item.strip() for item in str(products).split(',')]
                transactions.append(items)
            
            st.subheader("🔧 Configuration")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
                min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.1)
                
                if st.button("🚀 Mine Association Rules", key="run_association"):
                    with st.spinner("Mining association rules..."):
                        try:
                            # Encode transactions
                            te = TransactionEncoder()
                            te_ary = te.fit(transactions).transform(transactions)
                            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                            
                            # Apply Apriori algorithm
                            frequent_itemsets = apriori(
                                df_encoded,
                                min_support=min_support,
                                use_colnames=True
                            )
                            
                            if len(frequent_itemsets) > 0:
                                # Generate rules
                                rules = association_rules(
                                    frequent_itemsets,
                                    metric="confidence",
                                    min_threshold=min_confidence
                                )
                                
                                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                                
                                st.session_state['association_rules'] = rules
                                st.session_state['frequent_itemsets'] = frequent_itemsets
                                
                                st.success(f"✅ Found {len(rules)} association rules!")
                            else:
                                st.warning("No frequent itemsets found. Try lowering the support threshold.")
                        
                        except Exception as e:
                            st.error(f"Error during association rule mining: {e}")
            
            with col2:
                if 'association_rules' in st.session_state:
                    rules = st.session_state['association_rules']
                    
                    st.metric("Total Rules Found", len(rules))
                    st.metric("Average Confidence", f"{rules['confidence'].mean():.2%}")
                    st.metric("Average Lift", f"{rules['lift'].mean():.2f}")
            
            # Display results
            if 'association_rules' in st.session_state:
                st.markdown("---")
                st.subheader("📊 Association Rules Results")
                
                rules = st.session_state['association_rules']
                
                tab1, tab2, tab3 = st.tabs([
                    "Top Rules",
                    "Visualizations",
                    "Business Recommendations"
                ])
                
                with tab1:
                    st.subheader("Top Association Rules by Lift")
                    
                    # Sort by lift
                    top_rules = rules.nlargest(20, 'lift')
                    
                    # Display as table
                    display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
                    st.dataframe(
                        top_rules[display_cols].style.background_gradient(
                            subset=['confidence', 'lift'],
                            cmap='Greens'
                        ),
                        use_container_width=True
                    )
                
                with tab2:
                    # Scatter plot
                    fig = px.scatter(
                        rules,
                        x='support',
                        y='confidence',
                        size='lift',
                        color='lift',
                        hover_data=['antecedents', 'consequents'],
                        title="Association Rules: Support vs Confidence (sized by Lift)",
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top product combinations
                    st.subheader("Top Product Combinations by Support")
                    
                    frequent_itemsets = st.session_state['frequent_itemsets']
                    top_itemsets = frequent_itemsets.nlargest(15, 'support')
                    
                    top_itemsets['items'] = top_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
                    
                    fig = px.bar(
                        top_itemsets,
                        x='support',
                        y='items',
                        orientation='h',
                        title="Most Frequent Product Combinations",
                        color='support',
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    st.subheader("🎯 Actionable Recommendations")
                    
                    # Generate recommendations
                    top_rules = rules.nlargest(10, 'lift')
                    
                    for idx, row in top_rules.iterrows():
                        st.markdown(f"""
                        <div style='background-color: white; padding: 15px; border-left: 5px solid #2c5f2d; margin: 10px 0; border-radius: 5px;'>
                            <h4 style='color: #2c5f2d;'>Bundle Opportunity #{idx+1}</h4>
                            <p><b>If customers buy:</b> {row['antecedents']}</p>
                            <p><b>They are likely to buy:</b> {row['consequents']}</p>
                            <ul>
                                <li><b>Confidence:</b> {row['confidence']:.1%} of the time</li>
                                <li><b>Lift:</b> {row['lift']:.2f}x more likely than random</li>
                                <li><b>Support:</b> {row['support']:.1%} of all transactions</li>
                            </ul>
                            <p><b>💡 Action:</b> Create a product bundle or place these items near each other in-store/online</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # ============================================================================
    # 6. WILLINGNESS TO PAY (REGRESSION)
    # ============================================================================
    elif choice == "💰 Willingness to Pay (Regression)":
        st.header("💰 Predicting Willingness to Pay")
        
        st.markdown("""
        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h4 style='color: #2c5f2d;'>🎯 Business Question:</h4>
            <p style='font-size: 16px;'>
                What factors influence how much customers are willing to pay for refill services?
                This helps in pricing strategy and understanding customer value perception.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare data for regression
        target_col = 'Willingness_to_Pay_AED'
        
        if target_col in df.columns:
            # Select features for modeling
            feature_cols = [col for col in df_processed.columns if col.endswith('_encoded')]
            
            X = df_processed[feature_cols].fillna(0)
            y = df_processed[target_col]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Model selection
            st.subheader("🔧 Model Configuration")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                model_choice = st.selectbox(
                    "Select Regression Model",
                    ["Linear Regression", "Ridge Regression", "Lasso Regression", "All Models"]
                )
                
                if st.button("🚀 Train Model(s)", key="train_regression"):
                    with st.spinner("Training models..."):
                        models = {}
                        results = {}
                        
                        if model_choice == "Linear Regression" or model_choice == "All Models":
                            lr = LinearRegression()
                            lr.fit(X_train, y_train)
                            models['Linear Regression'] = lr
                            
                            y_pred = lr.predict(X_test)
                            r2 = r2_score(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            mae = np.mean(np.abs(y_test - y_pred))
                            
                            results['Linear Regression'] = {
                                'r2': r2,
                                'rmse': rmse,
                                'mae': mae,
                                'predictions': y_pred,
                                'model': lr
                            }
                        
                        if model_choice == "Ridge Regression" or model_choice == "All Models":
                            ridge = Ridge(alpha=1.0)
                            ridge.fit(X_train, y_train)
                            models['Ridge Regression'] = ridge
                            
                            y_pred = ridge.predict(X_test)
                            r2 = r2_score(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            mae = np.mean(np.abs(y_test - y_pred))
                            
                            results['Ridge Regression'] = {
                                'r2': r2,
                                'rmse': rmse,
                                'mae': mae,
                                'predictions': y_pred,
                                'model': ridge
                            }
                        
                        if model_choice == "Lasso Regression" or model_choice == "All Models":
                            lasso = Lasso(alpha=1.0)
                            lasso.fit(X_train, y_train)
                            models['Lasso Regression'] = lasso
                            
                            y_pred = lasso.predict(X_test)
                            r2 = r2_score(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            mae = np.mean(np.abs(y_test - y_pred))
                            
                            results['Lasso Regression'] = {
                                'r2': r2,
                                'rmse': rmse,
                                'mae': mae,
                                'predictions': y_pred,
                                'model': lasso
                            }
                        
                        # Store in session state
                        st.session_state['regression_results'] = results
                        st.session_state['regression_models'] = models
                        st.session_state['y_test_reg'] = y_test
                        
                        st.success("✅ Models trained successfully!")
            
            with col2:
                if 'regression_results' in st.session_state:
                    results = st.session_state['regression_results']
                    
                    # Display metrics
                    st.subheader("📊 Model Performance")
                    
                    metrics_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'R² Score': [results[m]['r2'] for m in results.keys()],
                        'RMSE': [results[m]['rmse'] for m in results.keys()],
                        'MAE': [results[m]['mae'] for m in results.keys()]
                    })
                    
                    st.dataframe(
                        metrics_df.style.background_gradient(
                            subset=['R² Score'],
                            cmap='Greens'
                        ).background_gradient(
                            subset=['RMSE', 'MAE'],
                            cmap='Reds_r'
                        ),
                        use_container_width=True
                    )
            
            # Detailed results
            if 'regression_results' in st.session_state:
                st.markdown("---")
                st.subheader("📈 Detailed Results")
                
                results = st.session_state['regression_results']
                y_test_reg = st.session_state['y_test_reg']
                
                tab1, tab2, tab3 = st.tabs([
                    "Prediction Plots",
                    "Residual Analysis",
                    "Feature Impact"
                ])
                
                with tab1:
                    for model_name, result in results.items():
                        st.write(f"**{model_name}**")
                        
                        # Actual vs Predicted
                        plot_df = pd.DataFrame({
                            'Actual': y_test_reg,
                            'Predicted': result['predictions']
                        })
                        
                        fig = px.scatter(
                            plot_df,
                            x='Actual',
                            y='Predicted',
                            title=f"{model_name}: Actual vs Predicted",
                            trendline="ols",
                            color_discrete_sequence=['#2c5f2d']
                        )
                        
                        # Add perfect prediction line
                        min_val = min(plot_df['Actual'].min(), plot_df['Predicted'].min())
                        max_val = max(plot_df['Actual'].max(), plot_df['Predicted'].max())
                        fig.add_trace(
                            go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("---")
                
                with tab2:
                    for model_name, result in results.items():
                        st.write(f"**{model_name} Residuals**")
                        
                        residuals = y_test_reg - result['predictions']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Residual plot
                            fig = px.scatter(
                                x=result['predictions'],
                                y=residuals,
                                title=f"{model_name}: Residual Plot",
                                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                color_discrete_sequence=['#2c5f2d']
                            )
                            fig.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Residual distribution
                            fig = px.histogram(
                                x=residuals,
                                title=f"{model_name}: Residual Distribution",
                                labels={'x': 'Residuals'},
                                color_discrete_sequence=['#2c5f2d']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("---")
                
                with tab3:
                    for model_name, result in results.items():
                        model = result['model']
                        
                        if hasattr(model, 'coef_'):
                            st.write(f"**{model_name} Feature Coefficients:**")
                            
                            coef_df = pd.DataFrame({
                                'Feature': [col.replace('_encoded', '') for col in X.columns],
                                'Coefficient': model.coef_
                            }).sort_values('Coefficient', key=abs, ascending=False).head(15)
                            
                            fig = px.bar(
                                coef_df,
                                x='Coefficient',
                                y='Feature',
                                orientation='h',
                                title=f"Top 15 Feature Impact - {model_name}",
                                color='Coefficient',
                                color_continuous_scale='RdYlGn'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown("---")
    
    # ============================================================================
    # 7. INTERACTIVE INSIGHTS
    # ============================================================================
    elif choice == "📊 Interactive Insights":
        st.header("📊 Interactive Customer Insights")
        
        st.markdown("""
        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h4 style='color: #2c5f2d;'>🔍 Explore Customer Data Interactively</h4>
            <p style='font-size: 16px;'>
                Filter and analyze specific customer segments to understand their behavior and preferences.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create interactive filters
        st.subheader("🔧 Customer Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_age = st.multiselect(
                "Age Group",
                options=df['Age_Group'].unique(),
                default=df['Age_Group'].unique()
            )
            
            selected_gender = st.multiselect(
                "Gender",
                options=df['Gender'].unique(),
                default=df['Gender'].unique()
            )
        
        with col2:
            selected_income = st.multiselect(
                "Income Range",
                options=df['Income'].unique(),
                default=df['Income'].unique()
            )
            
            selected_education = st.multiselect(
                "Education",
                options=df['Education'].unique(),
                default=df['Education'].unique()
            )
        
        with col3:
            selected_emirate = st.multiselect(
                "Emirate",
                options=df['Emirate'].unique(),
                default=df['Emirate'].unique()
            )
            
            uses_eco = st.multiselect(
                "Uses Eco Products",
                options=df['Uses_Eco_Products'].unique(),
                default=df['Uses_Eco_Products'].unique()
            )
        
        # Apply filters
        filtered_data = df[
            (df['Age_Group'].isin(selected_age)) &
            (df['Gender'].isin(selected_gender)) &
            (df['Income'].isin(selected_income)) &
            (df['Education'].isin(selected_education)) &
            (df['Emirate'].isin(selected_emirate)) &
            (df['Uses_Eco_Products'].isin(uses_eco))
        ]
        
        st.markdown("---")
        
        # Display insights
        st.subheader("📊 Filtered Segment Insights")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Segment Size", len(filtered_data))
        
        with col2:
            likely_pct = (filtered_data['Likely_to_Use_ReFillHub']=='Yes').sum() / len(filtered_data) * 100 if len(filtered_data) > 0 else 0
            st.metric("Likely to Use RefillHub", f"{likely_pct:.1f}%")
        
        with col3:
            avg_wtp = filtered_data['Willingness_to_Pay_AED'].mean() if len(filtered_data) > 0 else 0
            st.metric("Avg Willingness to Pay", f"AED {avg_wtp:.2f}")
        
        with col4:
            eco_pct = (filtered_data['Uses_Eco_Products']=='Yes').sum() / len(filtered_data) * 100 if len(filtered_data) > 0 else 0
            st.metric("Eco-Conscious", f"{eco_pct:.1f}%")
        
        # Detailed charts
        if len(filtered_data) > 0:
            st.markdown("---")
            
            tab1, tab2, tab3 = st.tabs([
                "Behavior Patterns",
                "Preferences",
                "Detailed Data"
            ])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Purchase frequency
                    freq_counts = filtered_data['Purchase_Frequency'].value_counts()
                    fig = px.pie(
                        values=freq_counts.values,
                        names=freq_counts.index,
                        title="Purchase Frequency Distribution",
                        color_discrete_sequence=px.colors.sequential.Greens
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Monthly spending
                    fig = px.histogram(
                        filtered_data,
                        x='Monthly_Spending_Range',
                        title="Monthly Spending Distribution",
                        color_discrete_sequence=['#2c5f2d']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Purchase location
                    loc_counts = filtered_data['Purchase_Location'].value_counts()
                    fig = px.bar(
                        x=loc_counts.index,
                        y=loc_counts.values,
                        title="Preferred Purchase Locations",
                        color=loc_counts.values,
                        color_continuous_scale='Greens'
                    )
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Payment mode
                    payment_counts = filtered_data['Preferred_Payment_Mode'].value_counts()
                    fig = px.pie(
                        values=payment_counts.values,
                        names=payment_counts.index,
                        title="Payment Mode Preferences",
                        color_discrete_sequence=px.colors.sequential.Greens
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Container preference
                    container_counts = filtered_data['Container_Type'].value_counts()
                    fig = px.bar(
                        x=container_counts.index,
                        y=container_counts.values,
                        title="Container Type Preferences",
                        color=container_counts.values,
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Refill location
                    refill_loc = filtered_data['Refill_Location'].value_counts()
                    fig = px.pie(
                        values=refill_loc.values,
                        names=refill_loc.index,
                        title="Preferred Refill Locations",
                        color_discrete_sequence=px.colors.sequential.Greens
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Importance scores
                    importance_avg = {
                        'Convenience': filtered_data['Importance_Convenience'].mean(),
                        'Price': filtered_data['Importance_Price'].mean(),
                        'Sustainability': filtered_data['Importance_Sustainability'].mean()
                    }
                    
                    fig = px.bar(
                        x=list(importance_avg.keys()),
                        y=list(importance_avg.values()),
                        title="Average Importance Scores",
                        labels={'x': 'Factor', 'y': 'Score (1-5)'},
                        color=list(importance_avg.values()),
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Discount sensitivity
                    discount_counts = filtered_data['Discount_Switch'].value_counts()
                    fig = px.pie(
                        values=discount_counts.values,
                        names=discount_counts.index,
                        title="Discount Sensitivity",
                        color_discrete_sequence=px.colors.sequential.Greens
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Filtered Customer Data")
                st.dataframe(filtered_data, use_container_width=True)
                
                # Download button
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data as CSV",
                    data=csv,
                    file_name='filtered_customers.csv',
                    mime='text/csv',
                )
        else:
            st.warning("⚠️ No customers match the selected filters. Please adjust your selections.")
    
    # ============================================================================
    # 8. BUSINESS RECOMMENDATIONS
    # ============================================================================
    elif choice == "🎯 Business Recommendations":
        st.header("🎯 Data-Driven Business Recommendations")
        
        st.markdown("""
        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #2c5f2d; text-align: center;'>Executive Summary: RefillHub Market Analysis</h3>
            <p style='font-size: 16px; text-align: center;'>
                Comprehensive insights and actionable recommendations based on customer data analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate key metrics
        total_customers = len(df)
        likely_users = len(df[df['Likely_to_Use_ReFillHub'] == 'Yes'])
        likely_pct = likely_users / total_customers * 100
        avg_wtp = df['Willingness_to_Pay_AED'].mean()
        high_wtp = len(df[df['Willingness_to_Pay_AED'] > 100])
        eco_conscious = len(df[df['Uses_Eco_Products'] == 'Yes'])
        
        # Key findings
        st.subheader("📊 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                <h4 style='color: #2c5f2d;'>1️⃣ Market Opportunity</h4>
                <ul style='font-size: 15px;'>
                    <li><b>{likely_pct:.1f}%</b> of surveyed customers are likely to use RefillHub</li>
                    <li>Potential customer base: <b>{likely_users:,}</b> customers</li>
                    <li>Average willingness to pay: <b>AED {avg_wtp:.2f}</b></li>
                    <li><b>{high_wtp:,}</b> customers willing to pay over AED 100</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                <h4 style='color: #2c5f2d;'>2️⃣ Customer Profile</h4>
                <ul style='font-size: 15px;'>
                    <li>Most common age group: <b>{df['Age_Group'].mode()[0]}</b></li>
                    <li>Primary emirate: <b>{df['Emirate'].mode()[0]}</b></li>
                    <li>Eco-conscious customers: <b>{eco_conscious:,}</b> ({eco_conscious/total_customers*100:.1f}%)</li>
                    <li>Most common purchase frequency: <b>{df['Purchase_Frequency'].mode()[0]}</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                <h4 style='color: #2c5f2d;'>3️⃣ Behavior Insights</h4>
                <ul style='font-size: 15px;'>
                    <li>Top purchase location: <b>{df['Purchase_Location'].mode()[0]}</b></li>
                    <li>Preferred payment: <b>{df['Preferred_Payment_Mode'].mode()[0]}</b></li>
                    <li>Average importance of sustainability: <b>{df['Importance_Sustainability'].mean():.1f}/5</b></li>
                    <li>Plastic ban awareness: <b>{len(df[df['Aware_Plastic_Ban']=='Yes']):,}</b> customers</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                <h4 style='color: #2c5f2d;'>4️⃣ Product Preferences</h4>
                <ul style='font-size: 15px;'>
                    <li>Most wanted container type: <b>{df['Container_Type'].mode()[0]}</b></li>
                    <li>Preferred refill location: <b>{df['Refill_Location'].mode()[0]}</b></li>
                    <li>Preferred packaging: <b>{df['Preferred_Packaging'].mode()[0]}</b></li>
                    <li>Average waste reduction score: <b>{df['Reduce_Waste_Score'].mean():.1f}/5</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Strategic recommendations
        st.markdown("---")
        st.subheader("🎯 Strategic Recommendations")
        
        recommendations = [
            {
                "title": "1. Target High-Value Segments",
                "priority": "High",
                "rationale": f"Focus on customers with WTP > AED 100 ({high_wtp} customers) who show strong purchase intent",
                "actions": [
                    "Create premium product bundles for high-WTP segments",
                    "Implement personalized pricing strategies",
                    "Offer loyalty programs for repeat customers",
                    "Develop targeted marketing campaigns for each segment"
                ],
                "expected_impact": "30-40% increase in revenue per customer"
            },
            {
                "title": "2. Optimize Location Strategy",
                "priority": "High",
                "rationale": f"Majority prefer {df['Purchase_Location'].mode()[0]} - align refill stations accordingly",
                "actions": [
                    f"Establish refill hubs in top 3 emirates: {', '.join(df['Emirate'].value_counts().head(3).index.tolist())}",
                    f"Partner with {df['Purchase_Location'].mode()[0].lower()}s for in-store refill stations",
                    "Create mobile refill units for underserved areas",
                    "Implement online ordering with home delivery option"
                ],
                "expected_impact": "50-60% improvement in customer accessibility"
            },
            {
                "title": "3. Sustainability Marketing",
                "priority": "High",
                "rationale": f"{eco_conscious/total_customers*100:.1f}% are eco-conscious - leverage sustainability messaging",
                "actions": [
                    "Highlight environmental impact metrics (plastic saved, carbon footprint)",
                    "Create educational content about plastic waste reduction",
                    "Partner with environmental organizations for credibility",
                    "Implement a waste tracking app for customers"
                ],
                "expected_impact": "25-35% increase in brand loyalty"
            },
            {
                "title": "4. Product Bundle Strategy",
                "priority": "Medium",
                "rationale": "Association rule mining reveals strong product combinations",
                "actions": [
                    "Create 'starter packs' with frequently bought-together items",
                    "Offer combo discounts (10-20% based on discount sensitivity)",
                    "Design refill stations to display complementary products together",
                    "Implement 'customers also bought' recommendations"
                ],
                "expected_impact": "20-30% increase in average basket size"
            },
            {
                "title": "5. Digital Payment Integration",
                "priority": "Medium",
                "rationale": f"Majority prefer {df['Preferred_Payment_Mode'].mode()[0]} - ensure seamless experience",
                "actions": [
                    "Integrate multiple payment options (Card, Mobile App, Subscription)",
                    "Offer auto-refill subscription services",
                    "Implement QR code-based contactless payments",
                    "Create a cashback/rewards program for digital payments"
                ],
                "expected_impact": "15-25% increase in transaction conversion"
            },
            {
                "title": "6. Customer Education Program",
                "priority": "Medium",
                "rationale": f"{len(df[df['Aware_Plastic_Ban']!='Yes']):,} customers unaware of plastic ban - opportunity for education",
                "actions": [
                    "Launch awareness campaigns about plastic regulations",
                    "Create in-store demonstrations of refill process",
                    "Develop video tutorials and social media content",
                    "Partner with influencers for eco-friendly lifestyle promotion"
                ],
                "expected_impact": "40-50% increase in brand awareness"
            }
        ]
        
        for rec in recommendations:
            priority_color = "#e74c3c" if rec["priority"] == "High" else "#f39c12"
            
            st.markdown(f"""
            <div style='background-color: white; padding: 20px; border-left: 5px solid {priority_color}; margin: 15px 0; border-radius: 5px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h3 style='color: #2c5f2d; margin: 0;'>{rec['title']}</h3>
                    <span style='background-color: {priority_color}; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;'>
                        {rec['priority']} Priority
                    </span>
                </div>
                <p style='font-size: 16px; color: #666; margin-top: 10px;'><b>Rationale:</b> {rec['rationale']}</p>
                <p style='font-size: 16px; margin-top: 10px;'><b>Recommended Actions:</b></p>
                <ul style='font-size: 15px;'>
                    {"".join([f"<li>{action}</li>" for action in rec['actions']])}
                </ul>
                <p style='font-size: 16px; color: #2c5f2d; margin-top: 10px;'><b>📈 Expected Impact:</b> {rec['expected_impact']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Implementation roadmap
        st.markdown("---")
        st.subheader("🗓️ Implementation Roadmap")
        
        roadmap = {
            "Phase 1 (Months 1-3)": [
                "Launch pilot program in Dubai with 5-10 refill stations",
                "Implement digital payment and mobile app",
                "Start customer education campaigns",
                "Begin data collection and tracking"
            ],
            "Phase 2 (Months 4-6)": [
                "Expand to Abu Dhabi and Sharjah",
                "Launch product bundle offerings",
                "Implement loyalty program",
                "Scale marketing efforts"
            ],
            "Phase 3 (Months 7-12)": [
                "Cover all major emirates",
                "Launch subscription services",
                "Implement advanced personalization",
                "Evaluate and optimize based on data"
            ]
        }
        
        for phase, tasks in roadmap.items():
            st.markdown(f"""
            <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <h4 style='color: #2c5f2d;'>{phase}</h4>
                <ul style='font-size: 15px;'>
                    {"".join([f"<li>{task}</li>" for task in tasks])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Success metrics
        st.markdown("---")
        st.subheader("📊 Success Metrics to Track")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px;'>
                <h4 style='color: #2c5f2d;'>Customer Metrics</h4>
                <ul style='font-size: 14px;'>
                    <li>Customer acquisition rate</li>
                    <li>Customer retention rate</li>
                    <li>Net Promoter Score (NPS)</li>
                    <li>Customer Lifetime Value</li>
                    <li>Repeat purchase rate</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px;'>
                <h4 style='color: #2c5f2d;'>Business Metrics</h4>
                <ul style='font-size: 14px;'>
                    <li>Revenue per station</li>
                    <li>Average transaction value</li>
                    <li>Refill frequency per customer</li>
                    <li>Product bundle adoption rate</li>
                    <li>Subscription conversion rate</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px;'>
                <h4 style='color: #2c5f2d;'>Impact Metrics</h4>
                <ul style='font-size: 14px;'>
                    <li>Plastic waste reduced (kg)</li>
                    <li>Carbon footprint saved</li>
                    <li>Customer awareness score</li>
                    <li>Market share growth</li>
                    <li>Brand sentiment score</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Download report
        st.markdown("---")
        st.subheader("📥 Export Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Complete Dataset",
                data=csv,
                file_name='refillhub_complete_data.csv',
                mime='text/csv',
            )
        
        with col2:
            # Create summary report
            summary_data = {
                'Metric': [
                    'Total Respondents',
                    'Likely Users',
                    'Likely Users %',
                    'Avg Willingness to Pay (AED)',
                    'High WTP Customers (>100 AED)',
                    'Eco-Conscious Customers',
                    'Plastic Ban Aware'
                ],
                'Value': [
                    total_customers,
                    likely_users,
                    f"{likely_pct:.1f}%",
                    f"{avg_wtp:.2f}",
                    high_wtp,
                    eco_conscious,
                    len(df[df['Aware_Plastic_Ban']=='Yes'])
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="📋 Download Executive Summary",
                data=summary_csv,
                file_name='executive_summary.csv',
                mime='text/csv',
            )

else:
    st.error("❌ Unable to load data. Please ensure 'RefillHub_SyntheticSurvey.csv' is in the same directory as this script.")
    st.info("💡 Tip: Place your CSV file in the same folder and refresh the page.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <p style='font-size: 14px;'>
        🌱 RefillHub Market Analysis Dashboard | Powered by Data Science & Machine Learning<br>
        Built with Streamlit, Scikit-learn, Plotly, and ❤️
    </p>
    <p style='font-size: 12px; color: #999;'>
        © 2024 RefillHub Analytics | Data-Driven Sustainability Solutions
    </p>
</div>
""", unsafe_allow_html=True)
