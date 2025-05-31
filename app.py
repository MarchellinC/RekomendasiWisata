import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('tourism_dataset.csv')
        # Remove extra spaces in column names
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("Dataset file 'tourism_dataset.csv' not found. Please upload the file.")
        return None

df = load_data()

if df is not None:
    # Display column names for verification
    st.sidebar.write("Dataset Columns:", df.columns.tolist())
    
    # Display first few rows
    with st.expander("View Dataset Sample"):
        st.write(df.head())
    
    # Check if required columns exist
    required_columns = ['Country', 'Category', 'Season', 'Place']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()
    
    # Create separate label encoders for each column
    country_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    season_encoder = LabelEncoder()
    
    # Store original values before encoding
    original_countries = df['Country'].unique()
    original_categories = df['Category'].unique()
    original_seasons = df['Season'].unique()
    
    # Encode categorical variables
    df_encoded = df.copy()
    df_encoded['Country_encoded'] = country_encoder.fit_transform(df['Country'])
    df_encoded['Category_encoded'] = category_encoder.fit_transform(df['Category'])
    df_encoded['Season_encoded'] = season_encoder.fit_transform(df['Season'])
    
    # Features for training
    X = df_encoded[['Country_encoded', 'Category_encoded', 'Season_encoded']]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train KNN model
    knn = NearestNeighbors(n_neighbors=min(5, len(df)), algorithm='auto')
    knn.fit(X_scaled)
    
    # Function to recommend places
    def recommend_place(country, category, season, n_recommendations=5):
        try:
            # Encode user input
            country_encoded = country_encoder.transform([country])[0]
            category_encoded = category_encoder.transform([category])[0]
            season_encoded = season_encoder.transform([season])[0]
            
            # Create input array
            user_input = np.array([[country_encoded, category_encoded, season_encoded]])
            
            # Scale the input
            user_input_scaled = scaler.transform(user_input)
            
            # Find nearest neighbors
            distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=min(n_recommendations, len(df)))
            
            # Get recommended places
            recommended_places = df.iloc[indices[0]].copy()
            recommended_places['Distance'] = distances[0]
            
            return recommended_places[['Place', 'Country', 'Category', 'Season', 'Distance']]
        
        except ValueError as e:
            st.error(f"Error in recommendation: {str(e)}")
            return pd.DataFrame()
    
    # Streamlit UI
    st.title("🌍 Tourism Recommendation System")
    st.markdown("Find the perfect travel destination based on your preferences!")
    
    # Create three columns for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        country = st.selectbox("🏳️ Select Country", sorted(original_countries))
    
    with col2:
        category = st.selectbox("🎯 Select Category", sorted(original_categories))
    
    with col3:
        season = st.selectbox("🗓️ Select Season", sorted(original_seasons))
    
    # Number of recommendations slider
    n_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=min(10, len(df)), value=5)
    
    # Recommendation button
    if st.button("🔍 Get Recommendations", type="primary"):
        with st.spinner("Finding the best destinations for you..."):
            recommended = recommend_place(country, category, season, n_recommendations)
            
            if not recommended.empty:
                st.success(f"Found {len(recommended)} recommendations!")
                
                # Display recommendations in a nice format
                st.subheader("🏆 Recommended Places")
                
                for idx, row in recommended.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**🏛️ {row['Place']}**")
                            st.markdown(f"📍 Country: {row['Country']}")
                            st.markdown(f"🎨 Category: {row['Category']}")
                            st.markdown(f"🌤️ Season: {row['Season']}")
                        
                        with col2:
                            similarity = max(0, 100 - (row['Distance'] * 20))  # Convert distance to similarity percentage
                            st.metric("Similarity", f"{similarity:.1f}%")
                        
                        st.divider()
                
                # Show detailed table
                with st.expander("📊 Detailed Recommendations Table"):
                    # Remove distance column for cleaner display
                    display_df = recommended.drop('Distance', axis=1).reset_index(drop=True)
                    display_df.index = display_df.index + 1  # Start index from 1
                    st.dataframe(display_df, use_container_width=True)
            
            else:
                st.warning("No recommendations found. Please try different criteria.")
    
    # Show dataset statistics
    with st.expander("📈 Dataset Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Places", len(df))
        
        with col2:
            st.metric("Countries", len(original_countries))
        
        with col3:
            st.metric("Categories", len(original_categories))
        
        with col4:
            st.metric("Seasons", len(original_seasons))
        
        # Distribution charts
        st.subheader("Data Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            country_counts = df['Country'].value_counts()
            st.bar_chart(country_counts, height=300)
            st.caption("Places by Country")
        
        with col2:
            category_counts = df['Category'].value_counts()
            st.bar_chart(category_counts, height=300)
            st.caption("Places by Category")

else:
    st.info("Please ensure 'tourism_dataset.csv' is in the same directory as this script.")
    st.markdown("""
    **Expected CSV format:**
    - Country: Name of the country
    - Category: Type of tourism (e.g., Adventure, Cultural, Beach, etc.)
    - Season: Best season to visit (Winter, Spring, Summer, Autumn)
    - Place: Name of the tourist destination
    """)