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
    knn = NearestNeighbors(n_neighbors=min(20, len(df)), algorithm='auto')
    knn.fit(X_scaled)
    
    # Function to recommend places with country priority
    def recommend_place(country, category, season, n_recommendations=5):
        try:
            # First, try to get recommendations from the selected country
            country_filtered = df[df['Country'] == country]
            
            if category != "Any":
                country_filtered = country_filtered[country_filtered['Category'] == category]
            if season != "Any":
                country_filtered = country_filtered[country_filtered['Season'] == season]
            
            recommendations = []
            
            # Priority 1: Exact matches from selected country
            if not country_filtered.empty:
                country_recs = country_filtered.head(n_recommendations)
                for idx, row in country_recs.iterrows():
                    rec = row.copy()
                    rec['Priority'] = 1
                    rec['Match_Type'] = f"Exact match in {country}"
                    recommendations.append(rec)
            
            # If we need more recommendations, use KNN for similar places
            if len(recommendations) < n_recommendations:
                try:
                    # Encode user input
                    country_encoded = country_encoder.transform([country])[0]
                    
                    # Handle category encoding
                    if category == "Any":
                        # Use all categories for broader search
                        category_encoded = category_encoder.transform([original_categories[0]])[0]
                    else:
                        category_encoded = category_encoder.transform([category])[0]
                    
                    # Handle season encoding
                    if season == "Any":
                        # Use all seasons for broader search
                        season_encoded = season_encoder.transform([original_seasons[0]])[0]
                    else:
                        season_encoded = season_encoder.transform([season])[0]
                    
                    # Create input array
                    user_input = np.array([[country_encoded, category_encoded, season_encoded]])
                    
                    # Scale the input
                    user_input_scaled = scaler.transform(user_input)
                    
                    # Find nearest neighbors
                    distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=min(20, len(df)))
                    
                    # Get similar places
                    similar_places = df.iloc[indices[0]].copy()
                    similar_places['Distance'] = distances[0]
                    
                    # Filter out places already in recommendations
                    existing_places = [rec['Place'] for rec in recommendations]
                    similar_places = similar_places[~similar_places['Place'].isin(existing_places)]
                    
                    # Add similar places with lower priority
                    remaining_slots = n_recommendations - len(recommendations)
                    for idx, row in similar_places.head(remaining_slots).iterrows():
                        rec = row.copy()
                        rec['Priority'] = 2
                        if row['Country'] == country:
                            rec['Match_Type'] = f"Similar in {country}"
                        else:
                            rec['Match_Type'] = f"Similar destination"
                        recommendations.append(rec)
                
                except ValueError as e:
                    st.warning(f"KNN search failed: {str(e)}")
            
            # Convert to DataFrame
            if recommendations:
                result_df = pd.DataFrame(recommendations)
                # Sort by priority first, then by rating if available
                if 'Rating' in result_df.columns:
                    result_df = result_df.sort_values(['Priority', 'Rating'], ascending=[True, False])
                else:
                    result_df = result_df.sort_values('Priority')
                
                return result_df.head(n_recommendations)
            else:
                return pd.DataFrame()
        
        except Exception as e:
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
        category_options = ["Any"] + sorted(original_categories)
        category = st.selectbox("🎯 Select Category", category_options)
    
    with col3:
        season_options = ["Any"] + sorted(original_seasons)
        season = st.selectbox("🗓️ Select Season", season_options)
    
    # Number of recommendations slider
    n_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=min(15, len(df)), value=8)
    
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
                        # Priority badge
                        if row.get('Priority', 2) == 1:
                            st.markdown("🏅 **TOP PRIORITY**")
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**🏛️ {row['Place']}**")
                            st.markdown(f"📍 Country: {row['Country']}")
                            st.markdown(f"🎨 Category: {row['Category']}")
                            st.markdown(f"🌤️ Season: {row['Season']}")
                            if 'Rating' in row and pd.notna(row['Rating']):
                                st.markdown(f"⭐ Rating: {row['Rating']}")
                            if 'Match_Type' in row:
                                st.markdown(f"🎯 {row['Match_Type']}")
                        
                        with col2:
                            if 'Distance' in row and pd.notna(row['Distance']):
                                similarity = max(0, 100 - (row['Distance'] * 20))
                                st.metric("Similarity", f"{similarity:.1f}%")
                            elif row.get('Priority', 2) == 1:
                                st.metric("Match", "100%")
                        
                        with col3:
                            # Google Maps link
                            if 'Google_Maps_Link' in row and pd.notna(row['Google_Maps_Link']):
                                st.markdown(f"[🗺️ View Map]({row['Google_Maps_Link']})")
                            else:
                                # Generate Google Maps link if not available
                                place_query = f"{row['Place']} {row['Country']}".replace(' ', '+')
                                maps_link = f"https://maps.google.com/maps?q={place_query}"
                                st.markdown(f"[🗺️ View Map]({maps_link})")
                        
                        st.divider()
                
                # Show detailed table
                with st.expander("📊 Detailed Recommendations Table"):
                    # Prepare display dataframe
                    display_df = recommended.copy()
                    
                    # Add clickable maps column
                    if 'Google_Maps_Link' in display_df.columns:
                        display_df['Maps'] = display_df['Google_Maps_Link'].apply(
                            lambda x: x if pd.notna(x) else f"https://maps.google.com/maps?q={display_df.loc[display_df['Google_Maps_Link'] == x, 'Place'].iloc[0].replace(' ', '+')}+{display_df.loc[display_df['Google_Maps_Link'] == x, 'Country'].iloc[0].replace(' ', '+')}"
                        )
                    else:
                        display_df['Maps'] = display_df.apply(
                            lambda row: f"https://maps.google.com/maps?q={row['Place'].replace(' ', '+')}+{row['Country'].replace(' ', '+')}", 
                            axis=1
                        )
                    
                    # Select columns to display
                    cols_to_show = ['Place', 'Country', 'Category', 'Season']
                    if 'Rating' in display_df.columns:
                        cols_to_show.append('Rating')
                    if 'Match_Type' in display_df.columns:
                        cols_to_show.append('Match_Type')
                    cols_to_show.append('Maps')
                    
                    display_df = display_df[cols_to_show].reset_index(drop=True)
                    display_df.index = display_df.index + 1
                    st.dataframe(display_df, use_container_width=True)
            
            else:
                st.warning(f"No recommendations found for {country}. Try selecting 'Any' for category or season to get broader results.")
    
    # Show country-specific statistics
    st.subheader(f"📊 Statistics for {country}")
    country_data = df[df['Country'] == country]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Places Available", len(country_data))
    with col2:
        st.metric("Categories", len(country_data['Category'].unique()))
    with col3:
        if 'Rating' in country_data.columns:
            avg_rating = country_data['Rating'].mean()
            st.metric("Avg Rating", f"{avg_rating:.1f}" if not pd.isna(avg_rating) else "N/A")
        else:
            st.metric("Seasons", len(country_data['Season'].unique()))
    
    # Show category distribution for selected country
    if not country_data.empty:
        with st.expander(f"📈 {country} Destination Breakdown"):
            col1, col2 = st.columns(2)
            
            with col1:
                category_counts = country_data['Category'].value_counts()
                st.bar_chart(category_counts, height=300)
                st.caption(f"Categories in {country}")
            
            with col2:
                season_counts = country_data['Season'].value_counts()
                st.bar_chart(season_counts, height=300)
                st.caption(f"Best Seasons for {country}")
    
    # Show overall dataset statistics
    with st.expander("🌍 Overall Dataset Statistics"):
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
        st.subheader("Global Data Distribution")
        
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
    - Place: Name of the tourist destination
    - Country: Name of the country
    - Category: Type of tourism (e.g., Nature, Culture, Culinary)
    - Season: Best season to visit (Winter, Spring, Summer, Autumn)
    - Rating: Rating score (optional)
    - Google_Maps_Link: Direct link to Google Maps (optional)
    """)
    
    # Show sample CSV format
    st.subheader("Sample CSV Format:")
    sample_data = {
        'Place': ['Bali Beach', 'Tokyo Tower', 'Eiffel Tower'],
        'Country': ['Indonesia', 'Japan', 'France'],
        'Category': ['Nature', 'Culture', 'Culture'],
        'Season': ['Summer', 'Winter', 'Spring'],
        'Rating': [4.8, 4.7, 4.6],
        'Google_Maps_Link': [
            'https://maps.google.com/maps?q=Kuta+Beach+Bali+Indonesia',
            'https://maps.google.com/maps?q=Tokyo+Tower+Japan',
            'https://maps.google.com/maps?q=Eiffel+Tower+Paris+France'
        ]
    }
    st.dataframe(pd.DataFrame(sample_data))