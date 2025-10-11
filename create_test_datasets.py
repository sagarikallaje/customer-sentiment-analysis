"""
Test script to verify large file upload functionality
This script creates a test CSV file to verify the upload limits work correctly
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def create_large_test_dataset(num_rows=50000):
    """Create a large test dataset to verify upload limits"""
    
    print(f"Creating test dataset with {num_rows:,} rows...")
    
    # Generate test data
    reviews = []
    
    # Sample review texts
    positive_texts = [
        "Excellent product! Highly recommend.",
        "Great quality and fast shipping.",
        "Perfect! Exactly what I needed.",
        "Amazing product! Works perfectly.",
        "Outstanding quality! Very satisfied."
    ]
    
    negative_texts = [
        "Poor quality. Not worth the money.",
        "Terrible product! Broke immediately.",
        "Disappointed with this purchase.",
        "Waste of money. Don't recommend.",
        "Poor construction and materials."
    ]
    
    neutral_texts = [
        "Average product. Does the job.",
        "Okay quality for the price.",
        "Standard product. Nothing special.",
        "It works as expected.",
        "Fair quality. Meets basic needs."
    ]
    
    for i in range(num_rows):
        # Generate rating with realistic distribution
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.35, 0.35])
        
        # Select review text based on rating
        if rating >= 4:
            review_text = random.choice(positive_texts)
            sentiment = "Positive"
        elif rating == 3:
            review_text = random.choice(neutral_texts)
            sentiment = "Neutral"
        else:
            review_text = random.choice(negative_texts)
            sentiment = "Negative"
        
        # Generate other fields
        product_names = ["Test Product A", "Test Product B", "Test Product C", "Test Product D", "Test Product E"]
        categories = ["Electronics", "Fashion", "Home & Garden", "Books", "Sports"]
        brands = ["Brand A", "Brand B", "Brand C", "Brand D", "Brand E"]
        
        # Generate review date (last year)
        start_date = datetime.now() - timedelta(days=365)
        random_days = random.randint(0, 365)
        review_date = start_date + timedelta(days=random_days)
        
        review = {
            'Review_ID': f"TEST_{str(i).zfill(6)}",
            'Review_Text': review_text,
            'Review_Title': f"Review {i+1}",
            'Rating': rating,
            'Review_Date': review_date.strftime('%Y-%m-%d'),
            'Verified_Purchase': random.choice([True, False]),
            'Product_ID': f"PROD_{str(i % 100).zfill(3)}",
            'Product_Name': random.choice(product_names),
            'Category': random.choice(categories),
            'Subcategory': f"Subcategory {random.randint(1, 5)}",
            'Brand': random.choice(brands),
            'Price': round(random.uniform(10, 1000), 2),
            'Features': f"Feature {random.randint(1, 10)}, Feature {random.randint(1, 10)}",
            'Customer_ID': f"CUST_{str(i % 1000).zfill(4)}",
            'Customer_Name': f"Customer_{i+1}",
            'Age_Group': random.choice(["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]),
            'Location': random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
            'Purchase_History': random.choice(["Frequent", "Regular", "Occasional", "New Customer"]),
            'Helpful_Votes': random.randint(0, 50),
            'Comments_Count': random.randint(0, 10),
            'Media_URLs': "",
            'Sentiment_Label': sentiment,
            'Aspect_Sentiments': '{"quality": "' + sentiment.lower() + '", "value": "' + sentiment.lower() + '"}'
        }
        
        reviews.append(review)
    
    df = pd.DataFrame(reviews)
    
    # Calculate file size
    csv_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    print(f"Dataset created successfully!")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Estimated CSV size: {csv_size:.2f} MB")
    
    return df

if __name__ == "__main__":
    # Create different sized test datasets
    test_sizes = [1000, 10000, 50000]  # Small, medium, large
    
    for size in test_sizes:
        print(f"\n{'='*50}")
        print(f"Creating test dataset with {size:,} rows")
        print(f"{'='*50}")
        
        test_data = create_large_test_dataset(size)
        
        # Save to CSV
        filename = f"data/test_dataset_{size}.csv"
        test_data.to_csv(filename, index=False)
        
        print(f"Test dataset saved to: {filename}")
        
        # Show sample
        print("\nSample data:")
        print(test_data.head(3))
        
        print(f"\nRating distribution:")
        rating_dist = test_data['Rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            print(f"  {rating} stars: {count:,} ({count/len(test_data)*100:.1f}%)")
    
    print(f"\n{'='*50}")
    print("Test datasets created successfully!")
    print("You can now test the upload functionality with these files.")
    print("The app supports files up to 200MB with progress indicators.")
    print(f"{'='*50}")
