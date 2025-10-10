"""
Sample Data Generator for Amazon Reviews
Creates a realistic dataset for the Streamlit sentiment analysis app
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List


def create_sample_amazon_reviews(num_reviews: int = 1000) -> pd.DataFrame:
    """
    Create a realistic Amazon review dataset
    
    Args:
        num_reviews (int): Number of reviews to generate
        
    Returns:
        pd.DataFrame: Generated review dataset
    """
    np.random.seed(42)
    random.seed(42)
    
    # Sample positive and negative review texts
    positive_reviews = [
        "Absolutely love this product! Great quality and fast shipping.",
        "Excellent value for money. Highly recommend to everyone.",
        "Perfect! Exactly what I was looking for. Will buy again.",
        "Amazing product! Works perfectly and looks great.",
        "Outstanding quality! Exceeded my expectations completely.",
        "Fantastic purchase! Great customer service too.",
        "Love it! Perfect fit and excellent material quality.",
        "Wonderful product! Fast delivery and great packaging.",
        "Excellent! Works exactly as described. Very satisfied.",
        "Amazing quality! Great price and perfect functionality.",
        "Perfect product! Highly recommend to others.",
        "Excellent value! Great quality and fast shipping.",
        "Love this! Exactly what I needed. Perfect condition.",
        "Outstanding! Great product and excellent service.",
        "Fantastic! Works perfectly and looks amazing.",
        "Amazing! Great quality and perfect fit.",
        "Excellent product! Highly satisfied with purchase.",
        "Perfect! Great value and excellent quality.",
        "Wonderful! Exactly as described and fast delivery.",
        "Outstanding quality! Highly recommend this product.",
        "This is exactly what I needed! Perfect quality and fast delivery.",
        "Amazing product! Works great and looks fantastic.",
        "Excellent purchase! Great value for the money.",
        "Love it! Perfect fit and excellent material.",
        "Outstanding! Great product and excellent service.",
        "Fantastic quality! Highly recommend to everyone.",
        "Perfect! Exactly as described and fast shipping.",
        "Amazing! Great quality and perfect functionality.",
        "Excellent! Works perfectly and looks great.",
        "Outstanding product! Highly satisfied with purchase."
    ]
    
    negative_reviews = [
        "Terrible product! Poor quality and not as described.",
        "Waste of money. Broke after just one week of use.",
        "Disappointed with this purchase. Quality is very poor.",
        "Not worth the price. Cheaply made and doesn't work well.",
        "Horrible experience! Product arrived damaged and broken.",
        "Very disappointed. Poor quality and bad customer service.",
        "Regret buying this. Doesn't work as advertised at all.",
        "Terrible quality! Broke immediately after opening.",
        "Waste of time and money. Product is completely useless.",
        "Poor quality! Not worth the money spent on this.",
        "Disappointed! Product doesn't match description at all.",
        "Terrible! Broke after first use. Very poor quality.",
        "Not recommended! Cheap material and poor construction.",
        "Horrible product! Doesn't work and poor customer service.",
        "Very poor quality! Waste of money completely.",
        "Disappointed with purchase. Product is defective.",
        "Terrible experience! Poor quality and bad packaging.",
        "Not as described! Poor quality and doesn't work.",
        "Waste of money! Product broke immediately.",
        "Poor quality! Very disappointed with this purchase.",
        "This product is terrible! Poor quality and doesn't work.",
        "Waste of money! Broke after just a few days.",
        "Very disappointed! Product doesn't match description.",
        "Terrible quality! Not worth the price at all.",
        "Horrible experience! Product arrived broken.",
        "Poor quality! Regret buying this product.",
        "Not recommended! Cheap material and poor construction.",
        "Disappointed! Product doesn't work as advertised.",
        "Terrible! Waste of money and time.",
        "Poor quality! Very unsatisfied with this purchase."
    ]
    
    # Product categories
    categories = [
        "Electronics", "Books", "Clothing", "Home & Garden", 
        "Sports & Outdoors", "Beauty & Personal Care", "Toys & Games",
        "Automotive", "Health & Household", "Office Products",
        "Kitchen & Dining", "Pet Supplies", "Baby Products",
        "Jewelry", "Shoes", "Watches", "Musical Instruments"
    ]
    
    # Product names
    product_names = [
        "Wireless Bluetooth Headphones", "Coffee Maker Deluxe", 
        "Smartphone Case Premium", "Yoga Mat Professional",
        "LED Desk Lamp", "Bluetooth Speaker", "Kitchen Knife Set",
        "Running Shoes Athletic", "Laptop Stand Adjustable",
        "Water Bottle Insulated", "Phone Charger Fast", "Book Light LED",
        "Travel Mug Stainless", "Mouse Pad Gaming", "Cable Organizer",
        "Desk Organizer Bamboo", "Phone Mount Car", "Bluetooth Earbuds",
        "Power Bank Portable", "Screen Protector Tempered",
        "Wireless Mouse", "Mechanical Keyboard", "Gaming Chair",
        "Stand Mixer", "Air Fryer", "Instant Pot", "Coffee Grinder",
        "Blender High Speed", "Food Processor", "Toaster Oven"
    ]
    
    reviews = []
    
    for i in range(num_reviews):
        # Generate rating (1-5 stars) with realistic distribution
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.35, 0.35])
        
        # Select review text based on rating
        if rating >= 4:
            review_text = random.choice(positive_reviews)
        else:
            review_text = random.choice(negative_reviews)
        
        # Add some variation to make reviews more realistic
        review_text = add_variation(review_text, rating)
        
        # Generate other fields
        product_name = random.choice(product_names)
        category = random.choice(categories)
        
        # Generate review date (last 2 years)
        start_date = datetime.now() - timedelta(days=730)
        random_days = random.randint(0, 730)
        review_date = start_date + timedelta(days=random_days)
        
        reviews.append({
            'reviewText': review_text,
            'overall': rating,
            'category': category,
            'productName': product_name,
            'reviewDate': review_date.strftime('%Y-%m-%d'),
            'reviewerID': f'R{str(i).zfill(6)}',
            'helpful': random.randint(0, 50)
        })
    
    df = pd.DataFrame(reviews)
    
    # Print summary
    print(f"Generated {len(df)} Amazon reviews")
    print(f"Rating distribution:")
    rating_counts = df['overall'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        print(f"  {rating} stars: {count} reviews ({count/len(df)*100:.1f}%)")
    
    print(f"Category distribution:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.head(5).items():
        print(f"  {category}: {count} reviews")
    
    return df


def add_variation(text: str, rating: int) -> str:
    """
    Add variation to review text to make it more realistic
    
    Args:
        text (str): Original review text
        rating (int): Rating of the review
        
    Returns:
        str: Modified review text
    """
    # Add some random elements
    variations = [
        f"I bought this {random.choice(['last week', 'yesterday', 'a month ago', 'recently'])} and ",
        f"After using this for {random.choice(['a few days', 'a week', 'several weeks', 'a month'])}, I can say that ",
        f"This is my {random.choice(['first', 'second', 'third'])}+ purchase and ",
        f"I've been using this product for {random.choice(['a while', 'some time', 'a few months'])}, and ",
        ""
    ]
    
    prefix = random.choice(variations)
    
    # Add some additional comments
    additional_comments = [
        " The packaging was great too!",
        " Shipping was very fast.",
        " Customer service was helpful.",
        " Great value for the price.",
        " Would definitely recommend!",
        " Very happy with this purchase.",
        " Not what I expected.",
        " Could be better quality.",
        " Shipping was slow.",
        " Customer service was unhelpful.",
        " Overpriced for the quality.",
        " Would not recommend.",
        " The material feels cheap.",
        " Much better than expected!",
        " Perfect for my needs.",
        " Exactly as advertised.",
        " Great addition to my collection.",
        " Disappointed with the quality.",
        " Works better than I thought.",
        " Not worth the money."
    ]
    
    suffix = random.choice(additional_comments) if random.random() > 0.3 else ""
    
    return prefix + text + suffix


if __name__ == "__main__":
    # Create sample dataset
    sample_data = create_sample_amazon_reviews(1000)
    
    # Save to CSV
    sample_data.to_csv('data/amazon_reviews.csv', index=False)
    print(f"\nDataset saved to 'data/amazon_reviews.csv'")
    
    # Show sample
    print("\nSample data preview:")
    print(sample_data.head())
