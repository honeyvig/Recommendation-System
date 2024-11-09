# Recommendation-System
create a recommendation system based on our provided data. The ideal candidate will have experience in machine learning and data analysis to help us deliver personalized suggestions to our users. You will be responsible for analyzing the dataset, designing the recommendation algorithms, and testing the system to ensure accuracy and effectiveness. If you have a passion for AI and have successfully built similar systems in the past, we would love to hear from you!
----
Creating a recommendation system involves analyzing user data to suggest relevant items to users based on either their past behavior or the behavior of similar users (collaborative filtering) or based on item characteristics (content-based filtering). You can also use hybrid methods that combine both techniques.

I'll walk you through creating a basic Collaborative Filtering recommendation system using Python and libraries such as pandas, scikit-learn, and surprise. This system will allow you to analyze user-item interaction data (like user ratings or clicks) and deliver personalized recommendations.

Here’s a step-by-step guide, assuming we are working with a user-item interaction dataset:
Step 1: Install Required Libraries

First, ensure that the necessary libraries are installed:

pip install pandas numpy scikit-learn surprise

Step 2: Sample Data (User-Item Interaction)

Let's assume the dataset is in the form of user-item interaction data, such as user ratings for products or movies. Here's an example dataset:
UserId	ItemId	Rating
1	101	4
1	102	5
2	101	3
2	103	4
3	102	2
3	103	5
Step 3: Collaborative Filtering with Surprise Library

We'll use the Surprise library, which is specifically designed for building recommendation systems. It supports algorithms like SVD (Singular Value Decomposition), KNN-based Collaborative Filtering, and others.
1. Prepare the Data

import pandas as pd
from surprise import Dataset, Reader

# Sample data (you can replace this with your dataset)
data = {
    'UserId': [1, 1, 2, 2, 3, 3],
    'ItemId': [101, 102, 101, 103, 102, 103],
    'Rating': [4, 5, 3, 4, 2, 5]
}

df = pd.DataFrame(data)

# Prepare the data for Surprise
reader = Reader(rating_scale=(1, 5))  # Assuming ratings are from 1 to 5
dataset = Dataset.load_from_df(df[['UserId', 'ItemId', 'Rating']], reader)

2. Train a Recommendation Model using KNN

We’ll train a KNN-based collaborative filtering model to recommend items based on user similarities. Here’s how to set it up:

from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Split the data into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Use KNN-based collaborative filtering
sim_options = {
    'name': 'cosine',  # Similarity measure
    'user_based': True  # True for user-based, False for item-based
}

# Train the KNN model
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Test the model and evaluate accuracy
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f"Root Mean Squared Error: {rmse:.4f}")

3. Making Predictions

After training the model, you can make personalized recommendations for a user.

def get_top_n(predictions, n=3):
    """Return the top N recommendations for each user."""
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Sort the predictions for each user and return the top N
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Get the top 3 recommendations for each user
top_n = get_top_n(predictions, n=3)

# Display the top recommendations
for uid, user_ratings in top_n.items():
    print(f"User {uid} - Top 3 recommended items:")
    for iid, rating in user_ratings:
        print(f"  Item {iid} with estimated rating {rating:.2f}")

Step 4: Content-Based Filtering (Optional)

For content-based filtering, the idea is to recommend items similar to those a user has liked, based on the features (such as category, description, etc.) of the items themselves.

You can use cosine similarity between item features to achieve this. Here’s an example of how to compute item similarity using TF-IDF Vectorizer and cosine similarity from scikit-learn.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample item descriptions (you can replace with your own item data)
item_data = {
    'ItemId': [101, 102, 103],
    'Description': [
        "This is a great item for gaming",
        "Perfect for watching movies",
        "Great for sports and outdoor activities"
    ]
}

item_df = pd.DataFrame(item_data)

# Vectorize the item descriptions using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(item_df['Description'])

# Compute cosine similarity between items
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print item-item similarity matrix
print("Item-Item Similarity Matrix:")
print(cosine_sim)

# Recommend items similar to Item 101
def recommend_content_based(item_id, top_n=3):
    idx = item_df[item_df['ItemId'] == item_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the item itself
    recommended_items = [item_df['ItemId'][i[0]] for i in sim_scores]
    return recommended_items

# Recommend 3 items similar to Item 101
recommended_items = recommend_content_based(101, top_n=3)
print(f"Items similar to Item 101: {recommended_items}")

Step 5: Evaluation and Testing

For evaluating the system, you can measure performance metrics such as Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Precision@k depending on the task at hand.
Step 6: Hybrid Model (Optional)

You can combine both collaborative filtering and content-based filtering approaches for a hybrid recommendation system. Here's a simple example of combining both methods:

def hybrid_recommendation(user_id, top_n=3):
    # Get collaborative filtering recommendations
    collaborative_recs = get_top_n(predictions, n=top_n).get(user_id, [])
    collaborative_item_ids = [item[0] for item in collaborative_recs]
    
    # Get content-based filtering recommendations
    content_based_recs = []
    for item_id in collaborative_item_ids:
        content_based_recs.extend(recommend_content_based(item_id, top_n=top_n))

    # Combine and filter out duplicates
    final_recs = list(set(collaborative_item_ids + content_based_recs))[:top_n]
    return final_recs

# Recommend top 3 items for User 1 using hybrid method
user_id = 1
recommended_items = hybrid_recommendation(user_id, top_n=3)
print(f"Hybrid Recommendations for User {user_id}: {recommended_items}")

Conclusion:

This code gives a basic implementation of a collaborative filtering and content-based filtering recommendation system. You can modify it based on your specific use case, such as by adding more features (e.g., timestamps, demographic data), tuning model parameters, or trying out advanced models like Matrix Factorization, Deep Learning-based approaches (Autoencoders, Neural Collaborative Filtering), or Hybrid Systems.
