import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import nltk
import matplotlib.pyplot as plt
import time
import argparse
nltk.download('vader_lexicon')

def plot_best_selling_dishes(data, output_file, cuisine, location):
    """Plot a bar chart of best-selling dishes and save as PNG."""
    if data.empty:
        print(f"No data to plot for {cuisine} in {location}")
        return
    plt.figure(figsize=(10, 6))
    data.value_counts().plot(kind="bar")
    plt.xlabel("Dishes")
    plt.ylabel("Number of Restaurants")
    plt.title(f"Most Popular {cuisine} Dishes in {location}")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

def get_restaurants_and_reviews(location, term, categories, limit=50, max_results=100):
    """Fetch restaurant reviews from Yelp API."""
    data = []
    for offset in range(0, max_results, limit):
        params = {"location": location, "term": term, "categories": categories, "limit": limit, "offset": offset}
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            restaurants = response.json().get("businesses", [])
            if not restaurants:
                print(f"No restaurants found for {location}, {categories}")
                break
            for restaurant in restaurants:
                restaurant_id = restaurant.get("id")
                restaurant_name = restaurant.get("name")
                review_response = requests.get(f"https://api.yelp.com/v3/businesses/{restaurant_id}/reviews", headers=headers)
                if review_response.status_code == 200:
                    reviews = review_response.json().get("reviews", [])
                    for review in reviews:
                        data.append({
                            "restaurant_id": restaurant_id,
                            "restaurant_name": restaurant_name,
                            "review_text": review.get("text", ""),
                            "rating": review.get("rating", 0),
                        })
                elif review_response.status_code == 429:
                    print("Rate limit hit. Waiting 60 seconds...")
                    time.sleep(60)
                else:
                    print(f"Review request failed for {restaurant_id}: {review_response.status_code}")
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            time.sleep(10)
    return data


def analyze_reviews(location, term, categories, dish_names):
    """Analyze reviews to find best-selling dishes and train a classifier."""
    data = get_restaurants_and_reviews(location, term, categories)
    if not data:
        print(f"No data retrieved for {categories} in {location}")
        return None, None
    df = pd.DataFrame(data)
    df.drop_duplicates(subset=["review_text"], inplace=True)
    if df.empty:
        print(f"No unique reviews for {categories} in {location}")
        return None, None

    # Sentiment analysis
    sid = SentimentIntensityAnalyzer()
    df["sentiment"] = df["review_text"].apply(lambda x: sid.polarity_scores(x)["compound"])

    # Dish detection
    for dish in dish_names:
        df[dish] = df["review_text"].apply(lambda x: 1 if dish.lower() in x.lower() else 0)

    # Aggregate dish mentions by restaurant
    df_dishes = df.groupby("restaurant_id")[dish_names].sum()
    df_dishes["best_selling_dish"] = df_dishes.idxmax(axis=1)
    df_best_selling_dishes = df_dishes.reset_index()[["restaurant_id", "best_selling_dish"]]
    df_best_selling_dishes = df_best_selling_dishes.merge(
        df[["restaurant_id", "restaurant_name"]].drop_duplicates(), on="restaurant_id", how="left"
    )

    # Machine learning
    X = df[["review_text", "sentiment"]]
    y = df[dish_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train["review_text"])
    X_test_vectorized = vectorizer.transform(X_test["review_text"])
    X_train_vectorized = np.hstack((X_train_vectorized.toarray(), X_train[["sentiment"]].to_numpy()))
    X_test_vectorized = np.hstack((X_test_vectorized.toarray(), X_test[["sentiment"]].to_numpy()))

    # Identify labels with both classes in y_train
    labels_with_both_classes = [col for col in y_train.columns if y_train[col].nunique() == 2]
    labels_with_only_zero = [col for col in y_train.columns if y_train[col].nunique() == 1 and y_train[col].unique()[0] == 0]

    # Optional: Inform user about skipped labels
    if labels_with_only_zero:
        print(f"Labels with only class 0 in training data (skipped for training): {labels_with_only_zero}")

    # Train classifier only on labels with both classes
    if labels_with_both_classes:
        y_train_both = y_train[labels_with_both_classes]
        y_test_both = y_test[labels_with_both_classes]
        clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
        clf.fit(X_train_vectorized, y_train_both)
        y_pred_both = clf.predict(X_test_vectorized)
        y_pred_both_df = pd.DataFrame(y_pred_both, columns=labels_with_both_classes)
    else:
        y_pred_both_df = pd.DataFrame()
        print(f"No labels with both classes in training data for {categories} in {location}. All predictions will be 0.")

    # Predict all zeros for labels with only one class in training data
    y_pred_only_zero_df = pd.DataFrame(0, index=range(X_test.shape[0]), columns=labels_with_only_zero)

    # Combine predictions
    y_pred_df = pd.concat([y_pred_both_df, y_pred_only_zero_df], axis=1)

    # Ensure column order matches y_test
    y_pred_df = y_pred_df[y_test.columns]

    # Convert to numpy array for evaluation
    y_pred = y_pred_df.to_numpy()

    # Calculate F1-score
    f1 = f1_score(y_test.to_numpy(), y_pred, average='micro')
    print(f"F1 Score (Micro) for {categories} in {location}: {f1:.2f}")

    return df, df_best_selling_dishes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze popular dishes from Yelp reviews")
    parser.add_argument("--location", default="Los Angeles, CA", help="City to analyze")
    parser.add_argument("--cuisine", default="Italian", help="Cuisine type (Italian, Indian, Mexican)")
    args = parser.parse_args()

    api_key = "YOUR API KEY HERE"  # Replace with your Yelp API key
    headers = {'Authorization': 'Bearer %s' % api_key}
    url = "https://api.yelp.com/v3/businesses/search"

    cuisine_dishes = {
        "Italian": ["pasta", "pizza", "lasagna", "risotto", "gnocchi", "spaghetti", "ravioli"],
        "Indian": ["biryani", "tandoori", "curry", "masala", "samosa", "dosa", "paneer"],
        "Mexican": ["tacos", "enchiladas", "burrito", "quesadilla", "guacamole", "salsa", "nachos"]
    }

    location = args.location
    cuisine = args.cuisine
    dish_names = cuisine_dishes.get(cuisine, cuisine_dishes["Italian"])

    df, df_best_selling_dishes = analyze_reviews(location, "restaurants", cuisine.lower(), dish_names)
    if df is not None and df_best_selling_dishes is not None:
        df.to_csv(f"{cuisine}_reviews.csv", index=False)
        df_best_selling_dishes.to_csv(f"{cuisine}_best_selling_dishes.csv", index=False)
        best_selling_dishes_by_restaurant = df_best_selling_dishes.groupby("restaurant_name")["best_selling_dish"].first()
        plot_best_selling_dishes(best_selling_dishes_by_restaurant, f"{cuisine}_best_selling_dishes.png", cuisine, location)