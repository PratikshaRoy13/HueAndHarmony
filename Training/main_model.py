# Install required libraries
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "D:\\Diversion\\Training\\updated_file.csv" 
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Functions to extract structured information
def extract_rating(text):
    """Extracts rating (0-10) from the text."""
    if pd.isna(text):
        return None
    match = re.search(r"(\d+(\.\d+)?)/10", str(text))
    return float(match.group(1)) if match else None

def extract_pros(text):
    """Extracts pros list from the text."""
    if pd.isna(text):
        return []
    match = re.search(r"Pros:\s*\n- (.*?)(\n\n|Cons:|$)", str(text), re.DOTALL)
    return [p.strip() for p in match.group(1).split("\n- ")] if match else []

def extract_cons(text):
    """Extracts cons list from the text."""
    if pd.isna(text):
        return []
    match = re.search(r"Cons:\s*\n- (.*?)(\n\n|Verdict:|$)", str(text), re.DOTALL)
    return [c.strip() for c in match.group(1).split("\n- ")] if match else []

def extract_verdict(text):
    """Extracts verdict from the text."""
    if pd.isna(text):
        return ""
    match = re.search(r"Verdict:\s*(.*)", str(text), re.DOTALL)
    return match.group(1).strip() if match else ""

# Apply extraction functions to each skin type column
for skin_type in ["oily_skin", "dry_skin", "combination_skin"]:
    df[f"{skin_type}_rating"] = df[skin_type].apply(extract_rating)
    df[f"{skin_type}_pros"] = df[skin_type].apply(extract_pros)
    df[f"{skin_type}_cons"] = df[skin_type].apply(extract_cons)
    df[f"{skin_type}_verdict"] = df[skin_type].apply(extract_verdict)

# Drop old text columns
df.drop(columns=["oily_skin", "dry_skin", "combination_skin"], inplace=True)

# Fill missing ratings with the average rating of each column
for skin_type in ["oily_skin", "dry_skin", "combination_skin"]:
    df[f"{skin_type}_rating"].fillna(df[f"{skin_type}_rating"].mean(), inplace=True)

# Ensure pros, cons, and verdicts are lists and strings respectively
for skin_type in ["oily_skin", "dry_skin", "combination_skin"]:
    df[f"{skin_type}_pros"] = df[f"{skin_type}_pros"].apply(lambda x: x if isinstance(x, list) else [])
    df[f"{skin_type}_cons"] = df[f"{skin_type}_cons"].apply(lambda x: x if isinstance(x, list) else [])
    df[f"{skin_type}_verdict"] = df[f"{skin_type}_verdict"].apply(lambda x: x if isinstance(x, str) else "")

# Vectorize ingredients using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["ingredients"].fillna("")).toarray()

# Define target variables (ratings)
y_oily = df["oily_skin_rating"].values
y_dry = df["dry_skin_rating"].values
y_comb = df["combination_skin_rating"].values

# Split dataset into training and testing sets
X_train, X_test, y_oily_train, y_oily_test = train_test_split(X, y_oily, test_size=0.2, random_state=42)
_, _, y_dry_train, y_dry_test = train_test_split(X, y_dry, test_size=0.2, random_state=42)
_, _, y_comb_train, y_comb_test = train_test_split(X, y_comb, test_size=0.2, random_state=42)

# Train Random Forest models
model_oily = RandomForestRegressor(n_estimators=100, random_state=42)
model_dry = RandomForestRegressor(n_estimators=100, random_state=42)
model_comb = RandomForestRegressor(n_estimators=100, random_state=42)

model_oily.fit(X_train, y_oily_train)
model_dry.fit(X_train, y_dry_train)
model_comb.fit(X_train, y_comb_train)

# Checking If Training Is Successful
print(model_oily)
print(model_dry)
print(model_comb)

# Function to predict rating and extract pros/cons/verdict
def predict_product(ingredients_list, skin_type):
    """Predicts the rating, pros, cons, verdict, and final suggestion for a product."""

    input_features = vectorizer.transform([" ".join(ingredients_list)]).toarray()

    if skin_type.lower() == "oily":
        rating = round(model_oily.predict(input_features)[0], 2)
        pros_column, cons_column, verdict_column = "oily_skin_pros", "oily_skin_cons", "oily_skin_verdict"
    elif skin_type.lower() == "dry":
        rating = round(model_dry.predict(input_features)[0], 2)
        pros_column, cons_column, verdict_column = "dry_skin_pros", "dry_skin_cons", "dry_skin_verdict"
    elif skin_type.lower() == "combination":
        rating = round(model_comb.predict(input_features)[0], 2)
        pros_column, cons_column, verdict_column = "combination_skin_pros", "combination_skin_cons", "combination_skin_verdict"
    else:
        return "Invalid skin type"

    # Find similar products based on ingredients
    similar_products = df[df["ingredients"].apply(lambda x: any(ing.lower() in x.lower() for ing in ingredients_list))]

    if similar_products.empty:
        return {
            "Predicted Rating": f"{rating} / 10",
            "Pros": ["No similar products found"],
            "Cons": ["No similar products found"],
            "Verdict": ["No verdict available"],
            "Final Suggestion": "Use with caution." if 5 <= rating <= 6 else "You can choose better alternatives." if rating < 5 else "Go for it!"
        }

    # Extract top pros, cons, and verdicts
    top_pros = similar_products[pros_column].explode().dropna().unique().tolist()[:5]
    top_cons = similar_products[cons_column].explode().dropna().unique().tolist()[:5]
    top_verdicts = similar_products[verdict_column].dropna().unique().tolist()[:1]  # Take only 1 verdict

    # Final suggestion
    final_suggestion = "Go for it!" if rating > 6 else "Use it if you want but not excessively" if rating in [5, 6] else "You can choose better alternatives."

    return {
        "Predicted Rating": f"{rating} / 10",
        "Pros": top_pros,
        "Cons": top_cons,
        "Verdict": top_verdicts,
        "Final Suggestion": final_suggestion
    }

print("\n")
test_ingredients = ["Hyaluronic Acid", "Glycerin"]
test_skin_type = "oily"
print(predict_product(test_ingredients, test_skin_type))

print("\n")
test_ingredients = ["Salicylic Acid", "Niacinamide", "Glycolic Acid"]
test_skin_type = "oily"
print(predict_product(test_ingredients, test_skin_type))
