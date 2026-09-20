# --- one-time uploader: upload Fake.csv and True.csv into MongoDB ---

import pandas as pd
from datetime import datetime
from pymongo import MongoClient

# CONFIG - set your MongoDB URI and DB name
MONGO_URI = "mongodb+srv://devanshkapoor5_fake_news:fake-news-26180209@cluster0.npvkut6.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "fake_news_db"
FAKE_COLL_NAME = "fake_news_training_data"
REAL_COLL_NAME = "real_news_training_data"
# local notebook / local environment CSV paths
fake_path = "../Training/Fake.csv"
true_path = "../Training/True.csv"

# Connect
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fake_coll = db[FAKE_COLL_NAME]
real_coll = db[REAL_COLL_NAME]

def df_to_mongo(df, label_value, collection_name, collection_object):
    records = []
    for _, row in df.iterrows():
        records.append({
            "title":   row["title"],
            "text":    row["text"],
            "subject": row["subject"],
            "date":    row["date"],
            "label":   int(label_value),
            "created_at": datetime.utcnow()
        })
    if records:
        collection_object.insert_many(records)
        print(f"Inserted {len(records)} documents with label={label_value} into {DB_NAME}.{collection_name}")


# Load CSVs
df_fake = pd.read_csv(fake_path)
df_true = pd.read_csv(true_path)

print("Fake rows:", len(df_fake), " | True rows:", len(df_true))

# Upload
df_to_mongo(df=df_true, label_value=0, collection_name=REAL_COLL_NAME, collection_object=real_coll)
df_to_mongo(df=df_fake, label_value=1, collection_name=FAKE_COLL_NAME, collection_object=fake_coll)

print("Done uploading CSVs to MongoDB.")
