import csv
import io
from google.cloud import storage

BUCKET = 'asl-mixi-project-bucket'

CATEGORIES = [
    '(Unknown)',
    'Arts & Entertainment',
    'Autos & Vehicles',
    'Beauty & Fitness',
    'Books & Literature',
    'Business & Industrial',
    'Computers & Electronics',
    'Finance',
    'Food & Drink',
    'Games',
    'Health',
    'Hobbies & Leisure',
    'Home & Garden',
    'Internet & Telecom',
    'Jobs & Education',
    'Law & Government',
    'News',
    'People & Society',
    'Pets & Animals',
    'Real Estate',
    'Reference',
    'Science',
    'Shopping',
    'Sports',
    'Travel',
]
CATEGORY_NUM = len(CATEGORIES)

def category2id(category_name):
    id_ = CATEGORIES.index(category_name)
    assert id_ >= 0
    return id_

def get_id_category_id_table(class_num):
    client = storage.Client()
    bucket = client.get_bucket(BUCKET)
    blob = bucket.get_blob('data/youtube-8m/vocabulary.csv')
    csv_bytes = blob.download_as_string()
    print(csv_bytes)
    
    id_category_id_table = [0] * class_num
    with io.StringIO(csv_bytes.decode()) as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            id_category_id_table[int(r[0])] = category2id(r[5])
    return id_category_id_table