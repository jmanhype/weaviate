import pandas as pd
import weaviate
from requests.exceptions import InvalidJSONError
import math

def replace_non_compliant_floats(value):
    """
    Replace non-compliant float values with a compliant substitute.
    """
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None  # or a default value like 0.0
    return value

def define_schema(client):
    """
    Define the "ProductInfo" class schema in Weaviate.
    """
    schema = {
        "classes": [{
            "class": "ProductInfo",
            "properties": [
                {"name": "title", "dataType": ["string"]},
                {"name": "productType", "dataType": ["string"]},
                {"name": "tags", "dataType": ["string[]"]},
                {"name": "description", "dataType": ["text"]},
                {"name": "onlineStoreUrl", "dataType": ["string"]},
                {"name": "salesGuidance", "dataType": ["string"]},
                {"name": "imageUrl", "dataType": ["string"]},
                {"name": "price", "dataType": ["number"]}
                # Add more properties as needed
            ]
        }]
    }
    # Delete the class if it already exists (optional)
    try:
        client.schema.delete_class('ProductInfo')
    except Exception as e:
        print(f"Could not delete class ProductInfo, possibly does not exist. Error: {e}")
    # Create the new class schema
    client.schema.create(schema)
    print("Schema for 'ProductInfo' created successfully.")
    
def import_data(client, csv_path):
    """
    Import data from a CSV file into the "ProductInfo" class in Weaviate.
    """
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        try:
            product_info = {
                "title": row["title"],
                "productType": row["productType"],
                "tags": row["tags"].split(',') if pd.notna(row["tags"]) else [],
                "description": row["description"],
                "onlineStoreUrl": row["onlineStoreUrl"],
                "salesGuidance": row["salesGuidance"],
                "imageUrl": row["image_url"],
                "price": replace_non_compliant_floats(row["price"]),
                # Map additional properties as needed
            }
            client.data_object.create(
                data_object=product_info,
                class_name="ProductInfo"
            )
        except InvalidJSONError as e:
            print(f"Invalid JSON for row {index}: {e}")

def main():
    """
    Main function to define schema and import data into Weaviate.
    """
    weaviate_url = "http://localhost:8080"
    csv_path = './stateofart_alkhemy_products_-_alkhemy_products.csv'

    client = weaviate.Client(weaviate_url)
    define_schema(client)
    import_data(client, csv_path)

if __name__ == "__main__":
    main()
