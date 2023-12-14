import json

# Load the JSON file
with open("jldp-animl-cct.json", 'r') as f:
    data = json.load(f)

# Extract category names from the JSON data
categories = data['categories']
category_names = [category['name'] for category in categories]

# Create the .yaml file
with open("data.yaml", "w") as f:
    f.write("names: [\n")
    for name in category_names:
        f.write(f"  '{name}',\n")
    f.write("]\n")
