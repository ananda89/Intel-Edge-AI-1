"""
    Calling the Food API for the nutritional information of the fruit/veg
    classified from the OpenVINO Inference Engine.
    It makes API calls to two different databases: Food and Nutritional
    database. Output of the former is required to make calls to the later.
    The outputs of the two API calls are stored in two JSON files.
"""

## Import the packages
import requests
import json

# API credentials
app_id = "d17b0b5a"
app_key = "4d793e156e0d01cccfeffcc5f2093ba1"

# name of the ingredient, output label for the test image
ingredient = 'mango'

# parameters for the food database API requests
parameters = {"ingr": ingredient,
                "app_id": app_id,
                "app_key": app_key,
                "category": "generic_foods"}

# URLs for the food and nutritional database
parser_url = "https://api.edamam.com/api/food-database/parser"
ntr_info_url = "https://api.edamam.com/api/food-database/nutrients"

# Calling the food database API
request = requests.get(parser_url, params=parameters)
if request.status_code:
    print("Success!! Getting the data...")
else:
    print("Request couldn't be processed!")

request_json = request.json()

# Writing the API output to a file
with open("parser.json", "w") as f:
    json.dump(request_json, f, indent=2)

# Input data for the nutritional API
foodId = request_json['parsed'][0]['food']['foodId']
measureURI = "http://www.edamam.com/ontologies/edamam.owl#Measure_gram"
quantity = 1

fruit_attr = {
    "ingredients": [
        {
            "quantity": quantity,
            "measureURI": measureURI,
            "foodId": foodId
        }
    ]
}

fruit_attr = json.dumps(fruit_attr)

# Defining headers and parameters for calling the nutritional database
headers = {
    "Content-Type": "application/json"
}

params = (
    ('app_id', app_id),
    ('app_key', app_key)
)

# Calling the nutritional database
response = requests.post(ntr_info_url, headers=headers, params=params, data=fruit_attr)
if response.status_code:
    print("Success!! Getting the nutritional data...")
else:
    print("Request couldn't be processed!")

response_json = response.json()

# Collecting nutritional information
ntr_vals = response_json['totalNutrients']
print('Nutritional information for ' + ingredient + '(1 gram):\n')

# Outputting the nutritional values
for k,v in ntr_vals.items(): 
    print('{0}:   {1:.5f} {2}'.format(v['label'], v['quantity'], v['unit'])) 

# saving it in a file
with open("nutrient_info.json", "w") as f:
    json.dump(response_json, f, indent=2)