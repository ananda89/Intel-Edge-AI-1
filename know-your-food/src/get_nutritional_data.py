"""
    Calling the Food API for the nutritional information of the fruit/veg
    classified from the Intel's OpenVINO Inference Engine.
    It makes API calls to two different databases: Food and Nutritional
    database. Output of the former is required to make calls to the later.
    The outputs of the two API calls are stored in two JSON files.
"""

## Import the packages
import requests
import json

# URLs for the food and nutritional database
parser_url = "https://api.edamam.com/api/food-database/parser"
ntr_info_url = "https://api.edamam.com/api/food-database/nutrients"
measureURI = "http://www.edamam.com/ontologies/edamam.owl#Measure_unit"

def get_nutritional_data(ingredient, api_id=None, api_key=None):

    # API credentials - as supplied in the CLI
    app_key = api_key
    app_id = api_id

    if app_key is None or app_id is None:
        with open('../data/nutrient_info.json') as f:
            response_json = json.load(f)
    else:
        # parameters for the food database API requests
        parameters = {"ingr": ingredient,
                        "app_id": app_id,
                        "app_key": app_key,
                        "category": "generic_foods"}

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

    # saving it in a file
    with open("../outputs/nutrient_info.json", "w") as f:
        json.dump(response_json, f, indent=2)

    # Collecting nutritional information
    ntr_vals = response_json['totalNutrients']
    return_string = 'Nutritional information for ' + ingredient + '(1 unit):\n'
    print(return_string)
    n_items = sorted(ntr_vals.items(), key=lambda item: item[1]['quantity'], reverse=True)
    for k,v in n_items: 
        current_string = '{:>25} : {:>8.2f} {}'.format(v['label'], v['quantity'], v['unit'])
        return_string += current_string + '\n'
        print(current_string)        
    
    return return_string.replace(u'\u00b5', "u")
