import pandas
import json
import os

df=pandas.io.json.read_json('/Users/mthomson/web_dev/upload_files/kaggle_avazu_test_json_10000.json')

for index, row in df[:3].iterrows():
    # apparently it's (a bit) simpler to call the shell command directly instead
    post_data = "'[" + row.to_json() + "]'"
    x='curl -H "Content-Type: application/json" -X POST -d ' + post_data +' 127.0.0.1:5000/json_input'
    #print(x)
    os.system(x) # + "> /Users/mthomson/output.txt")