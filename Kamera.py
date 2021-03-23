import json


with open("/home/caner/Documents/AirSim/settings.json", "r") as write_file:
    json.load(write_file) 

print(write_file)