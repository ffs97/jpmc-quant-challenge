import urllib2
import pyperclip
import numpy as np

from datetime import datetime
from sklearn.neural_network import MLPRegressor

tr_url = "https://s3-ap-southeast-1.amazonaws.com/mettl-arq/questions/codelysis/machine-learning/fare-prediction/train.csv"
ts_url = "https://s3-ap-southeast-1.amazonaws.com/mettl-arq/questions/codelysis/machine-learning/fare-prediction/test.csv"

def read_data(url, split_string=",", test=False):
    data = urllib2.urlopen(url)
    data.readline()
    data = data.readlines()
    
    def process(line):
        line = line.strip().split(split_string)
        
        if test:
            return line[1:]
        
        return line[1:-1], line[-1]
        
    data = [process(line) for line in data]
    
    if test:
        return data
    
    X, Y = zip(*data)
    
    X = list(X)
    Y = np.array(Y, dtype=float)
    
    return X, Y

X_tr, Y_tr = read_data(tr_url)
X_ts = read_data(ts_url, test=True)

cities = [(x[1], x[2]) for x in X_tr]
c1,c2 = zip(*cities)
cities = c1 + c2
cities = list(set(cities))
cities.sort()
tuple_cities = {}
index = 0
for i, city_1 in enumerate(cities):
    for j, city_2 in enumerate(cities[(i+1):]):
        tuple_cities[city_1 + city_2] = index
        tuple_cities[city_2 + city_1] = index
        index += 1

def process_features(X, tuple_cities):
    today = datetime.today()
    time_0 = datetime.strptime("0:0", "%H:%M")

    for i in range(len(X)):
        x = X[i]

        cities_index = tuple_cities[x[1] + x[2]]
        cities_one_hot = [0] * len(tuple_cities)
        cities_one_hot[cities_index]  = 1

        flight_day = datetime.strptime(x[3] + " " + x[4], "%Y-%m-%d %H:%M")
        bookind_day = datetime.strptime(x[5], "%Y-%m-%d")
        days_diff = (flight_day - bookind_day).days

        dob = datetime.strptime(X[i][0], "%Y-%m-%d")
        age = int(round((today - dob).days / 365.0))

        flight_time = datetime.strptime(x[4], "%H:%M")
        flight_time = (time_0 - flight_time).seconds / 600

        bclass = 0 if x[6] == "Economy" else 1

    #     X[i] = np.array([bclass, cities_index, flight_time, days_diff, age])
        X[i] = np.array([bclass, flight_time, days_diff, age] + cities_one_hot)

    X = np.array(X)
    return X

X_tr = process_features(X_tr, tuple_cities)
X_ts = process_features(X_ts, tuple_cities)

### Regression

#regressor = MLPRegressor(hidden_layer_sizes=(50, 20, 5), max_iter=10000)
#regressor.fit(X_tr, Y_tr)


eco_indices = (X_tr[:, 0] == 0)
regressor_eco = MLPRegressor(hidden_layer_sizes=(10, 20, 5), max_iter=10000, tol=0.00001)
regressor_eco.fit(X_tr[eco_indices][:, 1:], Y_tr[eco_indices])
regressor_eco.score(X_tr[eco_indices][:, 1:], Y_tr[eco_indices])

bus_indices = (X_tr[:, 0] == 1)
regressor_bus = MLPRegressor(hidden_layer_sizes=(10, 20, 5), max_iter=10000)
regressor_bus.fit(X_tr[bus_indices][:, 1:], Y_tr[bus_indices])
regressor_bus.score(X_tr[bus_indices][:, 1:], Y_tr[bus_indices])
Y_ts = regressor.predict(X_ts)

pyperclip.copy("return [" + ", ".join([str(y) for y in Y_ts]) + "]")
