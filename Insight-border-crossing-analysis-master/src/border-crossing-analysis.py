"""
Insight Border Crossing Analysis Source Code
"""

inpt = "./input/Border_Crossing_Entry_Data.csv"

# create list of lists using entire dataset (sans 1st row), converting into an m x n array
with open(inpt, 'r') as file:
    datalist = []
    next(file, None) # skip first line (i.e., variable name) in data
    for line in file:
        row = line.split(',')
        datalist.append(row)
    file.close()

# create two separate lists for both borders; each list contains 4 columns
def border_list(array, border):
    lst = []
    for col in array:
        if col[3]==border:
            lst.append([col[4], col[5], int(col[6])])
    return lst
    
uscan = border_list(datalist,"US-Canada Border")
usmex = border_list(datalist,"US-Mexico Border")

# find unique Measure for each Border
def unique(border_list, measure):
    lst = []
    for col in border_list:
        if col[measure] not in lst: 
            lst.append(col[measure]) 
    # print list 
    return lst

unique_uscan = unique(uscan,1)
unique_usmex = unique(usmex,1)

# now create lists of separate measures with total aggregated crossings per month
def date_measure_sum(border_list, border, measure):
    lst = []
    for col in border_list:
        if col[1]==measure:
            lst.append([col[0], int(col[2])])
    sums = {}
    for i in lst:
        sums[tuple(i[:-1])] = (sums.get(tuple(i[:-1]),0) + i[-1])
        x = [[border,str(a),measure,sums[(a)]] for a in sums]
    # clean up Date column
    for y in x:
        y[1] = (y[1].replace("('", ''))
        y[1] = (y[1].replace("',)", ''))
    return x


#Instead of a cumulative rolling avg, use this code for a simple 2-month moving average
# calculate a 2-month moving average for total number of crossings
def rolling_average(measure_list):
    import math
    if len(measure_list) > 1:
        els = [x[-1] for x in measure_list] # list of last element in each list; sum per list
        moving_average = [math.ceil(sum(els[i:i+2])/2) for i in range(len(els)-1)]
        moving_average.insert(len(moving_average),0)
        for x, y in zip(measure_list, moving_average):
            x.append(int(y))
            t = measure_list
        return t
    else:
        none = [0]
        for x, y in zip(measure_list, none):
            x.append(y)
            t = measure_list
        return t

'''
# calculate a cumulative rolling average for total number of crossings
def rolling_average(measure_list):
    import math
    if len(measure_list) > 1:
        els = [x[-1] for x in measure_list] # list of last element in each list; sum per list
        sums, run_avg = 0, [0]
        for index, value in enumerate(els[::-1], start=1):
         sums += value
         run_avg.append(math.ceil(sums/index))
        run_avg.pop()
        for x, y in zip(measure_list, run_avg[::-1]):
            x.append(int(y))
            t = measure_list
        return t
    else:
        none = [0]
        for x, y in zip(measure_list, none):
            x.append(y)
            t = measure_list
        return t
'''
# for each unique measure at each border, create new lists using both functions above
data = []
print("Measures at US-Canada border:\n", unique_usmex)
if 'Truck Containers Full' in unique_usmex:
    usmex_truck_con_full = date_measure_sum(usmex,"US-Mexico Border",'Truck Containers Full')
    usmex_truck_con_full = rolling_average(usmex_truck_con_full)
    data.extend(usmex_truck_con_full)
if 'Truck Containers Empty' in unique_usmex:
    usmex_truck_con_empty = date_measure_sum(usmex,"US-Mexico Border",'Truck Containers Empty')
    usmex_truck_con_empty = rolling_average(usmex_truck_con_empty)
    data.extend(usmex_truck_con_empty)
if 'Personal Vehicles' in unique_usmex:
    usmex_personal_v = date_measure_sum(usmex,"US-Mexico Border",'Personal Vehicles')
    usmex_personal_v = rolling_average(usmex_personal_v)
    data.extend(usmex_personal_v)
if 'Personal Vehicle Passengers' in unique_usmex:
    usmex_personal_p = date_measure_sum(usmex,"US-Mexico Border",'Personal Vehicle Passengers')
    usmex_personal_p = rolling_average(usmex_personal_p)
    data.extend(usmex_personal_p)
if 'Pedestrians' in unique_usmex:
    usmex_peds = date_measure_sum(usmex,"US-Mexico Border",'Pedestrians')
    usmex_peds = rolling_average(usmex_peds)
    data.extend(usmex_peds)
if 'Buses' in unique_usmex:
    usmex_buses = date_measure_sum(usmex,"US-Mexico Border",'Buses')
    usmex_buses = rolling_average(usmex_buses)
    data.extend(usmex_buses)
if 'Bus Passengers' in unique_usmex:
    usmex_bus_p = date_measure_sum(usmex,"US-Mexico Border",'Bus Passengers')
    usmex_bus_p = rolling_average(usmex_bus_p)
    data.extend(usmex_bus_p)
if 'Trucks' in unique_usmex:
    usmex_trucks = date_measure_sum(usmex,"US-Mexico Border",'Trucks')
    usmex_trucks = rolling_average(usmex_trucks)
    data.extend(usmex_trucks)
if 'Trains' in unique_usmex:
    usmex_trains = date_measure_sum(usmex,"US-Mexico Border",'Trains')
    usmex_trains = rolling_average(usmex_trains)
    data.extend(usmex_trains)
if 'Train Passengers' in unique_usmex:
    usmex_train_p = date_measure_sum(usmex,"US-Mexico Border",'Train Passengers')
    usmex_train_p = rolling_average(usmex_train_p)
    data.extend(usmex_train_p)
if 'Rail Containers Full' in unique_usmex:
    usmex_rail_con_full = date_measure_sum(usmex,"US-Mexico Border",'Rail Containers Full')
    usmex_rail_con_full = rolling_average(usmex_rail_con_full)
    data.extend(usmex_rail_con_full)
if 'Rail Containers Empty' in unique_usmex:
    usmex_rail_con_empty = date_measure_sum(usmex,"US-Mexico Border",'Rail Containers Empty')
    usmex_rail_con_empty = rolling_average(usmex_rail_con_empty)
    data.extend(usmex_rail_con_empty)

print("Measures at US-Canada border:\n", unique_uscan)
if 'Truck Containers Full' in unique_uscan:
    uscan_truck_con_full = date_measure_sum(uscan,"US-Canada Border",'Truck Containers Full')
    uscan_truck_con_full = rolling_average(uscan_truck_con_full)
    data.extend(uscan_truck_con_full)
if 'Truck Containers Empty' in unique_uscan:
    uscan_truck_con_empty = date_measure_sum(uscan,"US-Canada Border",'Truck Containers Empty')
    uscan_truck_con_empty = rolling_average(uscan_truck_con_empty)
    data.extend(uscan_truck_con_empty)
if 'Personal Vehicles' in unique_uscan:
    uscan_personal_v = date_measure_sum(uscan,"US-Canada Border",'Personal Vehicles')
    uscan_personal_v = rolling_average(uscan_personal_v)
    data.extend(uscan_personal_v)
if 'Personal Vehicle Passengers' in unique_uscan:
    uscan_personal_p = date_measure_sum(uscan,"US-Canada Border",'Personal Vehicle Passengers')
    uscan_personal_p = rolling_average(uscan_personal_p)
    data.extend(uscan_personal_p)
if 'Pedestrians' in unique_uscan:
    uscan_peds = date_measure_sum(uscan,"US-Canada Border",'Pedestrians')
    uscan_peds = rolling_average(uscan_peds)
    data.extend(uscan_peds)
if 'Buses' in unique_uscan:
    uscan_buses = date_measure_sum(uscan,"US-Canada Border",'Buses')
    uscan_buses = rolling_average(uscan_buses)
    data.extend(uscan_buses)
if 'Bus Passengers' in unique_uscan:
    uscan_bus_p = date_measure_sum(uscan,"US-Canada Border",'Bus Passengers')
    uscan_bus_p = rolling_average(uscan_bus_p)
    data.extend(uscan_bus_p)
if 'Trucks' in unique_uscan:
    uscan_trucks = date_measure_sum(uscan,"US-Canada Border",'Trucks')
    uscan_trucks = rolling_average(uscan_trucks)
    data.extend(uscan_trucks)
if 'Trains' in unique_uscan:
    uscan_trains = date_measure_sum(uscan,"US-Canada Border",'Trains')
    uscan_trains = rolling_average(uscan_trains)
    data.extend(uscan_trains)
if 'Train Passengers' in unique_uscan:
    uscan_train_p = date_measure_sum(uscan,"US-Canada Border",'Train Passengers')
    uscan_train_p = rolling_average(uscan_train_p)
    data.extend(uscan_train_p)
if 'Rail Containers Full' in unique_uscan:
    uscan_rail_con_full = date_measure_sum(uscan,"US-Canada Border",'Rail Containers Full')
    uscan_rail_con_full = rolling_average(uscan_rail_con_full)
    data.extend(uscan_rail_con_full)
if 'Rail Containers Empty' in unique_uscan:
    uscan_rail_con_empty = date_measure_sum(uscan,"US-Canada Border",'Rail Containers Empty')
    uscan_rail_con_empty = rolling_average(uscan_rail_con_empty)
    data.extend(uscan_rail_con_empty)

# sort by date
data.sort(key=lambda x: x[1], reverse=True)

# create column names
col_names = ("Border","Date","Measure","Value","Average")
data.insert(0,col_names)

import os
os.chdir("./output/")

# export data
import csv
with open("report.csv", "a",newline='') as f:
    writer = csv.writer(f)
    if type(data) is not int:
        writer.writerows(data)
    else:
        writer.writerow([data])
          
# Open and print the first five lines to the screen
with open("report.csv", "r",newline='') as f:
    a_line = f.readline()
    n_lines = 0
    while len(a_line) > 0 and n_lines < 6:
        n_lines += 1
        print(a_line)
        a_line = f.readline()
