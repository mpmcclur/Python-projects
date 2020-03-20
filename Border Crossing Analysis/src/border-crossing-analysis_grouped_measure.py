"""
Created on Wed Feb 12 19:23:24 2020

@author: mmcclure
"""
inpt = "./input/Border_Crossing_Entry_Data.csv"

# create list of lists using entire dataset (sans 1st row), converting into a >351k x 7 array
with open(inpt, 'r') as file:
    datalist = []
    next(file, None) # skip first line (i.e., variable name) in data
    for line in file:
        row = line.split(',')
        datalist.append(row[0:]) 
    file.close()

# create two separate lists for both borders; each list contains 4 columns
def border_list(array, border):
    lst = []
    for col in array:
        if col[3]==border:
            lst.append([col[4], col[5], col[6]])
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
def date_measure_sum(border_list, measure):
    lst = []
    for col in border_list:
        if col[1]==measure:
            lst.append([col[0], int(col[2])])
    sums = {}
    for i in lst:
        sums[tuple(i[:-1])] = (sums.get(tuple(i[:-1]),0) + i[-1])
        x = [[str(a),sums[(a)]] for a in sums]
    # clean up Date column
    for y in x:
        y[0] = (y[0].replace("('", ''))
        y[0] = (y[0].replace("',)", ''))
    return x

def moving_average(measure_list):
    import math
    if len(measure_list) > 1:
        els = [x[-1] for x in measure_list] # list of last element in each list
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
Instead of calculating a simple 2-month moving average, use the code below for a cumulative rolling avg
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
'''
The following code requires the user to know which Measure to aggregate if the full dataset isn't used.
'''
print("Measures at US-Mexico border:\n", unique_usmex)
if len(unique_usmex) != 12:
    question = input("Refer to the list generated above.'\n'Enter a Vehicle to aggregate for crossings at the US-Mexico border (if none, enter 'n'): ")
    if question in ["Personal Vehicles", "Buses", "Trucks", "Trains"]:
        usmex_vehicles = date_measure_sum(usmex,question)
        usmex_vehicles = moving_average(usmex_vehicles)
        question2 = input ("Enter another vehicle? (y/n): ")
        if question2 == "y":
            question = input("Enter another vehicle: ")
            usmex_vehicles = date_measure_sum(usmex,question)
            usmex_vehicles = moving_average(usmex_vehicles)
            question2 = input ("Enter another vehicle? (y/n): ")
            if question2 == "y":
                question = input("Enter another vehicle: ")
                usmex_vehicles = date_measure_sum(usmex,question)
                usmex_vehicles = moving_average(usmex_vehicles)
                question2 = input ("Enter another vehicle? (y/n): ")
                if question2 == "y":
                    question = input("Enter another vehicle: ")
                    usmex_vehicles = date_measure_sum(usmex,question)
                    usmex_vehicles = moving_average(usmex_vehicles)

    question = input("Enter Equipment to aggregate for crossings at the US-Mexico border (if none, enter 'n'): ")
    if question in ["Truck Containers Full", "Truck Containers Empty", "Rail Containers Full", "Rail Containers Empty"]:
        usmex_equipment = date_measure_sum(usmex,question)
        usmex_equipment = moving_average(usmex_equipment)
        question2 = input ("Enter other equipment? (y/n): ")
        usmex_equipment = date_measure_sum(usmex,question)
        usmex_equipment = moving_average(usmex_equipment)
        if question2 == "y":
            question = input("Enter other equipment: ")
            usmex_equipment = date_measure_sum(usmex,question)
            usmex_equipment = moving_average(usmex_equipment)
            question2 = input ("Enter other equipment? (y/n): ")
            if question2 == "y":
                question = input("Enter other equipment: ")
                usmex_equipment = date_measure_sum(usmex,question)
                usmex_equipment = moving_average(usmex_equipment)
                question2 = input ("Enter other equipment? (y/n): ")
                if question2 == "y":
                    question = input("Enter other equipment: ")
                    usmex_equipment = date_measure_sum(usmex,question)
                    usmex_equipment = moving_average(usmex_equipment)

    question = input("Enter a Passenger to aggregate for crossings at the US-Mexico border (if none, enter 'n'): ")
    if question in ["Personal Vehicle Passengers", "Bus Passengers", "Train Passengers"]:
        usmex_passengers = date_measure_sum(usmex,question)
        usmex_passengers = moving_average(usmex_passengers)
        question2 = input ("Enter another passenger? (y/n): ")
        if question2 == "y":
            question = input("Enter another passenger: ")
            usmex_passengers = date_measure_sum(usmex,question)
            usmex_passengers = moving_average(usmex_passengers)
            question2 = input ("Enter another passenger? (y/n): ")
            if question2 == "y":
                question = input("Enter another passenger: ")
                usmex_passengers = date_measure_sum(usmex,question)
                usmex_passengers = moving_average(usmex_passengers)
                question2 = input ("Enter another passenger? (y/n): ")
                if question2 == "y":
                    question = input("Enter another passenger: ")
                    usmex_passengers = date_measure_sum(usmex,question)
                    usmex_passengers = moving_average(usmex_passengers)
    question = input("Were there any Pedestrians at the US-Mexico border? If none, enter 'n'; if yes, enter 'y': ")
    if "y" in question:
        usmex_pedestrians = date_measure_sum(usmex,"Pedestrians")
        usmex_pedestrians = moving_average(usmex_pedestrians)

print("Measures at US-Canada border:\n", unique_uscan)
if len(unique_uscan) != 12:
    question = input("Refer to the list generated above.\nEnter a Vehicle to aggregate for crossings at the US-Canada border (if n, enter 'none'): ")
    if question in ["Personal Vehicles", "Buses", "Trucks", "Trains"]:
        uscan_vehicles = date_measure_sum(uscan,question)
        uscan_vehicles = moving_average(uscan_vehicles)
        question2 = input ("Enter another vehicle? (y/n): ")
        if question2 == "y":
            question = input("Enter another vehicle: ")
            uscan_vehicles = date_measure_sum(uscan,question)
            uscan_vehicles = moving_average(uscan_vehicles)
            question2 = input ("Enter another vehicle? (y/n): ")
            if question2 == "y":
                question = input("Enter another vehicle: ")
                uscan_vehicles = date_measure_sum(uscan,question)
                uscan_vehicles = moving_average(uscan_vehicles)
                question2 = input ("Enter another vehicle? (y/n): ")
                if question2 == "y":
                    question = input("Enter another vehicle: ")
                    uscan_vehicles = date_measure_sum(uscan,question)
                    uscan_vehicles = moving_average(uscan_vehicles)
                    
    question = input("Enter Equipment to aggregate for crossings at the US-Canada border (if none, enter 'n'): ") 
    if question in ["Truck Containers Full",  "Truck Containers Empty", "Rail Containers Full",  "Rail Containers Empty"]:
        uscan_equipment = date_measure_sum(uscan,question)
        uscan_equipment = moving_average(uscan_equipment)
        question2 = input ("Enter other equipment? (y/n): ")
        if question2 == "y":
            question = input("Enter other equipment: ")
            uscan_equipment = date_measure_sum(uscan,question)
            uscan_equipment = moving_average(uscan_equipment)
            question2 = input ("Enter other equipment? (y/n): ")
            if question2 == "y":
                question = input("Enter other equipment: ")
                uscan_equipment = date_measure_sum(uscan,question)
                uscan_equipment = moving_average(uscan_equipment)
                question2 = input ("Enter other equipment? (y/n): ")
                if question2 == "y":
                    question = input("Enter other equipment: ")
                    usmex_equipment = date_measure_sum(uscan,question)
                    uscan_equipment = moving_average(uscan_equipment)                

    question = input("Enter a Passenger to aggregate for crossings at the US-Canada border (if none, enter 'n'): ")
    if question in ["Personal Vehicle Passengers",  "Bus Passengers", "Train Passengers"]:
        uscan_passengers = date_measure_sum(uscan,question)
        uscan_passengers = moving_average(uscan_passengers)
        question2 = input ("Enter another passenger? (y/n): ")
        if question2 == "y":
            question = input("Enter another passenger: ")
            uscan_passengers = date_measure_sum(uscan,question)
            uscan_passengers = moving_average(uscan_passengers)
            question2 = input ("Enter another passenger? (y/n): ")
            if question2 == "y":
                question = input("Enter another passenger: ")
                uscan_passengers = date_measure_sum(uscan,question)
                uscan_passengers = moving_average(uscan_passengers)
                question2 = input ("Enter another passenger? (y/n): ")
                if question2 == "y":
                    question = input("Enter another passenger: ")
                    uscan_passengers = date_measure_sum(uscan,question)
                    uscan_passengers = moving_average(uscan_passengers)
    question = input("Were there any Pedestrians at the US-Canada border? If none, enter 'n'; if yes, enter 'y': ")
    if "Pedestrians" in question:
        uscan_pedestrians = date_measure_sum(uscan,"Pedestrians")
        uscan_pedestrians = moving_average(uscan_pedestrians)

def q3():
    usmex_truck_con_full = date_measure_sum(usmex,"Truck Containers Full")
    usmex_truck_con_full = moving_average(usmex_truck_con_full)
    usmex_truck_con_empty = date_measure_sum(usmex,"Truck Containers Empty")
    usmex_truck_con_empty = moving_average(usmex_truck_con_empty)
    usmex_personal_v = date_measure_sum(usmex,"Personal Vehicles")
    usmex_personal_v = moving_average(usmex_personal_v)
    usmex_personal_p = date_measure_sum(usmex,"Personal Vehicle Passengers")
    usmex_personal_p = moving_average(usmex_personal_p)
    usmex_peds = date_measure_sum(usmex,"Pedestrians")
    usmex_peds = moving_average(usmex_peds)
    usmex_buses = date_measure_sum(usmex,"Buses")
    usmex_buses = moving_average(usmex_buses)
    usmex_bus_p = date_measure_sum(usmex,"Bus Passengers")
    usmex_bus_p = moving_average(usmex_bus_p)
    usmex_trucks = date_measure_sum(usmex,"Trucks")
    usmex_trucks = moving_average(usmex_trucks)
    usmex_trains = date_measure_sum(usmex,"Trains")
    usmex_trains = moving_average(usmex_trains)
    usmex_train_p = date_measure_sum(usmex,"Train Passengers")
    usmex_train_p = moving_average(usmex_train_p)
    usmex_rail_con_full = date_measure_sum(usmex,"Rail Containers Full")
    usmex_rail_con_full = moving_average(usmex_rail_con_full)
    usmex_rail_con_empty = date_measure_sum(usmex,"Rail Containers Empty")
    usmex_rail_con_empty = moving_average(usmex_rail_con_empty)
    return (usmex_truck_con_full,usmex_truck_con_empty,usmex_personal_v,usmex_personal_p,usmex_peds,usmex_buses,usmex_bus_p,usmex_trucks,usmex_trains,usmex_train_p,usmex_rail_con_full,usmex_rail_con_empty)

def q4():
    uscan_truck_con_full = date_measure_sum(uscan,"Truck Containers Full")
    uscan_truck_con_full = moving_average(uscan_truck_con_full)
    uscan_truck_con_empty = date_measure_sum(uscan,"Truck Containers Empty")
    uscan_truck_con_empty = moving_average(uscan_truck_con_empty)
    uscan_personal_v = date_measure_sum(uscan,"Personal Vehicles")
    uscan_personal_v = moving_average(uscan_personal_v)
    uscan_personal_p = date_measure_sum(uscan,"Personal Vehicle Passengers")
    uscan_personal_p = moving_average(uscan_personal_p)
    uscan_peds = date_measure_sum(uscan,"Pedestrians")
    uscan_peds = moving_average(uscan_peds)
    uscan_buses = date_measure_sum(uscan,"Buses")
    uscan_buses = moving_average(uscan_buses)
    uscan_bus_p = date_measure_sum(uscan,"Bus Passengers")
    uscan_bus_p = moving_average(uscan_bus_p)
    uscan_trucks = date_measure_sum(uscan,"Trucks")
    uscan_trucks = moving_average(uscan_trucks)
    uscan_trains = date_measure_sum(uscan,"Trains")
    uscan_trains = moving_average(uscan_trains)
    uscan_train_p = date_measure_sum(uscan,"Train Passengers")
    uscan_train_p = moving_average(uscan_train_p)
    uscan_rail_con_full = date_measure_sum(uscan,"Rail Containers Full")
    uscan_rail_con_full = moving_average(uscan_rail_con_full)
    uscan_rail_con_empty = date_measure_sum(uscan,"Rail Containers Empty")
    uscan_rail_con_empty = moving_average(uscan_rail_con_empty)
    return (uscan_truck_con_full,uscan_truck_con_empty,uscan_personal_v,uscan_personal_p,uscan_peds,uscan_buses,uscan_bus_p,uscan_trucks,uscan_trains,uscan_train_p,uscan_rail_con_full,uscan_rail_con_empty)


if len(unique_usmex) == 12:
    usmex_truck_con_full,usmex_truck_con_empty,usmex_personal_v,usmex_personal_p,usmex_peds,usmex_buses,usmex_bus_p,usmex_trucks,usmex_trains,usmex_train_p,usmex_rail_con_full,usmex_rail_con_empty=q3()

if len(unique_uscan) == 12:
    uscan_truck_con_full,uscan_truck_con_empty,uscan_personal_v,uscan_personal_p,uscan_peds,uscan_buses,uscan_bus_p,uscan_trucks,uscan_trains,uscan_train_p,uscan_rail_con_full,uscan_rail_con_empty=q4()

  
# combine vehicles, equipment, passengers, and pedestrians by border
# US-Canada
if globals() in ['uscan_personal_v', 'uscan_buses', 'uscan_trucks', 'uscan_trains']:
    uscan_vehicles = (uscan_personal_v+uscan_buses+uscan_trucks+uscan_trains)
elif 'uscan_personal_v' in globals():
    uscan_vehicles = (uscan_personal_v)
elif 'uscan_buses' in globals():
    uscan_vehicles = (uscan_buses)
elif 'uscan_trucks' in globals():
    uscan_vehicles = (uscan_trucks)
elif 'uscan_trains' in globals():
    uscan_vehicles = (uscan_trains)
    
if globals() in ['uscan_truck_con_full', 'uscan_truck_con_empty', 'uscan_rail_con_full', 'uscan_rail_con_empty']:
    uscan_equipment = (uscan_truck_con_full+uscan_truck_con_empty+uscan_rail_con_full+uscan_rail_con_empty)
elif 'uscan_truck_con_full' in globals():
    uscan_equipment = (uscan_truck_con_full)
elif 'uscan_truck_con_empty' in globals():
    uscan_equipment = (uscan_truck_con_empty)
elif 'uscan_rail_con_full' in globals():
    uscan_equipment = (uscan_rail_con_full)  
elif 'uscan_rail_con_empty' in globals():
    uscan_equipment = (uscan_rail_con_empty)
    
if globals() in ['uscan_personal_p', 'uscan_bus_p', 'uscan_train_p']:
    uscan_passengers = (uscan_personal_p+uscan_bus_p+uscan_train_p)
elif 'uscan_personal_p' in globals():
    uscan_passengers = (uscan_personal_p)
elif 'uscan_bus_p' in globals():
    uscan_passengers = (uscan_bus_p)
elif 'uscan_train_p' in globals():
    uscan_passengers = (uscan_train_p)
    
if 'uscan_peds' in globals():
    uscan_pedestrians = (uscan_peds)

# US-Mexico
if globals() in ['usmex_personal_v', 'usmex_buses', 'usmex_trucks', 'usmex_trains']:
    usmex_vehicles = (usmex_personal_v+usmex_buses+usmex_trucks+usmex_trains)
elif 'usmex_personal_v' in globals():
    usmex_vehicles = (usmex_personal_v)
elif 'usmex_buses' in globals():
    usmex_vehicles = (usmex_buses)
elif 'usmex_trucks' in globals():
    usmex_vehicles = (usmex_trucks)
elif 'usmex_trains' in globals():
    usmex_vehicles = (usmex_trains)
    
if globals() in ['usmex_truck_con_full', 'usmex_truck_con_empty', 'usmex_rail_con_full', 'usmex_rail_con_empty']:
    usmex_equipment = (usmex_truck_con_full+usmex_truck_con_empty+usmex_rail_con_full+usmex_rail_con_empty)
elif 'usmex_truck_con_full' in globals():
    usmex_equipment = (usmex_truck_con_full)
elif 'usmex_truck_con_empty' in globals():
    usmex_equipment = (usmex_truck_con_empty)
elif 'usmex_rail_con_full' in globals():
    usmex_equipment = (usmex_rail_con_full)  
elif 'usmex_rail_con_empty' in globals():
    usmex_equipment = (usmex_rail_con_empty)
    
if globals() in ['usmex_personal_p', 'usmex_bus_p','usmex_train_p']:
    usmex_passengers = (usmex_personal_p+usmex_bus_p+usmex_train_p)
elif 'usmex_personal_p' in globals():
    usmex_passengers = (usmex_personal_p)
elif 'usmex_bus_p' in globals():
    usmex_passengers = (usmex_bus_p)
elif 'usmex_train_p' in globals():
    usmex_passengers = (usmex_train_p)
    
if 'usmex_peds' in globals():
    usmex_pedestrians = (usmex_peds)

# combine dataset by adding back in border, vehicles, equipment, passengers, and pedestrians 
def insert_column(border,lst,tpe):
    for i in lst:
            i.insert(0, border)
            i.insert(2, tpe)
    return lst

if 'uscan_vehicles' in globals():
    uscan_vehicles = insert_column("US-Canada",uscan_vehicles,"Vehicles")
    data = uscan_vehicles
if 'uscan_equipment' in globals():
    uscan_equipment = insert_column("US-Canada",uscan_equipment,"Equipment")
    data = data + uscan_equipment
if 'uscan_passengers' in globals():
    uscan_passengers = insert_column("US-Canada",uscan_passengers,"Passengers")
    data = data + uscan_passengers
if 'uscan_pedestrians' in globals():
    uscan_pedestrians = insert_column("US-Canada",uscan_pedestrians,"Pedestrians")
    data = data + uscan_pedestrians
if 'usmex_vehicles' in globals():
    usmex_vehicles = insert_column("US-Mexico",usmex_vehicles,"Vehicles")
    data = data + usmex_vehicles
if 'usmex_equipment' in globals():
    usmex_equipment = insert_column("US-Mexico",usmex_equipment,"Equipment")
    data = data + usmex_equipment
if 'usmex_passengers' in globals():
    usmex_passengers = insert_column("US-Mexico",usmex_passengers,"Passengers")
    data = data + usmex_passengers
if 'usmex_pedestrians' in globals():
    usmex_pedestrians = insert_column("US-Mexico",usmex_pedestrians,"Pedestrians")
    data = data + usmex_pedestrians

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

