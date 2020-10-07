  # -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:14:52 2018

@author: Matt
"""

# Write a program in Python that prompts the user for two integers (a base and exponent).
# Calculate the result of taking the base to the power of the exponent, as in:

#Easy method
import math
int1 = int(input("Input your base. "))
int2 = int(input("Input your exponent "))
print("The result is "+ str(pow(int1,int2))+".")

#Loop method
base = input('Base: ') 
base = int(base) 
exponent =input('Exponent: ') 
exponent = int(exponent) 
base1 = 1 
if exponent >= 0: 
    for i in range(exponent): 
        base1 = base1 * base 
else: 
    for i in range(-exponent): 
        base1 = base1 * base 
    base1 = 1/base1 
print(base1)



'''Write a program in Python that prompts the user for an integer. Calculate the factorial for the integer the user provides. Use only logic that you provide'''
# Do not use 3rd-party code, functions or snippets (including looking up solutions online).
# Only use the 4 mathematical operators (+, -, x, /).
# Python code to demonstrate naive method
integer = input("Enter an integer. ")
num = 1
for i in range(1,int(integer)+1): 
    num = num * i
print("The factorial of ",integer," is : ",num,".")





#11/13/2018
# Display the results from the web page in a list in reverse alphabetical order. Be sure to display child nodes or attributes.
from bs4 import BeautifulSoup
link = "https://getpocket.com/explore/item/i-spent-2-years-cleaning-houses-what-i-saw-makes-me-never-want-to-be-rich-983414393"
html = request.urlopen(link).read().decode('utf8')
soup = BeautifulSoup(html, 'html.parser')
soup = soup.findAll('a')
sorted(soup, key=lambda elem: elem.text, reverse=True)




# Write a program in Python that calculate factorials. Create two functions that calculate the factorial for a given integer.
# One function should be recursive, while the other function should be iterative. Create a third function that calls both the
# iterative and recursive versions, compares the results, and raises an error if the results do not match.
# Call the third function 10 times for a range of integers from 1 to 10. 

#Three functions: 
    # 1. factorial_iter 

    # 2. factorial_recur 

    # 3. factorial 

# Use only logic that you provide – do not use 3rd-party code, functions or snippets (including looking up solutions online).
# Only use the 4 mathematical operators (+, -, x, /).

# iterative
def iterative(integer):
    num = 1
    for i in range(1,int(integer)+1): 
        num = num * i
    return num
print(iterative(5))
# another one
def recursiveFactorial(input): 
    if (input < 1): 
        return('Input is less than 1')     
    elif(input > 1): 
        return input * recursiveFactorial(input - 1) 
    else: return 1
    
# recursive
def factorial_recur(num):
    try: 
        assert type(num) == int 
        if num == 1: 
            return(1) 
        else: 
            return(num * factorial_recur(num-1)) 
    except: 
        print('Please enter an integer.')

# comparison function
def factorial(number): 
    assert iterative(integer) == factorial_recur(num), 'result of functions do not match' 
    return(iterative(integer))
    
for n in range(10): 
    print(factorial(n))
factorial(10)




# Create a function to extract and pull select information from a stock portfolio JSON file.
import pandas as pd
import urllib
import json
portfolio = pd.read_csv('Portfolio.csv')
url_start = 'https://api.iextrading.com/1.0/stock/'
url_end = '/quote'

def pull_stock_stuff(json):
        stock_d = {}
        stock_d['symbol'] = json['symbol']
        stock_d['latestprice'] = json['latestPrice']
        stock_d['change'] = json['change']
        stock_d['change_pct'] = json['changePercent']
        return(stock_d)
stock_list = []
for stock in portfolio['Symbol']:
    response = urllib.request.urlopen(url_start + stock + url_end)
    stock_response = json.loads(response.read())
    stock_list.append(pull_stock_stuff(stock_response))
#look at Jupyter file from Brandon



