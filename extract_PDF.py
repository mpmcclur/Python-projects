"""
Reading in PDF data
"""

from tika import parser
import glob
import pandas as pd

"""This program is designed to import Thorlabs' paystub PDF files from Paychex, extract specific information,
   and then export that information in a CSV file. This program takes old- and new-style paystubs. During testing,
   this program pulled the correct information ~95% of the time.
   
   Follow the steps below.
"""


"""Step 1:
    Ensure your old paystubs and your new paystubs are in two different folders.
    IMPORTANT: Set the working directory to the location of the old paystubs and execute the 'location' and 'pdfs' variables.
"""

location = '*.pdf' # Do not edit this; set your directory within Python or IDE
pdfs = glob.glob(location)



"""Step 2:
    Execute the function below to read in all old paystubs and export to a CSV.
"""

def extractmultiple_OldFormat():
    dflist = []
    for file in pdfs:
        raw = parser.from_file(file)
        raw = raw["content"]
        with open('blank.txt','w') as target:
            target.write(raw)
            file = open("blank.txt", "r")
        pdfline = [ ]
        for line in file:
            textline = line.strip()
            # split the line on whitespace
            items = textline.split()
            # add the list of items to the empty list
            pdfline.append(items)
        # data frame
        data = pd.DataFrame(pdfline).dropna(0,'all').reset_index(drop=True)
        data = data.drop(data.index[0:13]).reset_index(drop=True)
        new_names = {0:'col1', 1:'col2', 2:'col3', 3:'col4',4:'col5',5:'col6'}
        data = data.rename(index=str, columns=new_names)
        # adjust for paystubs w/o PTO line
        for row in data:
            if data.iloc[0]['col1'] == 'Pay':
                break
            else:
                line = pd.DataFrame({"col1": "Pay"}, index=[0])
                data = pd.concat([data.iloc[:0], line, data.iloc[0:]],sort=True).reset_index(drop=True)
        data = data.drop(data.index[3:16]).reset_index(drop=True)
        for row in data:
            if data.iloc[9]['col1'] == 'DISCRETIONARY':
                break
            elif data.iloc[10]['col1'] == 'DISCRETIONARY':
                line = pd.DataFrame({"col1": "DISCRETIONARY"}, index=[10])
                data = pd.concat([data.iloc[:10], line, data.iloc[10:]],sort=True).reset_index(drop=True)
            else:
                line = pd.DataFrame({"col1": "DISCRETIONARY"}, index=[9])
                data = pd.concat([data.iloc[:9], line, data.iloc[9:]],sort=True).reset_index(drop=True)
        for row in data:
            if data.iloc[13]['col1'] == 'LTD':
                break
            elif data.iloc[14]['col1'] == 'LTD':
                line = pd.DataFrame({"col1": "LTD"}, index=[14])
                data = pd.concat([data.iloc[:14], line, data.iloc[14:]],sort=True).reset_index(drop=True)
            else:
                line = pd.DataFrame({"col1": "LTD"}, index=[13])
                data = pd.concat([data.iloc[:13], line, data.iloc[13:]],sort=True).reset_index(drop=True)
        for row in data:
            if data.iloc[14]['col1'] == 'NON':
                break
            elif data.iloc[15]['col1'] == 'NON':
                line = pd.DataFrame({"col1": "NON"}, index=[15])
                data = pd.concat([data.iloc[:15], line, data.iloc[15:]],sort=True).reset_index(drop=True)
            elif data.iloc[13]['col1'] == 'NON':
                line = pd.DataFrame({"col1": "NON"}, index=[13])
                data = pd.concat([data.iloc[:13], line, data.iloc[13:]],sort=True).reset_index(drop=True)
            else:
                line = pd.DataFrame({"col1": "NON"}, index=[14])
                data = pd.concat([data.iloc[:14], line, data.iloc[14:]],sort=True).reset_index(drop=True)
        for row in data:
            if data.iloc[15]['col1'] == 'TAXABLE':
                break
            elif data.iloc[14]['col1'] == 'TAXABLE':
                line = pd.DataFrame({"col1": "TAXABLE"}, index=[14])
                data = pd.concat([data.iloc[:14], line, data.iloc[14:]],sort=True).reset_index(drop=True)
            elif data.iloc[13]['col1'] == 'TAXABLE':
                line = pd.DataFrame({"col1": "TAXABLE"}, index=[13])
                data = pd.concat([data.iloc[:13], line, data.iloc[13:]],sort=True).reset_index(drop=True)
            else:
                line = pd.DataFrame({"col1": "TAXABLE"}, index=[15])
                data = pd.concat([data.iloc[:15], line, data.iloc[15:]],sort=True).reset_index(drop=True)
        data = data.reset_index(drop=True)
        # organize data from data variable into new df
        data = data[data.col1 != 'DEDUCTIONS']
        data = data[data.col1 != 'LTD']
        data = data[data.col1 != 'TOTAL']
        data = data[data.col1 != 'Happy']
        data = data[data.col1 != 'DESCRIPTION']
        data = data[data.col1 != 'DISCRETIONARY']
        data = data[data.col1 != 'CHECKING']
        data = data[data.col1 != '______']
        data = data[data.col1 != 'Net']
        data = data[data.col1 != '>']
        data = data[data.col1 != 'EARNINGS'].reset_index(drop=True)
        df1 = [data['col3'].iat[1],data['col1'].iat[27],data['col2'].iat[27],data['col3'].iat[13],data['col3'].iat[17],data['col4'].iat[20],data['col4'].iat[10],data['col4'].iat[11],data['col4'].iat[21],data['col4'].iat[22],data['col5'].iat[23],data['col4'].iat[6],data['col4'].iat[7]]
        # assign data to columns
        dflist.append(df1)
    data = pd.DataFrame(dflist,columns=['Date','Net Pay PER','Net Pay YTD','Tot Hours','Fed Tax','NJ Tax','401K Cont','401K Match','NJ Disability','NJ Unemploy','NJ EE Work Dev','CAF Dental','CAF Medical'])
    data.to_csv('PayStub_Data.csv',index=False)
extractmultiple_OldFormat()



"""Step 3:
    Drag your new CSV file and place it in the folder where your new paystubs are located.
    IMPORTANT: Change the working directory to the location of the new paystubs.
    and execute the 'location' and 'pdfs' variables.
"""
location = '*.pdf' # Do not edit this; set your directory within Python or IDE
pdfs = glob.glob(location)



"""Step 4:
    Execute the following function to add the new paystub info to your existing CSV file.
"""
def extractmultiple():
    dflist = []
    for file in pdfs:
        raw = parser.from_file(file)
        raw = raw["content"]
        with open('blank.txt','w') as target:
            target.write(raw)
            file = open("blank.txt", "r")
        pdfline = [ ]
        for line in file:
            textline = line.strip()
            # split the line on whitespace
            items = textline.split()
            # add the list of items to the empty list
            pdfline.append(items)
        # data frame
        data = pd.DataFrame(pdfline).dropna(0,'all')
        data = data.drop(data.index[3:35]).reset_index(drop=True)
        new_names = {0:'col1', 1:'col2', 2:'col3', 3:'col4',4:'col5',5:'col6'}
        data = data.rename(index=str, columns=new_names)
        # adjust for paystubs w/o PTO line
        for row in data:
            if data.iloc[20]['col1'] == 'Paid':
                line = pd.DataFrame({"col1": "Paid"}, index=[20])
                data = pd.concat([data.iloc[:20], line, data.iloc[20:]],sort=True).reset_index(drop=True)
            elif data.iloc[21]['col1'] == 'Paid':
                line = pd.DataFrame({"col1": "Paid"}, index=[21])
                data = pd.concat([data.iloc[:21], line, data.iloc[21:]],sort=True).reset_index(drop=True)
            elif data.iloc[19]['col1'] == 'Paid':
                line = pd.DataFrame({"col1": "Paid"}, index=[19])
                data = pd.concat([data.iloc[:19], line, data.iloc[19:]],sort=True).reset_index(drop=True)
            elif data.iloc[21]['col1'] == 'Paid':
                line = pd.DataFrame({"col1": "Paid"}, index=[21])
                data = pd.concat([data.iloc[:21], line, data.iloc[21:]],sort=True).reset_index(drop=True)
            else:
                line = pd.DataFrame({"col1": "Paid"}, index=[21])
                data = pd.concat([data.iloc[:21], line, data.iloc[21:]],sort=True).reset_index(drop=True)
        for row in data:
            if data.iloc[18]['col1'] == 'Retro':
                line = pd.DataFrame({"col1": "Retro"}, index=[18])
                data = pd.concat([data.iloc[:18], line, data.iloc[18:]],sort=True).reset_index(drop=True)
            elif data.iloc[17]['col1'] == 'Retro':
                line = pd.DataFrame({"col1": "Retro"}, index=[17])
                data = pd.concat([data.iloc[:17], line, data.iloc[17:]],sort=True).reset_index(drop=True)
            elif data.iloc[19]['col1'] == 'Retro':
                line = pd.DataFrame({"col1": "Retro"}, index=[19])
                data = pd.concat([data.iloc[:19], line, data.iloc[19:]],sort=True).reset_index(drop=True)
            else:
                line = pd.DataFrame({"col1": "Retro"}, index=[18])
                data = pd.concat([data.iloc[:18], line, data.iloc[18:]],sort=True).reset_index(drop=True)
        for row in data:
            if data.iloc[17]['col1'] == 'DiscretionaryCarry':
                line = pd.DataFrame({"col1": "DiscretionaryCarry"}, index=[17])
                data = pd.concat([data.iloc[:17], line, data.iloc[17:]],sort=True).reset_index(drop=True)
            elif data.iloc[16]['col1'] == 'DiscretionaryCarry':
                line = pd.DataFrame({"col1": "DiscretionaryCarry"}, index=[16])
                data = pd.concat([data.iloc[:16], line, data.iloc[16:]],sort=True).reset_index(drop=True)
            elif data.iloc[18]['col1'] == 'DiscretionaryCarry':
                line = pd.DataFrame({"col1": "DiscretionaryCarry"}, index=[18])
                data = pd.concat([data.iloc[:18], line, data.iloc[18:]],sort=True).reset_index(drop=True)
            else:
                line = pd.DataFrame({"col1": "DiscretionaryCarry"}, index=[17])
                data = pd.concat([data.iloc[:17], line, data.iloc[17:]],sort=True).reset_index(drop=True)
        # organize data from data variable into new df
        data = data[data.col1 != 'Paid']
        data = data[data.col1 != 'Retro']
        data = data[data.col1 != 'DiscretionaryCarry'].reset_index(drop=True)
        df1 = [data['col1'].iat[0],data['col3'].iat[44],data['col4'].iat[44],data['col4'].iat[21],data['col6'].iat[28],data['col6'].iat[29],data['col4'].iat[37],data['col4'].iat[24],data['col3'].iat[30],data['col3'].iat[31],data['col5'].iat[32],data['col3'].iat[35],data['col3'].iat[36]]
        # assign data to columns
        dflist.append(df1)
    data = pd.DataFrame(dflist,columns=['Date','Net Pay PER','Net Pay YTD','Tot Hours','Fed Tax','NJ Tax','401K Cont','401K Match','NJ Disability','NJ Unemploy','NJ EE Work Dev','CAF Dental','CAF Medical'])
    data.to_csv('PayStub_Data.csv', header=None, mode='a',index=False)
extractmultiple()


"""Step 5 (Optional)"
    To add a new paystub to the CSV file, one at a time, add the filename for the variable 'file' within the function below.
    Finally, execute all the code below, and a new row will be added to the file.
"""
def add_singlePDF():
    file = 'FILENAME.pdf'
    raw = parser.from_file(file)
    raw = raw["content"]
    raw = raw.strip('\n')
    with open('blank.txt','w') as target:
        target.write(raw)
        file = open("blank.txt", "r")
    pdfline = [ ]
    for line in file:
        textline = line.strip()
        # split the line on whitespace
        items = textline.split()
        # add the list of items to the empty list
        pdfline.append(items)
    # data frame
    data = pd.DataFrame(pdfline).dropna(0,'all')
    data = data.drop(data.index[3:35]).reset_index(drop=True)
    new_names = {0:'col1', 1:'col2', 2:'col3', 3:'col4',4:'col5',5:'col6'}
    data = data.rename(index=str, columns=new_names)
    for row in data:
        if data.iloc[20]['col1'] == 'Paid':
            line = pd.DataFrame({"col1": "Paid"}, index=[20])
            data = pd.concat([data.iloc[:20], line, data.iloc[20:]],sort=True).reset_index(drop=True)
        elif data.iloc[21]['col1'] == 'Paid':
            line = pd.DataFrame({"col1": "Paid"}, index=[21])
            data = pd.concat([data.iloc[:21], line, data.iloc[21:]],sort=True).reset_index(drop=True)
        elif data.iloc[19]['col1'] == 'Paid':
            line = pd.DataFrame({"col1": "Paid"}, index=[19])
            data = pd.concat([data.iloc[:19], line, data.iloc[19:]],sort=True).reset_index(drop=True)
        elif data.iloc[21]['col1'] == 'Paid':
            line = pd.DataFrame({"col1": "Paid"}, index=[21])
            data = pd.concat([data.iloc[:21], line, data.iloc[21:]],sort=True).reset_index(drop=True)
        else:
            line = pd.DataFrame({"col1": "Paid"}, index=[21])
            data = pd.concat([data.iloc[:21], line, data.iloc[21:]],sort=True).reset_index(drop=True)
    for row in data:
        if data.iloc[18]['col1'] == 'Retro':
            line = pd.DataFrame({"col1": "Retro"}, index=[18])
            data = pd.concat([data.iloc[:18], line, data.iloc[18:]],sort=True).reset_index(drop=True)
        elif data.iloc[17]['col1'] == 'Retro':
            line = pd.DataFrame({"col1": "Retro"}, index=[17])
            data = pd.concat([data.iloc[:17], line, data.iloc[17:]],sort=True).reset_index(drop=True)
        elif data.iloc[19]['col1'] == 'Retro':
            line = pd.DataFrame({"col1": "Retro"}, index=[19])
            data = pd.concat([data.iloc[:19], line, data.iloc[19:]],sort=True).reset_index(drop=True)
        else:
            line = pd.DataFrame({"col1": "Retro"}, index=[18])
            data = pd.concat([data.iloc[:18], line, data.iloc[18:]],sort=True).reset_index(drop=True)
    for row in data:
        if data.iloc[17]['col1'] == 'DiscretionaryCarry':
            line = pd.DataFrame({"col1": "DiscretionaryCarry"}, index=[17])
            data = pd.concat([data.iloc[:17], line, data.iloc[17:]],sort=True).reset_index(drop=True)
        elif data.iloc[16]['col1'] == 'DiscretionaryCarry':
            line = pd.DataFrame({"col1": "DiscretionaryCarry"}, index=[16])
            data = pd.concat([data.iloc[:16], line, data.iloc[16:]],sort=True).reset_index(drop=True)
        elif data.iloc[18]['col1'] == 'DiscretionaryCarry':
            line = pd.DataFrame({"col1": "DiscretionaryCarry"}, index=[18])
            data = pd.concat([data.iloc[:18], line, data.iloc[18:]],sort=True).reset_index(drop=True)
        else:
            line = pd.DataFrame({"col1": "DiscretionaryCarry"}, index=[17])
            data = pd.concat([data.iloc[:17], line, data.iloc[17:]],sort=True).reset_index(drop=True)
    # organize data from data variable into new df
    data = data[data.col1 != 'Paid']
    data = data[data.col1 != 'Retro']
    data = data[data.col1 != 'DiscretionaryCarry'].reset_index(drop=True)
    df1 = [data['col1'].iat[0],data['col3'].iat[44],data['col4'].iat[44],data['col4'].iat[21],data['col6'].iat[28],data['col6'].iat[29],data['col4'].iat[37],data['col4'].iat[24],data['col3'].iat[30],data['col3'].iat[31],data['col5'].iat[32],data['col3'].iat[35],data['col3'].iat[36]]
    cols = ['Date','Net Pay PER','Net Pay YTD','Tot Hours','Fed Tax','NJ Tax','401K Cont','401K Match','NJ Disability','NJ Unemploy','NJ EE Work Dev','CAF Dental','CAF Medical']
    data = pd.DataFrame(columns=cols)
    data.loc[0] = df1
    data.to_csv('PayStub_Data.csv', header=None, mode='a',index=False)
    
add_singlePDF()
