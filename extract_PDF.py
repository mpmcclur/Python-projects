"""
Reading in PDF data
"""

from tika import parser
import glob
import pandas as pd


"""Step 1:
    Set the working directory to the location of the PDFs and execute the 'location' variable. Also, save a blank TXT file in this location and name it "blank".
"""

location = '*.pdf'
pdfs = glob.glob(location)



"""Step 2:
    Execute all the defined functions below.
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
    data.to_csv('PayStub_Data.csv',index=False)




"""Step 3:
    Execute the following function to export information from PDFs to a concatenated CSV file.
"""
extractmultiple()




"""Step 4 (Optional)"
    To add a new paystub to the CSV file, type in the filename for the variable 'file'. Then execute all the code below.
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
