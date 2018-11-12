#Justin Klemperer
#INFO-3401
#Problem Set 9 & 10

#Collaborators: Steven, Lucas, Harold, Zach, Marissa


                                        #Monday:
#Imports
import csv
import pandas as pd
import numpy as np
import parser
import matplotlib
import matplotlib.pyplot as plt

#sklearn imports:
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score


#####PART A#####

class AnalysisData:
    def __init__(self):
        
        self.dataset = []
        self.variables = []     
    def parserFile(self, candy_file):
        
        self.dataset = pd.read_csv(candy_file)
        for column in self.dataset.columns.values:
            if column != "competitorname":
                
                self.variables.append(column)
            
            
#####PART B#####

class LinearAnalysis:
    def __init__(self, data_targetY):
        
        self.bestX = ""
        self.targetY = data_targetY
        self.fit = ""
        
    def runSimpleAnalysis(self, data):
        
        highest_sugar = ""
        highest_r2 = -1
        
        for column in data.variables:
            if column != self.targetY:
                
                data_variable = data.dataset[column].values
                data_variable = data_variable.reshape(len(data_variable),1)
                
                regr = LinearRegression()
                regr.fit(data_variable, data.dataset[self.targetY])
                variable_prediction = regr.predict(data_variable)
                r_score = r2_score(data.dataset[self.targetY],variable_prediction)
                
                if r_score > highest_r2:
                    
                    highest_r2 = r_score
                    highest_sugar = column
        self.bestX = highest_sugar
        print(highest_sugar, highest_r2)
        
    
#####PART C#####

class LogisticAnalysis:
    def __init__(self, data_targetY):
        self.bestX = ""
        self.targetY = data_targetY
        self.fit = ""


#####PROBLEM 1#####

candy_data_curation = AnalysisData()
candy_data_curation.parserFile('candy-data.csv')

#####PROBLEM 2#####
    #Attached in B & C
    
    
---------

#####PROBLEM SET 10#####

                                        #MONDAY & WEDNESDAY

class LogisticAnalysis:
    def __init__(self, data_targetY):
        
        self.bestX = ""
        self.targetY = data_targetY
        self.fit = -1
        
    def runSimpleAnalysis1(self, data):
        
        highest_sugar = data.dataset
        highest_r2 = -1
        
        for column in data.variables:
            if column != self.targetY:
                
                data_variable = data.dataset[column].values
                data_variable = data_variable.reshape(len(data_variable),1)
                
                regr = LinearRegression()
                regr.fit(data_variable, data.dataset[self.targetY])
                variable_prediction = regr.predict(data_variable)
                r_score = r2_score(data.dataset[self.targetY],variable_prediction)
                if r_score > highest_r2:
                    
                    highest_r2 = r_score
                    highest_sugar = column
        self.bestX = highest_sugar
        print(highest_sugar, highest_r2)
        
        print('Simple Logistic Regression Analysis coefficients: ', regr.coef_)
        print('Simple Logistic Regression Analysis intercept: ', regr.intercept_)
        


        
    def runMultipleRegression(self, data):
        
            multiple_reg = LogisticRegression()
            mp_r = [val for val in data.variables if val != self.targetY]
            multiple_reg.fit(data.dataset[mp_r], data.dataset[self.targetY])
            variable_prediction = multiple_reg.predict(data.dataset[mp_r])
            r_score = r2_score(data.dataset[self.targetY],variable_prediction)
            
            
            print("fruity", r_score)
            print('Multiple Regression Analysis coefficients: ', multiple_reg.coef_)
            print('Multiple Regression Analysis intercept: ', multiple_reg.intercept_)

    
  #### 1 ####      
        

candy_analysis = AnalysisData()

candy_analysis.parserFile("candy-data.csv")

### Linear Analysis ###
candy_data_analysis = LinearAnalysis("sugarpercent")

candy_data_analysis.runSimpleAnalysis(candy_analysis)


### Logistic Analysis ###
candy_data_analysis = LogisticAnalysis("chocolate")

candy_data_analysis.runSimpleAnalysis1(candy_analysis)

#"Price Percent at 0.10870."
#"Fruity at 0.55015" -----> this is the best to use when fitting this data

    ### 2 ###

candy_data_analysis = LogisticAnalysis("chocolate")

candy_data_analysis.runMultipleRegression(candy_analysis)

#Multiple analysis model outperforms the simple analysis, as multiple analysis out value was .760 and will run all the data instead of doing it step by step through colums and variables like how simple analysis does. This overall creates a more accurate fit

    ### 3 ### #(recieved help from Steven)

#Linear Regression    ------>  y = b0 + b1x
#Logistic Regression  ------>  p = 1/1+e^-(b0+b1x)
#Multiple Regression  ------>  p = 1/1+e^-(b0+b1x+b2x+b3x+....b11x)

#Linear Regression Equation   -----> y = 0.257063291665 + 0.00440378
#Logistic Regression Equation -----> p = 1/1+e^-(-0.650265328323 + 0.02157451x)
#Multiple Regression Equation -----> p = 1/1+e^-(-1.68260553 + -2.52858047 + -0.19697876 + 0.03940308 -0.16539952 + 0.49783674 + -0.47591613 0.81511886 + -0.59971553 + -0.2581028 + 0.3224988 + 0.05387906)


                                        #FRIDAY

    ### 4 ###
#A
    #The independent variable represents the different types of candies that are possible,and this is a categorical type, while the dependent variable is the sugar percentage, and this is a continious type. The Null hyptohesis would tell us candies with caramel or chocolate produce the same amount of sugar percentages. Basically meaning that there would be no significant differntiation between the two variables and the output.
    
#B
    #The dependent variable is the split ticket voters, and it is continious, while the independent variables would be the blue or red states, and this is categorical. The null hypothesis would be that blue and red state generate the same amount of the split ticket voer outcomes.

#C 
    #The independent variable would be the overall duration of the phone battery life span, and it would be continious. The dependent variable would be the different phone sale rates, and that would be continious. The null hyptohes would be that regardless of short or long battern life spans, the sales rate for both would be indistinguishable.