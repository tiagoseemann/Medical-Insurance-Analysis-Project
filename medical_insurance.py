import csv 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
# 1. Turn the data into a list of lists
file_p = r"C:\Users\tiago\Desktop\ProjetosPy\python-portfolio-project-starter-files\insurance.csv"
data = pd.read_csv(file_p)

ages = data.age.tolist()
sexes = data.sex.tolist()
bmis = data.bmi.tolist()
childrens = data.children.tolist()
smokers= data.smoker.tolist() 
regions = data.region.tolist()
charges = data.charges.tolist()

class PatientsInfo:
    def __init__(self, PatientsAges, PatientsSexes, PatientsBmis, PatientsChildrens, PatientsSmokers, PatientsRegions, PatientsCharges):
        self.PatientsAges = PatientsAges
        self.PatientsSexes = PatientsSexes
        self.PatientsBmis = PatientsBmis
        self.PatientsChildrens = PatientsChildrens
        self.PatientsSmokers = PatientsSmokers
        self.PatientsRegions = PatientsRegions
        self.PatientsCharges = PatientsCharges
        self.model = None
        self.le_sex = LabelEncoder()
        self.le_smoker = LabelEncoder()
        self.le_region = LabelEncoder()
    def analyze_ages(self):
        total_age = 0
        for age in self.PatientsAges:
            total_age += int(age)
        return ("avg patient age = ", round(total_age / len(self.PatientsAges), 2))
    def analyze_sexes(self):
        males = 0
        females = 0
        for sex in self.PatientsSexes:
            if sex == "male":
                males += 1
            elif sex == "female":
                females += 1
        print("number of males: " + str(males))
        print("number of females: " + str(females))
    def unique_regions(self):
        unique_regions = []
        for region in self.PatientsRegions:
            if region not in unique_regions:
                unique_regions.append(region)
        return unique_regions
    def average_charges(self):
        total_charges = 0
        for charge in self.PatientsCharges:
            total_charges += float(charge)
        return ("average insurance charge: " + str(round(total_charges / len(self.PatientsCharges), 2)) + " usd")
    def create_dictionary(self):
        self.PatientsDict = {}
        self.PatientsDict["age"] = [int(age) for age in self.PatientsAges]  
        self.PatientsDict["sex"] = self.PatientsSexes
        self.PatientsDict["bmi"] = self.PatientsBmis
        self.PatientsDict["children"] = self.PatientsChildrens
        self.PatientsDict["smoker"] = self.PatientsSmokers
        self.PatientsDict["regions"] = self.PatientsRegions
        self.PatientsDict["charges"] = self.PatientsCharges
        return self.PatientsDict
    
    def train_test(self):
        self.PatientsSexes = self.le_sex.fit_transform(self.PatientsSexes)
        self.PatientsSmokers = self.le_smoker.fit_transform(self.PatientsSmokers)
        self.PatientsRegions = self.le_region.fit_transform(self.PatientsRegions)

        features = pd.DataFrame({
            "age": self.PatientsAges,
            "sex": self.PatientsSexes,
            "bmi": self.PatientsBmis,
            "children": self.PatientsChildrens,
            "smoker": self.PatientsSmokers,
            "regions": self.PatientsRegions,
            "charges": self.PatientsCharges
        })

        target = features["charges"]
        X_train, X_test, y_train, y_test = train_test_split(features.drop("charges", axis=1), target, test_size=0.2, random_state=42)

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        print("training successful")
        return X_test, y_test

    def predict_charges(self, new_patient_data):
        if self.model == None:
            print("no model trained yet")  
            return None 
        
        new_patient_data['sex'] = self.le_sex.transform(new_patient_data['sex'])
        new_patient_data['smoker'] = self.le_smoker.transform(new_patient_data['smoker'])
        new_patient_data['regions'] = self.le_region.transform(new_patient_data['regions'])
        
        predicted_charge = self.model.predict(new_patient_data)
        
        return predicted_charge
    def evaluate_model(self, X_test, y_test):
        if self.model == None:
            print("no model trained yet")  
            return None 
        y_test_predicted = self.model.predict(X_test)
        r2 = r2_score(y_test, y_test_predicted)
        mse = mean_squared_error(y_test, y_test_predicted)
        return ("r2: " + str(r2), "mse: " + str(mse))

patientInfo = PatientsInfo(ages, sexes, bmis, childrens, smokers, regions, charges)
patientInfo.train_test()
            
#print(patientInfo.analyze_ages())
#print(patientInfo.analyze_sexes())
#print(patientInfo.unique_regions())
#print(patientInfo.average_charges())
#print(patientInfo.create_dictionary())
new_patient_data = pd.DataFrame({
    "age": [65],
    "sex": ["male"],
    "bmi": [23.5],
    "children": [0],
    "smoker": ["yes"],
    "regions": ["southwest"]
})

predicted_charge = patientInfo.predict_charges(new_patient_data)
print("Predicted Insurance Charge for the new patient:", predicted_charge)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Convert the lists to a DataFrame
df = pd.DataFrame({
    'ages': ages,
    'sexes': sexes,
    'bmis': bmis,
    'childrens': childrens,
    'smokers': smokers,
    'regions': regions,
    'charges': charges
})

# Create a LabelEncoder object
le = LabelEncoder()

# Convert the categorical variables to numerical values
df['sexes'] = le.fit_transform(df['sexes'])
df['smokers'] = le.fit_transform(df['smokers'])
df['regions'] = le.fit_transform(df['regions'])

# Create a correlation matrix
corr = df.corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.show()

# Plot scatter plots for 'charges' vs other numerical variables
plt.figure(figsize=(10, 8))
plt.scatter(df['ages'], df['charges'])
plt.xlabel('Ages')
plt.ylabel('Charges')
plt.title('Charges vs Ages')
plt.show()