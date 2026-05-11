## Important Libraries--
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

## Read the data--
df = pd.read_csv(r"C:\Users\anand\PrimeBatch\Data\DataSets\loan_approval_data.csv")

## Info of data--
df.info()
df.head()
df.shape

### EDA--
### Handel missing values--

# Dividing into two catagoris-- numerical, catagorical
catagorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns

# Filling the Null values--- 
from sklearn.impute import SimpleImputer

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

ctg_imp = SimpleImputer(strategy="most_frequent")
df[catagorical_cols]= ctg_imp.fit_transform(df[catagorical_cols])

## Anlysis of data with different aspect--

## Credit Score analysis--
plt.figure(figsize=(11,4))
plt.subplot(1, 2, 1)
sns.histplot(
    data = df,
    x = "Credit_Score", 
    bins = 20,
    hue = "Loan_Approved",
    multiple = "dodge"
    
)

plt.subplot(1, 2, 2)
sns.histplot(
    data = df,
    x = "Collateral_Value",
    bins = 20,
    hue = "Loan_Approved",
    multiple = "dodge"
)

## Income analysis--
plt.figure(figsize=(11,4))

plt.subplot(1,2,1)
sns.histplot(
    data = df,
    x = "Applicant_Income",
    bins = 20
)

plt.subplot(1,2,2)
sns.histplot(
    data = df,
    x = "Coapplicant_Income",
    bins = 20
)

## Employ status--
emp_stat = df["Employment_Status"].value_counts()
plt.figure(figsize=(11, 4))

plt.subplot(1, 2, 1)
bar = sns.barplot(x= emp_stat.index, y= emp_stat.values)
bar.bar_label(bar.containers[0])
plt.title("Employment Status Count")

plt.subplot(1, 2, 2)
plt.pie( emp_stat, labels=["Salaried", "Contract", "Self-employed", "Unemployed"], autopct= "%1.1f%%")
plt.title("Employment Status")


plt.tight_layout()
plt.show()

## Finding the outlier--
plt.figure(figsize=(11, 8)) # Set the overall size once

plt.subplot(2, 2, 1)
sns.boxplot(data =df,x = "Loan_Approved",y = "Applicant_Income")

plt.subplot(2, 2, 2)
sns.boxplot(data =df,x = "Loan_Approved",y = "Credit_Score")

plt.subplot(2, 2, 3)
sns.boxplot(data =df,x = "Loan_Approved",y = "DTI_Ratio")

plt.subplot(2, 2, 4)
sns.boxplot(data =df,x = "Loan_Approved",y = "Loan_Amount")

plt.tight_layout() # Keeps them from overlapping
plt.show()

## Balencing(data distribution) check--
class_val_count = df["Loan_Approved"].value_counts()
plt.pie(class_val_count, labels=["No","Yes"],autopct="%1.1f%%")
plt.title("Lone Approved or not")

## Property area--
Loan_Purpose_count = df["Property_Area"].value_counts()
plt.figure(figsize=(11,4)) # Set the overall size once

# --- Plot 1 (Left Side) ---
plt.subplot(1, 2, 1) # (1 row, 2 columns, index 1)
ax1 = sns.barplot(x=Loan_Purpose_count.index, y=Loan_Purpose_count.values)
ax1.bar_label(ax1.containers[0])
plt.title("Property_Area Count")

# --- Plot 2 (Right Side) ---
plt.subplot(1, 2, 2) # (1 row, 2 columns, index 2)
plt.pie(Loan_Purpose_count, labels=Loan_Purpose_count.index, autopct="%1.1f%%")
plt.title("Property_Area Percentage%")

plt.tight_layout() # Keeps them from overlapping
plt.show()

## Loan purpose--
Loan_Purpose_count = df["Loan_Purpose"].value_counts()
plt.figure(figsize=(11, 4)) # Set the overall size once

# --- Plot 1 (Left Side) ---
plt.subplot(1, 2, 1) # (1 row, 2 columns, index 1)
ax1 = sns.barplot(x=Loan_Purpose_count.index, y=Loan_Purpose_count.values)
ax1.bar_label(ax1.containers[0])
plt.title("Loan Purpose Count")

# --- Plot 2 (Right Side) ---
plt.subplot(1, 2, 2) # (1 row, 2 columns, index 2)
plt.pie(Loan_Purpose_count, labels=Loan_Purpose_count.index, autopct="%1.1f%%")
plt.title("Loan Purpose Percentage%")

plt.tight_layout() # Keeps them from overlapping
plt.show()

## Loan Appruval distribution--
plt.figure(figsize=(11, 4)) # Set the overall size once

# --- Plot 2 (Right Side) ---
## Balencing(data distribution) check--
plt.subplot(1, 2, 1) # (1 row, 2 columns, index 1)
class_val_count = df["Gender"].value_counts()
plt.pie(class_val_count, labels=["Male","FeMale"],autopct="%1.1f%%")
plt.title("Lone Approved or not")

# --- Plot 2 (Right Side) ---
plt.subplot(1, 2, 2) # (1 row, 2 columns, index 2)
class_val_count = df["Gender"].value_counts()
ax = sns.barplot(class_val_count)
ax.bar_label(ax.containers[0])

plt.tight_layout() # Keeps them from overlapping
plt.show()

## Remove the unsecessary columns--
df = df.drop(columns=["Applicant_ID"])

### Encoding--
# rutein check-
df.info()
df.head()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
## Label Encoding--
le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

## One Hot Encoding--
col = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
oneHot = OneHotEncoder(drop="first",sparse_output=False, handle_unknown= "ignore")

encode = oneHot.fit_transform(df[col])
encode_df = pd.DataFrame(encode, columns= oneHot.get_feature_names_out(col), index=df.index)
df = pd.concat([df.drop(columns=col), encode_df], axis = 1)


## Corelation check--
num_col = df.select_dtypes(include="number")
num_matrix = num_col.corr()

plt.figure(figsize=(15,8))
sns.heatmap(
    num_matrix,
    annot = True,
    fmt = ".2f",
    cmap = "coolwarm"
    
)

## Checking the correlation values with Loan_Approved--
num_col.corr()["Loan_Approved"].sort_values(ascending=False)

### Train Test Split--
## Divide the data into input/output section--
X= df.drop(columns="Loan_Approved")
y= df["Loan_Approved"]
## Train Test split--
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 42)


## Scaling the data--
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


## Model making -- logistic Regression, KNN, Naive Bayes

## Logistic Regration Model---
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=20000)
model.fit(X_train_scaled, y_train)

## Predict the output--
y_pred = model.predict(X_test_scaled)

## Evolution of Logistic Regression Model--
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, f1_score, recall_score, confusion_matrix
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100,"%" )
print("Precision Score:",precision_score(y_test, y_pred)*100,"%")
print("Recall Score:", recall_score(y_test, y_pred)*100,"%")
print("MSE Score", mean_squared_error(y_test, y_pred)*100,"%")
print("F1 Score:", f1_score(y_test, y_pred)*100,"%")
print("CMatrix", confusion_matrix(y_test, y_pred))


## KNN Model---
from sklearn.neighbors import KNeighborsClassifier
nebor = KNeighborsClassifier(n_neighbors=7)
nebor.fit(X_train_scaled, y_train)

## Predict the output--
y_pred = nebor.predict(X_test_scaled)

## Evolution of Logistic Regression Model--
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, confusion_matrix,f1_score, recall_score
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100,"%" )
print("Precision Score:",precision_score(y_test, y_pred)*100,"%")
print("Recall Score:", recall_score(y_test, y_pred)*100,"%")
print("MSE Score", mean_squared_error(y_test, y_pred)*100,"%")
print("F1 Score:", f1_score(y_test, y_pred)*100,"%")
print("CMatrix", confusion_matrix(y_test, y_pred))

## Cross Validation for Hyperperameter tuning using GridSearchCV---
from sklearn.model_selection import GridSearchCV

classifire = KNeighborsClassifier()
param_grid ={ "n_neighbors":[3,5,7,9]}
grdCV = GridSearchCV(
    classifire,
    param_grid,
    cv=5,
    scoring= "precision"
)


grdCV.fit(X_train_scaled, y_train)
y_pred = grdCV.predict(X_test_scaled)

## Evolution of Logistic Regression Model--
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, mean_squared_error, f1_score, recall_score
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100,"%" )
print("Precision Score:",precision_score(y_test, y_pred)*100,"%")
print("Recall Score:", recall_score(y_test, y_pred)*100,"%")
print("MSE Score", mean_squared_error(y_test, y_pred)*100,"%")
print("F1 Score:", f1_score(y_test, y_pred)*100,"%")
print("CMatrix", confusion_matrix(y_test, y_pred))
print(grdCV.best_params_)


## NaiveBase Model---
from sklearn.naive_bayes import GaussianNB
nbmodel = GaussianNB()
nbmodel.fit(X_train_scaled, y_train)

## Predict the output--
y_pred = nbmodel.predict(X_test_scaled)


## Evolution of Logistic Regression Model--
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, f1_score, recall_score, confusion_matrix
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100,"%" )
print("Precision Score:",precision_score(y_test, y_pred)*100,"%")
print("Recall Score:", recall_score(y_test, y_pred)*100,"%")
print("MSE Score", mean_squared_error(y_test, y_pred)*100,"%")
print("F1 Score:", f1_score(y_test, y_pred)*100,"%")
print("CMatrix", confusion_matrix(y_test, y_pred))

## Feature Engineering--
## Adding more impat of that particular column
df["DTI_Ratio_sq"] = df["DTI_Ratio"]**2
df["Credit_Score_sq"] = df["Credit_Score"]**2

## Removingn the Squed(some values in a column is higher) data impact -- 
# df["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

## Spliting of data--
X = df.drop(columns=["Loan_Approved", "DTI_Ratio", "Credit_Score"])
y = df["Loan_Approved"]

## Train-Test-Split--
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

## Scalling the data--
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Again Traning different model and Test for better result---

## Logistic Regration Model---
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=20000)
model.fit(X_train_scaled, y_train)

## Predict the output--
y_pred = model.predict(X_test_scaled)

## Evolution of Logistic Regression Model--
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, f1_score, recall_score, confusion_matrix
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100,"%" )
print("Precision Score:",precision_score(y_test, y_pred)*100,"%")
print("Recall Score:", recall_score(y_test, y_pred)*100,"%")
print("MSE Score", mean_squared_error(y_test, y_pred)*100,"%")
print("F1 Score:", f1_score(y_test, y_pred)*100,"%")
print("CMatrix", confusion_matrix(y_test, y_pred))

## Cross Validation for Hyperperameter tuning using GridSearchCV---
from sklearn.model_selection import GridSearchCV

classifire = KNeighborsClassifier()
param_grid ={ "n_neighbors":[3,5,7,9]}
grdCV = GridSearchCV(
    classifire,
    param_grid,
    cv=5,
    scoring= "precision"
)


grdCV.fit(X_train_scaled, y_train)
y_pred = grdCV.predict(X_test_scaled)

## Evolution of Logistic Regression Model--
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, mean_squared_error, f1_score, recall_score
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100,"%" )
print("Precision Score:",precision_score(y_test, y_pred)*100,"%")
print("Recall Score:", recall_score(y_test, y_pred)*100,"%")
print("MSE Score", mean_squared_error(y_test, y_pred)*100,"%")
print("F1 Score:", f1_score(y_test, y_pred)*100,"%")
print("CMatrix", confusion_matrix(y_test, y_pred))
print(grdCV.best_params_)

## NaiveBase Model---
from sklearn.naive_bayes import GaussianNB

nbmodel = GaussianNB()
nbmodel.fit(X_train_scaled, y_train)

## Predict the output--
y_pred = nbmodel.predict(X_test_scaled)


## Evolution of Logistic Regression Model--
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, f1_score, recall_score, confusion_matrix
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100,"%" )
print("Precision Score:",precision_score(y_test, y_pred)*100,"%")
print("Recall Score:", recall_score(y_test, y_pred)*100,"%")
print("MSE Score", mean_squared_error(y_test, y_pred)*100,"%")
print("F1 Score:", f1_score(y_test, y_pred)*100,"%")
print("CMatrix", confusion_matrix(y_test, y_pred))