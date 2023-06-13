import pandas as pd

columns_names = "age,workclass,fnlwgt,education,educational-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country,income".split(",")

f = open("adult.test", "r")
lines = f.readlines()

# remove first line and last line
lines = lines[1:]
lines = lines[:-1]

# remove whitespace
lines = [line.replace(" ", "") for line in lines]

# loading data into pandas dataframe
df = pd.DataFrame([line.split(",") for line in lines], columns=columns_names)

# Training data
f = open("adult.data", "r")
lines = f.readlines()

# remove last line
lines = lines[:-1]

# remove whitespace
lines = [line.replace(" ", "") for line in lines]

# loading data into pandas dataframe
df = pd.concat([df, pd.DataFrame([line.split(",") for line in lines], columns=columns_names)])

df["income"] = df["income"].apply(lambda x: x.replace("\n", ""))
df["income"] = df["income"].apply(lambda x: x.replace(".", ""))

df.reset_index(drop=True, inplace=True)

df.to_parquet("adult.parquet", index=False)
print("File saved to adult.parquet")