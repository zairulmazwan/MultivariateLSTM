import pandas as pd

# df = pd.DataFrame(
#     {"date" : ["28-04-1983", "01-04-2021"]}
# )

df = pd.DataFrame(
    {"date" : ["28/04/1983", "01/04/2021"]}
)

print(df)
#df['DOB1'] = df['DOB'].dt.strftime('%m/%d/%Y')
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
#df["date"] = pd.to_datetime(df.date).dt.strftime('%d-%m-%Y')
#df["date"] = df["date"].dt.strftime('%d-%m-%Y')
print("After date format : ",df)
yearOfBirth = df.iloc[0,0].year


import datetime
today = datetime.datetime.now()
thisYear = today.year

print(yearOfBirth)
print(thisYear)

print("Your age : ", thisYear-yearOfBirth)
