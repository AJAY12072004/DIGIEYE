import pandas as pd

from ydata_profiling import ProfileReport


df = pd.read_csv("digital-eye.csv")
print(df)
df.drop(['Name'], axis=1, inplace=True)
df.fillna(method='ffill', inplace=True) 
df = df.drop_duplicates()
print(df)
profile = ProfileReport(df)
profile.to_file(output_file="digital-eye.html")

