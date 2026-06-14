import pandas as pd
df = pd.read_csv(r"C:\Users\drman\Downloads\Dataset 2 (1).csv")
#Q6average age of users
print("Average Age:", df["Age"].mean())

#Q7 average watch hours per week
print("Average Watch Hours Per Week:", df["WatchHoursPerWeek"].mean())

#Q8 average monthly spending of users
print("Average Monthly Spending:", df["MonthlySpend"].mean())

#Q9 number of users in each subscription category
print("\nUsers in Each Subscription Category:")
print(df["SubscriptionType"].value_counts())

#Q10 percentage of users who renewed their subscriptions
renewed_percentage = (df["SubscriptionRenewed"] == "Yes").mean() * 100
print("\nPercentage of Users Who Renewed:", renewed_percentage, "%")