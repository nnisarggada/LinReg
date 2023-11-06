import csv
import random

file_path = 'data.csv'

years_experience = []
age = []
salary = []

with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        years_experience.append(float(row[0]))
        age.append(float(row[1]))
        salary.append(float(row[2]))

data = list(zip(years_experience, age, salary))

train_percent = 0.9
test_percent = 0.1

random.shuffle(data)
split_index = int(len(data) * train_percent)
train_data = data[:split_index]
test_data = data[split_index:]

def predict_salary(years_experience, age, theta0, theta1, theta2):
    return theta0 + theta1 * years_experience + theta2 * age

def mean_squared_error(predictions, targets):
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(targets)

theta0 = random.uniform(0, 1)
theta1 = random.uniform(0, 1)
theta2 = random.uniform(0, 1)

learning_rate = 0.001
num_iterations = 1000

for _ in range(num_iterations):
    predictions = [predict_salary(x[0], x[1], theta0, theta1, theta2) for x in train_data]
    error = [p - x[2] for p, x in zip(predictions, train_data)]
    theta0 -= learning_rate * sum(error) / len(train_data)
    theta1 -= learning_rate * sum(error[i] * train_data[i][0] for i in range(len(train_data))) / len(train_data)
    theta2 -= learning_rate * sum(error[i] * train_data[i][1] for i in range(len(train_data))) / len(train_data)

test_predictions = [predict_salary(x[0], x[1], theta0, theta1, theta2) for x in test_data]
test_targets = [x[2] for x in test_data]

def mean_absolute_percentage_error(predictions, targets):
    error_sum = sum(abs(p - t) / t for p, t in zip(predictions, targets))
    return (error_sum / len(targets)) * 100

mape = mean_absolute_percentage_error(test_predictions, test_targets)

print()
print(f"Accuracy: {100-mape:.2f}%")
print()

while(True):
    input_experience = float(input("Enter Years of Experience: "))
    input_age = float(input("Enter Age: "))

    predicted_salary = predict_salary(input_experience, input_age, theta0, theta1, theta2)
    print()
    print(f"Predicted Salary: {predicted_salary:.2f}")
    print('\n')
