# Mean and Variance of Student Marks

marks = [80, 75, 90, 85, 70]

# Mean
mean = sum(marks) / len(marks)

# Variance
variance = sum((x - mean) ** 2 for x in marks) / len(marks)

print("Marks:", marks)
print("Mean:", mean)
print("Variance:", variance)
