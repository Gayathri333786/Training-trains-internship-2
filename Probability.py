# Email Spam Detection using Probability

spam_emails = 300
total_emails = 1000

# Probability of an email being spam
probability_spam = spam_emails / total_emails

print("Probability of Spam Email:", probability_spam)

# Predict based on probability
if probability_spam > 0.5:
    print("Most emails are Spam")
else:
    print("Most emails are Not Spam")
