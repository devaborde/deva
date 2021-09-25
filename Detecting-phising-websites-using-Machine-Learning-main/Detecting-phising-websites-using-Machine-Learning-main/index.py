import warnings
warnings.filterwarnings("ignore")

import joblib
import inputScript

classifier = joblib.load('random_forest.pkl')

print("enter url")
url = input()

checkprediction = inputScript.main(url)
prediction = classifier.predict(checkprediction)
print(prediction)

if(prediction == 1):
    print("The site is a phishing site")
else:
    print("The site is not a phishing site")
