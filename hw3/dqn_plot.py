import pickle

with open("874e145c-f423-49f0-8baa-7d118d7bc2f5.pkl", 'rb') as f:
	vanilla = pickle.load(f)
print(vanilla)

with open("932235fb-5da6-44f8-808b-9fe051bc0bf3.pkl", 'rb') as f:
	double = pickle.load(f)
print(double)
