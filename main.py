import joblib


count_vector = joblib.load('count.pkl')
model = joblib.load('model.pkl')

def process_input(X_test, count_vector):
    X_test2=count_vector.transform([X_test])
    X_test3=X_test2.toarray()
    return X_test3

text=input('Enter Message: ')

x= process_input(text, count_vector)
prediction=model.predict(x)[0]
if prediction == 1 :
    print('Fake News')
else :
    print('Genuine News')