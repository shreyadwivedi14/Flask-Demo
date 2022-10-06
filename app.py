#from crypt import methods
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features=[float(x) for x in request.form.values()]
    final=[int_features]
    # print("##########1: ",int_features)
    # print("##########2: ",final)
    prediction=model.predict(np.array(final))
    print("##########3: ",prediction)
    output= str(prediction[0])
    return output


if __name__ == '__main__':
    app.run(debug=True)