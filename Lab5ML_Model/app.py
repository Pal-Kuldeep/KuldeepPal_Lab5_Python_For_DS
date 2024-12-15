from flask import Flask, render_template, request
import pickle
import numpy as np 
import sklearn


model = pickle.load(open('car_data_pickle.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def main():
    return render_template('Index.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    arr = np.array([[data1, data2, data3, data4,data5]])
    pred = model.predict(arr)
    return render_template('output.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)

