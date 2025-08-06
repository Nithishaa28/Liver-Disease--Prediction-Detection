import os
import MySQLdb
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
 
 
from werkzeug.utils import secure_filename
import numpy as np
import joblib
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from database import *
app = Flask(__name__)
app.secret_key='detection'
 
app.config['UPLOAD_FOLDER'] = 'static/uploads'
 


selected_features = ['Age','Gender_Male','Total_Bilirubin','Direct_Bilirubin',
                      'Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']
 


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/menu2")
def menu2():
    return render_template("menu2.html")

@app.route("/signupa")
def signupa():
    return render_template("signup.html")

@app.route("/Doctor")
def Doctor():
    return render_template("doctor.html")

@app.route("/rec")
def rec():
    return render_template("rec.html")

@app.route("/logina")
def logina():
    return render_template("login.html")


@app.route("/signup",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        status = user_reg(username,email,password)
        if status == 1:
            return render_template("/login.html")
        else:
            return render_template("/signup.html",m1="failed")        
    

@app.route("/login",methods=['POST','GET'])
def login():
    if request.method=='POST':
        email=request.form['email']
        password=request.form['password']
        status = user_loginact(request.form['email'], request.form['password'])
        print(status)
        if status == 1:                                      
            return render_template("/menu.html", m1="sucess")
        else:
            return render_template("/login.html", m1="Login Failed")
             
 
@app.route('/predict')
def predictpage():
    return render_template('predict.html', selected_features=selected_features)    
 


app.static_folder = 'static'

    
# @app.route("/")
# def home():
#     return render_template("index.html")
    

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        form_values_list = list(request.form.values())  
        Age = int(form_values_list[0])
        Gender_Male = str(form_values_list[1])
        Total_Bilirubin = float(form_values_list[2])
        Direct_Bilirubin = float(form_values_list[3])
        Alkaline_Phosphotase = int(form_values_list[4])
        Alamine_Aminotransferase = int(form_values_list[5])
        Aspartate_Aminotransferase = int(form_values_list[6])
        Total_Protiens = float(form_values_list[7])
        Albumin = float(form_values_list[8])
        Albumin_and_Globulin_Ratio = float(form_values_list[9])
        vector = np.vectorize(np.float)
        check = np.array([Age,Gender_Male,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]).reshape(1, -1)

        Labe=LabelEncoder()
        check[:,1]=Labe.fit_transform(check[:,1])
        model_path = os.path.join(os.path.dirname(__file__), 'liVERFCMODEL.sav')    
        check = vector(check)
        
        #print(X_test[0:1])
        #print(check[[0]] )
        clf = joblib.load(model_path)
        B_pred = clf.predict(check[[0]])
        if B_pred == 2:
            result1="lIVER DISEASE DETECTED"
            print("Liver Disease Detected")
        if B_pred == 1:
            result1="NO DISEASE DETECTED"
            print("Patient is Healthy")
       # result = predict_pcos(request.form.values())
       # treatment_tip = get_treatment_tip(result)
        return render_template('predict.html',result=result1)
    except Exception as e:
         return render_template('predict.html', error=str(e))

@app.route('/prediction1', methods=['GET', 'POST'])
def predict1():
    if request.method == 'POST':
        # Get the image file from the request
        image_file = request.files['image']
        print("ddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
        print(image_file)
        filename = secure_filename(image_file.filename)
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Read the image using Pillow
        image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Preprocess the image
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        classes = {0:"notumour",1:"tumour"}
     
        # load the model we saved
        model = load_model('tumour.h5')
      
        result = np.argmax(model.predict(image))
        print(result)       
        prediction1 = classes[result]
        print(prediction1)

        # Render the HTML template with the prediction result and image
        return render_template('rec.html', prediction=prediction1)


if __name__ == "__main__":
    app.run(debug=True)