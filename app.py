from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "secret123"

# ================= LOAD MODELS =================

rf_model = joblib.load("models/rf_model_proper.pkl")
crop_encoder = joblib.load("models/crop_encoder.pkl")
district_encoder = joblib.load("models/district_encoder.pkl")

lstm_model = load_model("models/lstm_yield_model.h5", compile=False)
lstm_scaler = joblib.load("models/lstm_scaler.pkl")

print("Models loaded successfully")

# ================= ROUTES =================

@app.route("/")
def home():
    return redirect(url_for("login"))


# LOGIN

@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        if email == "admin@gmail.com" and password == "1234":

            session["user"] = email
            return redirect(url_for("predict_ui"))

        else:
            return render_template("login.html",error="Invalid credentials")

    return render_template("login.html")


# SIGNUP

@app.route("/signup",methods=["GET","POST"])
def signup():

    if request.method=="POST":
        return redirect(url_for("login"))

    return render_template("signup.html")


# LOGOUT

@app.route("/logout")
def logout():

    session.pop("user",None)
    return redirect(url_for("login"))


# ================= PREDICTION =================

@app.route("/predict-ui",methods=["GET","POST"])
def predict_ui():

    if "user" not in session:
        return redirect(url_for("login"))

    prediction=None
    user_prediction=None
    best_crop=None
    best_yield=None
    season=None
    user_crop=None
    error=None

    crops=list(crop_encoder.classes_)
    districts=list(district_encoder.classes_)

    if request.method=="POST":

        try:

            district=request.form["District_Name"]
            crop=request.form["Crop"]
            month=int(request.form["Month"])
            year=int(request.form["Crop_Year"])
            area=float(request.form["Area"])

            user_crop=crop

            # season logic
            if month in [6,7,8,9]:
                season="Kharif"
            elif month in [10,11,12,1,2]:
                season="Rabi"
            else:
                season="Zaid"

            # encode

            crop_encoded=crop_encoder.transform([crop])[0]
            district_encoded=district_encoder.transform([district])[0]

            X=np.array([[crop_encoded,district_encoded,year,area]])

            # RF prediction

            # user_prediction=round(float(rf_model.predict(X)[0]),2)
            user_prediction = float(rf_model.predict(X)[0])
            user_prediction = round(user_prediction / area, 2)

            # best crop

            crop_predictions={}

            # for c in crop_encoder.classes_:

            #     c_encoded=crop_encoder.transform([c])[0]

            #     X_test=np.array([[c_encoded,district_encoded,year,area]])

            #     pred=round(float(rf_model.predict(X_test)[0]),2)

            #     crop_predictions[c]=pred
            for c in crop_encoder.classes_:
                c_encoded = crop_encoder.transform([c])[0]
                X_test = np.array([[c_encoded, district_encoded, year, area]])

                pred = float(rf_model.predict(X_test)[0])
                pred = round(pred / area, 2)

                crop_predictions[c] = pred

            best_crop=max(crop_predictions,key=crop_predictions.get)
            best_yield=crop_predictions[best_crop]

            # LSTM forecast

            last_yields=np.array([[2.1],[2.4],[2.2]])

            scaled=lstm_scaler.transform(last_yields)
            X_seq=scaled.reshape(1,3,1)

            lstm_out=lstm_model.predict(X_seq)[0][0]

            prediction=round(
                float(lstm_scaler.inverse_transform([[lstm_out]])[0][0]),2
            )

        except Exception as e:
            error=str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        user_prediction=user_prediction,
        best_crop=best_crop,
        best_yield=best_yield,
        season=season,
        user_crop=user_crop,
        error=error,
        crops=crops,
        districts=districts
    )


if __name__=="__main__":
    app.run(debug=True)
    
    
    
    
    