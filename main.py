from flask import Flask,render_template,redirect,url_for,request
import pandas as pd
import numpy as np
from joblib import load,dump
import pickle
import json

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ev_predictor",methods=['GET','POST'])
def ev_predictor():
    if request.method=='POST':
        battery=float(request.form['battery'])
        efficiency=int(request.form['efficiency'])
        fast_charge=float(request.form['fast_charge'])
        range=int(request.form['range'])
        top_speed=int(request.form['top_speed'])
        acceleration__0_100_=float(request.form['acceleration__0_100_'])
        brand=request.form['brand']
        drive_type=request.form['drive_type']
        variant=request.form['variant']

        new_df=pd.DataFrame([{
            "battery":[battery],
            "efficiency":[efficiency],
            "fast_charge":[fast_charge],
            "range":[range],
            "top_speed":[top_speed],
            "acceleration__0_100_":[acceleration__0_100_],
            "brand":[brand],
            "drive_type":[drive_type],
            "variant":[variant]
        }])

        print(new_df)

        with open("mean_encoded/mean_encodings.json",'r')as f:
            loaded_encodings=json.load(f)
        new_df['brand']=new_df['brand'].map(loaded_encodings['brand'])
        new_df['variant']=new_df['variant'].map(loaded_encodings['variant'])

        preprocessor=load("joblib_files/ev_preprocessor.joblib")
        model=load("joblib_files/ev_model.joblib")

        new_df_scaled=preprocessor.transform(new_df)
        pred=model.predict(new_df_scaled)

        result=pred[0]
        lower_range = round(pred[0] * 0.9, 2)
        upper_range = round(pred[0] * 1.1, 2)

        return render_template("ev_predictor.html",result=result,lower_range=lower_range,upper_range=upper_range)

    return render_template("ev_predictor.html")

@app.route("/resale_prices", methods=['GET','POST'])
def resale_price():
    if request.method=='POST':
        vehicle_type = request.form.get('vehicle_type')

        if vehicle_type == 'car':
            brand             = request.form.get('brand')
            model_name        = request.form.get('model')
            vehicle_age       = float(request.form.get('vehicle_age'))
            km_driven         = float(request.form.get('km_driven'))
            seller_type       = request.form.get('seller_type')
            fuel_type         = request.form.get('fuel_type')
            transmission_type = request.form.get('transmission_type')
            mileage           = float(request.form.get('mileage'))
            engine            = float(request.form.get('engine'))
            max_power         = float(request.form.get('max_power'))
            seats             = int(request.form.get('seats'))

            new_df = pd.DataFrame({
                'brand':             [brand],
                'model':             [model_name],
                'vehicle_age':       [vehicle_age],
                'km_driven':         [km_driven],
                'seller_type':       [seller_type],
                'fuel_type':         [fuel_type],
                'transmission_type': [transmission_type],
                'mileage':           [mileage],
                'engine':            [engine],
                'max_power':         [max_power],
                'seats':             [seats]
            })

            with open("pickle_files/car_encoders.pkl",'rb')as f:
                encoders=pickle.load(f)

            new_df['brand'] = encoders['brand'].transform(new_df['brand'])
            new_df['model'] = encoders['model'].transform(new_df['model'])

            with open("pickle_files/car_preprocessor.pkl",'rb')as f:
                preprocessor=pickle.load(f)

            with open("pickle_files/car_price_predictor_model.pkl",'rb')as f:
                model=pickle.load(f)

            new_df_scaled=preprocessor.transform(new_df)
            pred=model.predict(new_df_scaled)

            res=pred[0]
            lower_range = round(pred[0] * 0.9, 2)
            upper_range = round(pred[0] * 1.1, 2)
            return render_template("resale_prices.html",res=res, vehicle_type='car',lower_range=lower_range,upper_range=upper_range)

        elif vehicle_type == 'bike':
            brand      = request.form.get('brand')
            owner      = request.form.get('owner')
            city       = request.form.get('city')
            age        = float(request.form.get('age'))
            kms_driven = float(request.form.get('kms_driven'))
            power      = float(request.form.get('power'))

            new_df = pd.DataFrame({
                'kms_driven': [kms_driven],
                'owner':      [owner],
                'age':        [age],
                'power':      [power],
                'brand':      [brand]
            })

            with open("pickle_files/bike_city_encoder.pkl", "rb") as f:
                city_encoder = pickle.load(f)
            with open("pickle_files/bike_preprocessor.pkl", "rb") as f:
                preprocessor = pickle.load(f)
            with open("pickle_files/bike_price_predictor_model.pkl", "rb") as f:
                bike_model = pickle.load(f)

            new_df["city_encoded"] = city_encoder.get(city, city_encoder.mean())
            new_df = preprocessor.transform(new_df)
            pred = bike_model.predict(new_df)

            res=pred[0]
            lower_range = round(pred[0] * 0.9, 2)
            upper_range = round(pred[0] * 1.1, 2)
            return render_template("resale_prices.html",res=res, vehicle_type='bike',lower_range=lower_range,upper_range=upper_range)

    return render_template("resale_prices.html")

@app.route("/fair_price", methods=['GET','POST'])
def fair_price():
    if request.method=='POST':
        vehicle_type=request.form.get('vehicle_type')

        if vehicle_type == 'used_car':
            brand             = request.form.get('brand')
            model_name        = request.form.get('model')
            vehicle_age       = float(request.form.get('vehicle_age'))
            km_driven         = float(request.form.get('km_driven'))
            seller_type       = request.form.get('seller_type')
            fuel_type         = request.form.get('fuel_type')
            transmission_type = request.form.get('transmission_type')
            mileage           = float(request.form.get('mileage'))
            engine            = float(request.form.get('engine'))
            max_power         = float(request.form.get('max_power'))
            seats             = int(request.form.get('seats'))

            new_df = pd.DataFrame({
                'brand':             [brand],
                'model':             [model_name],
                'vehicle_age':       [vehicle_age],
                'km_driven':         [km_driven],
                'seller_type':       [seller_type],
                'fuel_type':         [fuel_type],
                'transmission_type': [transmission_type],
                'mileage':           [mileage],
                'engine':            [engine],
                'max_power':         [max_power],
                'seats':             [seats]
            })

            with open("pickle_files/car_encoders.pkl",'rb')as f:
                encoders=pickle.load(f)

            new_df['brand'] = encoders['brand'].transform(new_df['brand'])
            new_df['model'] = encoders['model'].transform(new_df['model'])

            with open("pickle_files/car_preprocessor.pkl",'rb')as f:
                preprocessor=pickle.load(f)

            with open("pickle_files/car_price_predictor_model.pkl",'rb')as f:
                model=pickle.load(f)

            new_df_scaled=preprocessor.transform(new_df)
            pred=model.predict(new_df_scaled)

            res=pred[0]
            lower_range = round(pred[0] * 0.9, 2)
            upper_range = round(pred[0] * 1.1, 2)

            asking_price = float(request.form.get('asking_price')) 

            return render_template("fair_price.html",res=res,asking_price=asking_price, vehicle_type='car',lower_range=lower_range,upper_range=upper_range)

        elif vehicle_type == 'used_bike':
            brand      = request.form.get('brand')
            owner      = request.form.get('owner')
            city       = request.form.get('city')
            age        = float(request.form.get('age'))
            kms_driven = float(request.form.get('kms_driven'))
            power      = float(request.form.get('power'))

            new_df = pd.DataFrame({
                'kms_driven': [kms_driven],
                'owner':      [owner],
                'age':        [age],
                'power':      [power],
                'brand':      [brand]
            })

            with open("pickle_files/bike_city_encoder.pkl", "rb") as f:
                city_encoder = pickle.load(f)
            with open("pickle_files/bike_preprocessor.pkl", "rb") as f:
                preprocessor = pickle.load(f)
            with open("pickle_files/bike_price_predictor_model.pkl", "rb") as f:
                bike_model = pickle.load(f)

            new_df["city_encoded"] = city_encoder.get(city, city_encoder.mean())
            new_df = preprocessor.transform(new_df)
            pred = bike_model.predict(new_df)

            res=pred[0]
            lower_range = round(pred[0] * 0.9, 2)
            upper_range = round(pred[0] * 1.1, 2)

            asking_price = float(request.form.get('asking_price'))
            return render_template("fair_price.html",res=res,asking_price=asking_price, vehicle_type='bike',lower_range=lower_range,upper_range=upper_range)
        
        elif vehicle_type == 'new_ev':
            battery              = float(request.form['battery'])
            efficiency           = int(request.form['efficiency'])
            fast_charge          = float(request.form['fast_charge'])
            range_               = int(request.form['range'])
            top_speed            = int(request.form['top_speed'])
            acceleration__0_100_ = float(request.form['acceleration__0_100_'])
            brand                = request.form['brand']
            drive_type           = request.form['drive_type']
            variant              = request.form['variant']

    
            new_df = pd.DataFrame([{
                "battery":              battery,
                "efficiency":           efficiency,
                "fast_charge":          fast_charge,
                "range":                range_,
                "top_speed":            top_speed,
                "acceleration__0_100_": acceleration__0_100_,
                "brand":                brand,
                "drive_type":           drive_type,
                "variant":              variant
            }])

            with open("mean_encoded/mean_encodings.json", 'r') as f:
                loaded_encodings = json.load(f)

            new_df['brand']   = new_df['brand'].map(loaded_encodings['brand'])
            new_df['variant'] = new_df['variant'].map(loaded_encodings['variant'])

            # fill NaN for unseen brands/variants with global mean
            global_mean = sum(loaded_encodings['brand'].values()) / len(loaded_encodings['brand'])
            new_df['brand']   = new_df['brand'].fillna(global_mean)
            new_df['variant'] = new_df['variant'].fillna(global_mean)

            preprocessor = load("joblib_files/ev_preprocessor.joblib")
            model        = load("joblib_files/ev_model.joblib")

            new_df_scaled = preprocessor.transform(new_df)
            pred          = model.predict(new_df_scaled)

            res          = pred[0]
            lower_range  = round(pred[0] * 0.9, 2)
            upper_range  = round(pred[0] * 1.1, 2)
            asking_price = float(request.form.get('asking_price'))

            res=res*91.5

            return render_template("fair_price.html",
                res=res,
                asking_price=asking_price,
                vehicle_type='new_ev',
                lower_range=lower_range,
                upper_range=upper_range
            )

    return render_template("fair_price.html")

@app.route("/budget_finder", methods=['GET', 'POST'])
def budget_finder():
    filtered_df = None
    error = None

    if request.method == 'POST':
        vehicle_type = request.form.get('vehicle_type')
        budget_input = request.form.get('budget')  # safer than direct []

        # Validate budget input
        if not budget_input:
            error = "Please enter a valid budget."
        else:
            try:
                budget = float(budget_input)

                if vehicle_type == 'car':
                    car_df = pd.read_csv("data/cleaned.csv")
                    car_df.drop('car_name', axis=1, inplace=True, errors='ignore')

                    
                    filtered_df = car_df[car_df['selling_price'] <= budget]
                    

                elif vehicle_type=='bike':
                    bike_df = pd.read_csv("data/cleaned_bike_data.csv")

                    # Filter cars under budget
                    filtered_df = bike_df[bike_df['price'] <= budget]
                

                elif vehicle_type=='ev':
                    ev_df = pd.read_csv("data/ev_cars_cleaned.csv")

                    
                    filtered_df = ev_df[ev_df['price_de_'] <= budget]
                    



            except ValueError:
                error = "Budget must be a number."

    return render_template("budget_finder.html", filtered_df=filtered_df, error=error)
if __name__ == '__main__':
    app.run(debug=True)