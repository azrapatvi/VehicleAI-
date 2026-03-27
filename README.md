# 🚗 VehicleAI — Vehicle Price Prediction Web App

A Flask-based machine learning web application that predicts vehicle prices — used cars, used bikes, and new electric vehicles (EVs). It also helps users check if a deal is fair, find vehicles within a budget, and estimate resale values.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Tech Stack](#tech-stack)
5. [How It Works — Page by Page](#how-it-works--page-by-page)
6. [ML Models Used](#ml-models-used)
7. [How to Run the Project](#how-to-run-the-project)
8. [File Descriptions](#file-descriptions)
9. [Data Files](#data-files)
10. [Model Files](#model-files)
11. [Known Limitations](#known-limitations)

---

## 📖 Project Overview

**VehicleAI** is a smart vehicle pricing tool powered by machine learning. It helps buyers and sellers in India make better decisions by:

- Predicting the **fair market price** of a used car or bike
- Estimating **new EV prices** based on specs (trained on German dataset)
- Checking if a **seller's asking price** is too high, too low, or fair
- Listing all vehicles that fit within a given **budget**

The app uses **3 separate ML models** — one for cars, one for bikes, and one for EVs — all integrated into a clean, simple web interface.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔋 EV Price Predictor | Predict new EV price from battery, range, speed, brand, etc. |
| 🚗 Resale Estimator | Predict resale price for used cars and bikes |
| ✅ Fair Price Checker | Enter a seller's asking price — the app says if it's a good deal or overpriced |
| 💰 Budget Finder | Enter your budget — see every car, bike, or EV that fits |
| 🏠 Home Dashboard | Overview of all tools with quick navigation |

---

## 🗂 Project Structure

```
vehicleai/
│
├── main.py                          # Flask app — all routes and logic
│
├── templates/                       # HTML templates (Jinja2)
│   ├── base.html                    # Base layout with navbar
│   ├── index.html                   # Home page
│   ├── ev_predictor.html            # EV price prediction form
│   ├── resale_prices.html           # Used car / bike resale form
│   ├── fair_price.html              # Fair price checker form
│   └── budget_finder.html          # Budget-based vehicle finder
│
├── data/                            # CSV datasets
│   ├── cleaned.csv                  # Cleaned used car data
│   ├── cleaned_bike_data.csv        # Cleaned used bike data
│   └── ev_cars_cleaned.csv         # Cleaned EV data (German dataset)
│
├── pickle_files/                    # Saved ML models and encoders (cars & bikes)
│   ├── car_encoders.pkl
│   ├── car_preprocessor.pkl
│   ├── car_price_predictor_model.pkl
│   ├── bike_city_encoder.pkl
│   ├── bike_preprocessor.pkl
│   └── bike_price_predictor_model.pkl
│
├── joblib_files/                    # Saved EV model and preprocessor
│   ├── ev_preprocessor.joblib
│   └── ev_model.joblib
│
├── mean_encoded/                    # Mean encoding mappings for EV categorical features
│   └── mean_encodings.json
│
└── notebooks/                       # Jupyter notebooks (model training)
    ├── Bike_Price_Prediction.ipynb
    ├── car_price_prediction.ipynb
    └── ev_vehicle.ipynb
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask |
| **ML / Data** | scikit-learn, pandas, numpy, joblib, pickle |
| **Frontend** | HTML, CSS (custom), Jinja2 templating |
| **Fonts** | Google Fonts — DM Sans, DM Serif Display |
| **Data Format** | CSV files for budget finder; pkl/joblib for models |

---

## 🔍 How It Works — Page by Page

### 🏠 Home Page (`/`)
- Landing page that introduces the app
- Shows 3 main prediction cards: Used Car, Used Bike, New EV
- Shows "Special Tools" section with quick access to Fair Price, Odometer Alert, etc.
- All cards are clickable and link to the relevant tool

---

### 🔋 EV Price Predictor (`/ev_predictor`)

**What it does:** Predicts the market price of a new Electric Vehicle in Euros (with optional Indian Rupee conversion).

**Input fields (all via sliders + dropdowns):**
- Battery capacity (kWh)
- Efficiency (Wh/km)
- Fast charge speed (km/h)
- Range (km)
- Top speed (km/h)
- 0–100 km/h acceleration (seconds)
- Brand (Tesla, BMW, Hyundai, etc.)
- Drive type (AWD / FWD / RWD / Unknown)
- Variant (Long Range, Performance, Standard, etc.)

**How it predicts:**
1. User fills the form and submits
2. A pandas DataFrame is created from the inputs
3. Brand and variant columns are encoded using **mean encoding** (loaded from `mean_encodings.json`)
4. The preprocessor (loaded from `ev_preprocessor.joblib`) scales the data
5. The model (loaded from `ev_model.joblib`) predicts the price in Euros
6. A ±10% price range is shown (lower = predicted × 0.9, upper = predicted × 1.1)
7. An "Indian Price" button converts the Euro price to INR using a rate of ₹91.5/€

**Output:** Predicted price in € with a confidence range

---

### 🚗 Resale Estimator (`/resale_prices`)

**What it does:** Predicts resale (second-hand) price of a used car or bike in India.

#### For Used Cars:
**Input fields:**
- Brand (Maruti, Hyundai, Toyota, etc.)
- Model name
- Vehicle age (years)
- KM driven
- Seller type (Individual / Dealer / Trustmark Dealer)
- Fuel type (Petrol / Diesel / CNG / Electric / LPG)
- Transmission type (Manual / Automatic)
- Mileage (kmpl)
- Engine capacity (cc)
- Max power (bhp)
- Number of seats

**How it predicts:**
1. Brand and model are encoded using Label Encoders (from `car_encoders.pkl`)
2. All features are scaled using the car preprocessor (from `car_preprocessor.pkl`)
3. The car model (from `car_price_predictor_model.pkl`) predicts the resale price
4. A ±10% confidence range is shown

#### For Used Bikes:
**Input fields:**
- Brand
- Owner type (1st / 2nd / 3rd owner, etc.)
- City
- Age (years)
- KMs driven
- Engine power (bhp)

**How it predicts:**
1. City is encoded using a city-level mean encoder (from `bike_city_encoder.pkl`)
2. Features are scaled using the bike preprocessor (from `bike_preprocessor.pkl`)
3. The bike model (from `bike_price_predictor_model.pkl`) predicts the price
4. A ±10% confidence range is shown

**Output:** Predicted resale price in ₹ with a confidence range

---

### ✅ Fair Price Checker (`/fair_price`)

**What it does:** You enter the seller's **asking price** along with the vehicle's details. The app predicts the **fair market price**, then compares — telling you if the deal is good or overpriced.

**Supports three vehicle types:**
- Used Car
- Used Bike
- New EV

**How it works:**
- Same prediction logic as the Resale Estimator / EV Predictor
- The **asking price** is collected as an extra field
- After prediction, the result page shows:
  - The **predicted fair price**
  - The **asking price** (with strikethrough if overpriced)
  - A **verdict** (e.g., "Good Deal" in green or "Overpriced" in red)

**Special note for EVs in Fair Price:**
- The EV model originally predicts in Euros
- For the Fair Price tool, the result is multiplied by `91.5` to convert to INR before comparing with the asking price in rupees

**Output:** Fair price vs. asking price with a deal verdict

---

### 💰 Budget Finder (`/budget_finder`)

**What it does:** You enter a budget in ₹. The app filters real dataset records and shows every vehicle that falls within that budget.

**Supports three vehicle types:**
- Used Car (filtered from `cleaned.csv` on `selling_price` column)
- Used Bike (filtered from `cleaned_bike_data.csv` on `price` column)
- New EV (filtered from `ev_cars_cleaned.csv` on `price_de_` column)

**How it works:**
1. User selects vehicle type and enters budget
2. The relevant CSV is loaded using pandas
3. Rows where the price column ≤ budget are returned
4. Results are displayed in a scrollable table showing all columns
5. A count of matching vehicles is shown

**Output:** A table of all matching vehicles with all their details

---

## 🤖 ML Models Used

### Car Price Model
- **File:** `car_price_predictor_model.pkl`
- **Trained on:** Indian used car dataset
- **Features:** Brand, model, vehicle age, km driven, seller type, fuel type, transmission, mileage, engine, max power, seats
- **Target:** Selling price (₹)

### Bike Price Model
- **File:** `bike_price_predictor_model.pkl`
- **Trained on:** Indian used bike dataset
- **Features:** KMs driven, owner history, age, engine power, brand, city
- **Target:** Resale price (₹)

### EV Price Model
- **File:** `ev_model.joblib`
- **Trained on:** German EV dataset (306 models)
- **Features:** Battery, efficiency, fast charge, range, top speed, acceleration, brand, drive type, variant
- **Target:** Market price (€)
- **Encoding:** Brand and variant use **mean encoding** stored in `mean_encodings.json`

---

## ▶️ How to Run the Project

### Prerequisites
Make sure you have Python 3.8+ installed.

### Step 1 — Clone or download the project
```bash
git clone <your-repo-url>
cd vehicleai
```

### Step 2 — Install dependencies
```bash
pip install flask pandas numpy scikit-learn joblib
```

### Step 3 — Make sure all model files are present
Ensure the following directories and files exist:
- `pickle_files/` — contains all `.pkl` model and encoder files
- `joblib_files/` — contains `ev_model.joblib` and `ev_preprocessor.joblib`
- `mean_encoded/mean_encodings.json` — contains mean encoding mappings
- `data/` — contains `cleaned.csv`, `cleaned_bike_data.csv`, `ev_cars_cleaned.csv`

### Step 4 — Run the Flask app
```bash
python main.py
```

### Step 5 — Open in browser
Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📄 File Descriptions

| File | Purpose |
|---|---|
| `main.py` | Main Flask application with all 5 routes |
| `base.html` | Base HTML template with navbar (extended by all pages) |
| `index.html` | Home page — shows all tools |
| `ev_predictor.html` | EV prediction form with sliders |
| `resale_prices.html` | Used car and bike resale form |
| `fair_price.html` | Fair price checker with asking price comparison |
| `budget_finder.html` | Budget-based vehicle filter with results table |
| `Bike_Price_Prediction.ipynb` | Jupyter notebook — bike model training |
| `car_price_prediction.ipynb` | Jupyter notebook — car model training |
| `ev_vehicle.ipynb` | Jupyter notebook — EV model training |

---

## 🗃 Data Files

| File | Description |
|---|---|
| `data/cleaned.csv` | Used car dataset — includes brand, model, fuel, transmission, selling price, etc. |
| `data/cleaned_bike_data.csv` | Used bike dataset — includes brand, power, city, age, price, etc. |
| `data/ev_cars_cleaned.csv` | EV dataset (German market) — includes battery, range, brand, price in € |

---

## 💾 Model Files

| File | Used For |
|---|---|
| `pickle_files/car_encoders.pkl` | Label encoders for car brand and model |
| `pickle_files/car_preprocessor.pkl` | Scaler/preprocessor for car features |
| `pickle_files/car_price_predictor_model.pkl` | Trained car price prediction model |
| `pickle_files/bike_city_encoder.pkl` | Mean encoder for bike city feature |
| `pickle_files/bike_preprocessor.pkl` | Scaler/preprocessor for bike features |
| `pickle_files/bike_price_predictor_model.pkl` | Trained bike price prediction model |
| `joblib_files/ev_preprocessor.joblib` | Scaler/preprocessor for EV features |
| `joblib_files/ev_model.joblib` | Trained EV price prediction model |
| `mean_encoded/mean_encodings.json` | Mean encoding values for EV brand and variant |

---

## ⚠️ Known Limitations

- **EV dataset is German** — the EV model is trained on European data, so Indian EV prices are approximated by converting Euros to INR at a fixed rate of ₹91.5/€. Real Indian prices may vary.
- **Budget Finder shows raw CSV data** — column names are as-is from the dataset and may not be user-friendly.
- **No real-time data** — prices are based on training data and won't reflect the current market live.
- **Unseen brands/variants in EV model** — if a brand or variant was not in training data, the app fills in the global mean value as a fallback.
- **The EV Predictor page** currently uses a JavaScript placeholder formula for instant display (not the actual model). The real ML prediction only happens on form submission via Flask POST.
- **No authentication** — the app has no login system; it is meant for local/demo use.

---

## 👨‍💻 Author Notes

This project was built as a machine learning portfolio project covering:
- Data cleaning and feature engineering (see Jupyter notebooks)
- Model training and serialization (pickle + joblib)
- Flask web app integration
- Jinja2 templating with dynamic form rendering
- Responsive frontend with pure CSS (no framework)
