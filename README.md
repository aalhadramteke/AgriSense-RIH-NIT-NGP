AgriSense: AI-Powered Crop Prediction Dashboard

AgriSense is a comprehensive, intelligent agricultural recommendation system that integrates Machine Learning models with a seamless Django backend to provide farmers with crop recommendations and yield predictions based on real-time soil and climate data.

🏗 Architecture Overview

```text
    [ Frontend UI ] (HTML/CSS/JS)
         |     ^ 
    Fetch|     | JSON Response
         v     |
     [ Django REST API ]
         |       |
         |       |---> [ OpenWeatherMap API ]
         |
    [ ML Models ] (Scikit-Learn)
      - RandomForest (Crop)
      - GradientBoosting (Yield)
         |
    [ SQLite Database ] -> Stores prediction history
```

⚙️ Setup Instructions

Follow these steps to run the project locally.

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\Activate
   # Linux/MacOS
   source venv/bin/activate
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Models**:
   The script first checks for `Crop_recommendation.csv` dataset. If missing, it uses a synthetic generator.
   ```bash
   python train_model.py
   ```

4. **Run Database Migrations**:
   ```bash
   python manage.py makemigrations predictor
   python manage.py migrate
   ```

5. **Start Development Server**:
   ```bash
   python manage.py runserver
   ```
   Navigate to `http://127.0.0.1:8000/` to use the dashboard!

---

📡 API Documentation

1. Crop Recommendation `POST /api/recommend/`
Analyzes soil and climate to predict the optimal crop.
**Request**:
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "pH": 6.5,
  "temperature": 20.8,
  "humidity": 82.0,
  "rainfall": 202.9
}
```

**Response**:
```json
{
  "recommended_crop": "rice",
  "confidence_score": 96.5,
  "full_prediction": { ... }
}
```

2. Yield Prediction `POST /api/predict-yield/`
**Request**: Same payload as `/api/recommend/`

**Response**:
```json
{
  "yield_estimate": 4256.4
}
```

3. Prediction History `GET /api/history/`
Returns the recent top 20 predictions made by users.

4. Fetch Weather `GET /api/weather/?lat=-&lon=-`
Proxies OpenWeatherMap to retrieve local climate based on GPS coordinates.

---

📸 Screenshots

- **Prediction Input Form:** *(Placeholder for UI Form input)*
- **Dashboard Results & Analytics:** *(Placeholder for charts and recommendations)*
- **History Viewer:** *(Placeholder for history table)*

---

🚀 Known Limitations and Future Improvements
- **Models Offline:** Requires running `train_model.py` which dynamically synthesizes data if none is found. True datasets provide higher accuracy.
- **Scalability:** Uses SQLite locally. Moving to PostgreSQL for production is advised.
- **Language Localization:** Include more languages via Web Speech API and UI.
- **Satellite Integration:** Possible future feature to grab soil data completely hands-free via satellite imaging APIs.
