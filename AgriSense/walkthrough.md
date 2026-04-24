# 🌱 AgriSense - AI-Powered Crop Prediction & Recommendation

## 🎯 Project Overview
**AgriSense** is a full-stack web application combining ML models (Random Forest + Gradient Boosting) with Django REST API and responsive Vanilla JS frontend. Farmers input soil parameters (N,P,K,pH,temp,humidity,rainfall) to get:

- ✅ Crop recommendations with confidence
- 📈 Yield predictions (kg/ha)
- 🌤️ Live weather via GPS + OpenWeather
- ⚠️ Risk alerts (pH, drought, etc.)
- 💰 Market prices
- 📊 Charts (bar, radar) + Voice announcements
- 📋 Prediction history (SQLite DB)

```
┌─────────────────────┐    ┌──────────────┐    ┌──────────────────┐
│   Frontend SPA      │───▶│   Django API  │───▶│   ML Models      │
│  index.html + JS    │    │ recommend/    │    │ crop_model.pkl   │
│ Charts/Voice/GPS    │    │ predict-yield │    │ yield_model.pkl  │
└─────────────────────┘    │ history/      │    │ SQLite DB        │
                           │ weather/      │    └──────────────────┘
                           └──────────────┘
```

## 🚀 Quick Setup & Run (Windows)

1. **Virtual Environment**
   ```
   cd AgriSense
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Train ML Models** (downloads dataset automatically)
   ```
   python train_model.py
   ```
   Expected: `Crop accuracy: ~0.97`, models saved to `./models/`

4. **Database & Migrations**
   ```
   python crop_predict/manage.py makemigrations predictor
   python crop_predict/manage.py migrate
   ```

5. **Set OpenWeather API Key** (Optional but recommended for weather)
   - Signup: https://openweathermap.org/api (free)
   - Edit `crop_predict/settings.py`: `OPENWEATHER_API_KEY = 'your_key_here'`
   - Or `.env` file.

6. **Run Server**
   ```
   cd crop_predict
   python manage.py runserver
   ```
   Open: http://127.0.0.1:8000/

## 🔧 API Documentation

| Endpoint | Method | Request JSON | Response |
|----------|--------|--------------|----------|
| `/api/recommend/` | POST | `{"n":50,"p":30,"k":40,"ph":6.2,"temperature":25,"humidity":70,"rainfall":120}` | `{"recommended_crop":"rice","confidence_score":0.95,"yield_estimate":2500}` |
| `/api/predict-yield/` | POST | Same | `{"yield_estimate":2500}` |
| `/api/history/` | GET | - | Array of last 20 predictions |
| `/api/weather/?lat=28.6&lon=77.2` | GET | - | `{"temperature":28,"humidity":65,"rainfall":0}` |

**Sample cURL:**
```bash
curl -X POST http://127.0.0.1:8000/api/recommend/ -H "Content-Type: application/json" -d "{\"n\":90,\"p\":42,\"k\":43,\"temperature\":20.879744,\"humidity\":82.002744,\"ph\":6.502985,\"rainfall\":202.935536}\"
```

## 📱 Frontend Features
- **Responsive**: Mobile-first glassmorphism design (deep green/teal/amber)
- **GPS Weather**: Auto-fills temp/humidity/rainfall
- **Charts**: Chart.js bar (yield) + radar (soil params)
- **Voice**: Web Speech API announces recommendation
- **Risks**: Color-coded alerts (red/high, yellow/med, green/low)
- **Market**: Static INR prices per quintal
- **History**: Real-time DB table

## 🗄️ Database Schema (SQLite db.sqlite3)
```
SoilRecord: n,p,k,ph,temperature,humidity,rainfall,location,created_at
Prediction: soil_record(FK), recommended_crop, yield_estimate, confidence_score, created_at
```

## ✅ Validation Checklist
- [ ] `python train_model.py` → models/*.pkl + accuracy >95%
- [ ] `makemigrations && migrate` → No errors, admin tables
- [ ] `runserver` → http://localhost:8000/ dashboard loads
- [ ] Form submit → Recommendation + charts + voice + history update
- [ ] GPS button → Weather fills (with API key)
- [ ] Responsive on mobile

## 🔒 Admin
```
http://127.0.0.1:8000/admin/
Superuser: python manage.py createsuperuser
```

## 🚀 Production Deployment
```
pip freeze > requirements-prod.txt
gunicorn crop_predict.wsgi
 whitenoise for static
 PostgreSQL
 .env for keys
```

## 📈 Model Performance
```
Crop Recommendation (RF): ~97% accuracy (22 crops)
Yield Prediction (GB): R² ~0.85 (synthetic targets)
Dataset: 2200 samples from Kaggle Crop Recommendation
```

## ♻️ Known Limitations & Future Improvements
- **Yield**: Synthetic targets - train on real yield dataset
- **Weather**: Rain proxy (use separate forecast API)
- **Crops**: India-focused, expand globally
- **Auth**: Add user accounts
- **PWA**: Offline mode, push notifications
- **More ML**: Disease risk from images (integrate existing project)
- **Prices**: Live API (Agmarknet)

**Built with ❤️ by BLACKBOXAI**  
*Empowering farmers with AI*
