from flask import Flask, render_template, request
import os
import requests
from datetime import datetime

# ================================
# ✅ SAFE MODEL LOADING
# ================================
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
    import numpy as np

    # ✅ Find model.h5 - check multiple locations
    possible_paths = [
        r'C:\project\model\model.h5',
        r'C:\Users\USER\models\model.h5',
        r'C:\Users\USER\Desktop\model.h5',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'model.h5')
    ]

    model = None
    MODEL_LOADED = False

    for path in possible_paths:
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            MODEL_LOADED = True
            print(f"✅ Model Loaded from: {path}")
            break

    if not MODEL_LOADED:
        print("⚠️ model.h5 not found - using dummy predictions")

except Exception as e:
    print(f"⚠️ TensorFlow error: {e}")
    MODEL_LOADED = False
    model = None

app = Flask(__name__)

# ================================
# CLASS LABELS
# ================================
# ================================
# CLASS LABELS
# ================================
class_labels = {
    0:  'Anthracnose_Mango',
    1:  'Black_rot_Grape',
    2:  'Healthy_Leaf',           # ✅ NEW
    3:  'Powdery_Mildew_Cotton',
    4:  'Powdery_Mildew_Mango',
    5:  'Powdery_Mildew_Pumpkin',
    6:  'RedRot_Sugarcane',
    7:  'Rust_Sugarcane',
    8:  'Sooty_Mould_Mango',
    9:  'Target_Spot_Tomato',
    10: 'Unknown_Disease',
    11: 'Yellow_Sugarcane'
}
# ✅ Check if prediction is healthy
def is_healthy(disease):
    return disease == 'Healthy_Leaf'
# ================================
# MEDICINE MAP
# ================================
medicine_map = {
    # ✅ Healthy - no medicine
    'Healthy_Leaf':           ('No Medicine Needed', 'healthy.jpg'),

    # Diseases
    'Anthracnose_Mango':      ('Carbendazim',   'carbendazim.jpg'),
    'Black_rot_Grape':        ('Mancozeb',       'mancozeb.jpg'),
    'Powdery_Mildew_Mango':   ('Sulfur',         'sulfur.jpg'),
    'Powdery_Mildew_Cotton':  ('Hexaconazole',   'hexaconazole.jpg'),
    'Powdery_Mildew_Pumpkin': ('Neem Oil',       'neem.jpg'),
    'RedRot_Sugarcane':       ('Carbendazim',    'carbendazim.jpg'),
    'Rust_Sugarcane':         ('Propiconazole',  'propiconazole.jpg'),
    'Sooty_Mould_Mango':      ('Neem Oil',       'neem.jpg'),
    'Target_Spot_Tomato':     ('Chlorothalonil', 'chlorothalonil.jpg'),
    'Yellow_Sugarcane':       ('Urea Spray',     'urea.jpg'),
    'Unknown_Disease':        ('Consult Expert', 'default.jpg')
}

# ================================
# DISEASE INFO MAP
# ================================
disease_info_map = {
    'Anthracnose_Mango': {
        "symptoms": "Black spots on leaves and fruits",
        "cause": "Fungal infection (Colletotrichum)",
        "prevention": "Use Carbendazim spray, remove infected parts"
    },
    'Black_rot_Grape': {
        "symptoms": "Dark circular lesions on leaves and fruit",
        "cause": "Fungus (Guignardia bidwellii)",
        "prevention": "Spray Mancozeb, remove infected clusters"
    },
    'Powdery_Mildew_Mango': {
        "symptoms": "White powdery coating on young leaves",
        "cause": "Fungal infection (Oidium mangiferae)",
        "prevention": "Apply sulfur spray or Hexaconazole"
    },
    'Powdery_Mildew_Cotton': {
        "symptoms": "White powder on upper leaf surface",
        "cause": "Fungus (Leveillula taurica)",
        "prevention": "Use Hexaconazole or Dinocap"
    },
    'Powdery_Mildew_Pumpkin': {
        "symptoms": "White patches on leaves",
        "cause": "Fungal growth (Sphaerotheca fuliginea)",
        "prevention": "Neem oil spray every 7 days"
    },
    'RedRot_Sugarcane': {
        "symptoms": "Red discoloration inside stem with sour smell",
        "cause": "Fungus (Colletotrichum falcatum)",
        "prevention": "Use resistant varieties, treat setts with Carbendazim"
    },
    'Rust_Sugarcane': {
        "symptoms": "Orange-brown rust pustules on leaves",
        "cause": "Fungus (Puccinia melanocephala)",
        "prevention": "Apply Propiconazole fungicide"
    },
    'Sooty_Mould_Mango': {
        "symptoms": "Black sooty coating on leaves",
        "cause": "Fungus growing on insect honeydew",
        "prevention": "Control mealybug/scale insects with Neem Oil"
    },
    'Target_Spot_Tomato': {
        "symptoms": "Circular brown spots with concentric rings",
        "cause": "Fungal infection (Corynespora cassiicola)",
        "prevention": "Use Chlorothalonil or Mancozeb spray"
    },
    'Yellow_Sugarcane': {
        "symptoms": "Yellowing of leaves, stunted growth",
        "cause": "Nutrient deficiency or viral infection",
        "prevention": "Apply balanced fertilizers, remove infected plants"
    },
    'Unknown_Disease': {
        "symptoms": "Disease could not be identified clearly",
        "cause": "Unknown or multiple causes",
        "prevention": "Consult your local agriculture officer"
    },
    'Healthy_Leaf': {
        "symptoms": "No disease symptoms detected",
        "cause":    "Plant is perfectly healthy",
        "prevention": "Continue regular care and monitoring"
    }
}


# ================================
# 🤖 PREDICT FUNCTION (built into app.py)
# ================================
def run_prediction(img_path):
    if not MODEL_LOADED or model is None:
        return 'Unknown_Disease', 75.0

    try:
        # ✅ Check what size your current model needs
        input_shape = model.input_shape  # prints (None, 128, 128, 3) or (None, 224, 224, 3)
        img_size = input_shape[1]        # gets 128 or 224 automatically
        print(f"✅ Model input size: {img_size}x{img_size}")

        img       = keras_image.load_img(img_path, target_size=(img_size, img_size))
        img_array = keras_image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions  = model.predict(img_array)[0]
        class_index  = int(np.argmax(predictions))
        confidence   = float(predictions[class_index] * 100)
        disease      = class_labels.get(class_index, 'Unknown_Disease')

        print(f"✅ Predicted: {disease} ({confidence:.2f}%)")
        return disease, round(confidence, 2)

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return 'Unknown_Disease', 0.0


# ================================
# 🌦️ LIVE WEATHER
# ================================
def get_live_weather(city="Pune"):
    API_KEY = "YOUR_OPENWEATHER_API_KEY"   # ← paste your key here
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        res = requests.get(url, timeout=5).json()
        return {
            "temp":      round(res["main"]["temp"]),
            "humidity":  res["main"]["humidity"],
            "wind":      round(res["wind"]["speed"] * 3.6),
            "rain":      res.get("rain", {}).get("1h", 0),
            "condition": res["weather"][0]["description"].title()
        }
    except:
        return {"temp": 30, "humidity": 60, "wind": 10, "rain": 0, "condition": "Sunny"}


# ================================
# 💡 FARMING TIPS
# ================================
def get_farming_tips(weather, crop):
    tips = []
    temp     = weather["temp"]
    humidity = weather["humidity"]
    rain     = weather["rain"]
    wind     = weather["wind"]

    if temp > 38:
        tips.append("🌡️ Very hot today! Water your crops early morning or evening only.")
    elif temp < 15:
        tips.append("🥶 Cold weather alert! Cover young seedlings to protect from frost.")
    else:
        tips.append("✅ Temperature is ideal for crop growth today.")

    if humidity > 80:
        tips.append(f"💧 High humidity ({humidity}%) — High risk of fungal disease! Spray fungicide on {crop} immediately.")
    elif humidity > 65:
        tips.append(f"⚠️ Moderate humidity — Monitor {crop} leaves for early signs of mildew or blight.")
    else:
        tips.append("✅ Humidity is normal. No immediate fungal risk.")

    if rain > 5:
        tips.append("🌧️ Heavy rain today — Avoid spraying pesticides, they will wash away.")
    elif rain > 0:
        tips.append("🌦️ Light rain — Good for crops but watch for waterlogging.")
    else:
        tips.append("☀️ No rain today — Consider irrigation if soil looks dry.")

    if wind > 30:
        tips.append("🌬️ Strong winds — Do NOT spray pesticides, they will drift away.")

    crop_tips = {
        "mango":     "🥭 Mango tip: High humidity increases Anthracnose risk. Use Carbendazim spray.",
        "tomato":    "🍅 Tomato tip: Watch for Target Spot in humid conditions. Apply Chlorothalonil.",
        "cotton":    "🌿 Cotton tip: Hot dry weather increases mite infestation. Check leaves.",
        "grapes":    "🍇 Grapes tip: Humidity above 70% triggers Black Rot. Spray Mancozeb.",
        "sugarcane": "🎋 Sugarcane tip: Wet weather increases Red Rot risk. Check stem color.",
        "pumpkin":   "🎃 Pumpkin tip: Spray Neem Oil in morning for Powdery Mildew control."
    }
    tips.append(crop_tips.get(crop.lower(), "🌾 Keep monitoring your crop regularly."))
    return tips


# ================================
# 🐛 PEST ALERT
# ================================
def get_pest_alerts(crop):
    month      = datetime.now().month
    month_name = datetime.now().strftime("%B")

    alerts = {
        "mango": {
            1:  {"pest": "Powdery Mildew",  "risk": "HIGH",   "action": "Spray Sulfur immediately"},
            2:  {"pest": "Mango Hopper",    "risk": "HIGH",   "action": "Spray Imidacloprid"},
            3:  {"pest": "Fruit Fly",       "risk": "MEDIUM", "action": "Use bait traps"},
            4:  {"pest": "Anthracnose",     "risk": "HIGH",   "action": "Spray Carbendazim"},
            5:  {"pest": "Fruit Fly",       "risk": "HIGH",   "action": "Harvest early if possible"},
            6:  {"pest": "Stem Borer",      "risk": "MEDIUM", "action": "Apply chlorpyrifos paste"},
            7:  {"pest": "Sooty Mould",     "risk": "MEDIUM", "action": "Control mealybugs first"},
            8:  {"pest": "Leaf Webber",     "risk": "LOW",    "action": "Remove webbed leaves"},
            9:  {"pest": "Scale Insects",   "risk": "LOW",    "action": "Spray neem oil"},
            10: {"pest": "Powdery Mildew",  "risk": "MEDIUM", "action": "Preventive sulfur spray"},
            11: {"pest": "Powdery Mildew",  "risk": "HIGH",   "action": "Spray Hexaconazole"},
            12: {"pest": "Powdery Mildew",  "risk": "HIGH",   "action": "Spray Carbendazim"}
        },
        "tomato": {
            1:  {"pest": "Late Blight",    "risk": "HIGH",   "action": "Spray Mancozeb"},
            2:  {"pest": "Leaf Miner",     "risk": "MEDIUM", "action": "Use yellow sticky traps"},
            3:  {"pest": "Whitefly",       "risk": "MEDIUM", "action": "Spray Imidacloprid"},
            4:  {"pest": "Target Spot",    "risk": "HIGH",   "action": "Spray Chlorothalonil"},
            5:  {"pest": "Fruit Borer",    "risk": "HIGH",   "action": "Use pheromone traps"},
            6:  {"pest": "Damping Off",    "risk": "MEDIUM", "action": "Improve drainage"},
            7:  {"pest": "Early Blight",   "risk": "HIGH",   "action": "Spray copper fungicide"},
            8:  {"pest": "Late Blight",    "risk": "HIGH",   "action": "Spray Metalaxyl"},
            9:  {"pest": "Fusarium Wilt",  "risk": "MEDIUM", "action": "Use resistant varieties"},
            10: {"pest": "Leaf Curl",      "risk": "MEDIUM", "action": "Control whitefly vector"},
            11: {"pest": "Late Blight",    "risk": "HIGH",   "action": "Spray every 7 days"},
            12: {"pest": "Late Blight",    "risk": "HIGH",   "action": "Spray Mancozeb"}
        },
        "sugarcane": {
            1:  {"pest": "Woolly Aphid",     "risk": "MEDIUM", "action": "Spray Chlorpyrifos"},
            2:  {"pest": "Early Shoot Borer","risk": "HIGH",   "action": "Apply Carbofuran"},
            3:  {"pest": "Red Rot",          "risk": "MEDIUM", "action": "Use disease-free setts"},
            4:  {"pest": "Pyrilla",          "risk": "MEDIUM", "action": "Release Epiricania"},
            5:  {"pest": "Top Borer",        "risk": "HIGH",   "action": "Spray Chlorantraniliprole"},
            6:  {"pest": "Red Rot",          "risk": "HIGH",   "action": "Remove infected plants"},
            7:  {"pest": "Grassy Shoot",     "risk": "HIGH",   "action": "Rogue out infected plants"},
            8:  {"pest": "Smut",             "risk": "MEDIUM", "action": "Use resistant varieties"},
            9:  {"pest": "Scale Insect",     "risk": "LOW",    "action": "Spray Malathion"},
            10: {"pest": "Ratoon Stunting",  "risk": "MEDIUM", "action": "Hot water treatment"},
            11: {"pest": "Yellow Leaf",      "risk": "MEDIUM", "action": "Apply micronutrients"},
            12: {"pest": "Woolly Aphid",     "risk": "LOW",    "action": "Monitor regularly"}
        },
        "cotton": {
            1:  {"pest": "None",           "risk": "LOW",    "action": "Land preparation time"},
            2:  {"pest": "None",           "risk": "LOW",    "action": "Prepare land"},
            3:  {"pest": "None",           "risk": "LOW",    "action": "Soil treatment"},
            4:  {"pest": "Aphid",          "risk": "MEDIUM", "action": "Spray Dimethoate"},
            5:  {"pest": "Thrips",         "risk": "MEDIUM", "action": "Spray Spinosad"},
            6:  {"pest": "Jassid",         "risk": "HIGH",   "action": "Spray Imidacloprid"},
            7:  {"pest": "Bollworm",       "risk": "HIGH",   "action": "Spray Profenofos"},
            8:  {"pest": "Pink Bollworm",  "risk": "HIGH",   "action": "Use pheromone traps"},
            9:  {"pest": "Whitefly",       "risk": "HIGH",   "action": "Spray Spiromesifen"},
            10: {"pest": "Mealy Bug",      "risk": "MEDIUM", "action": "Spray Buprofezin"},
            11: {"pest": "Aphid",          "risk": "LOW",    "action": "Final monitoring"},
            12: {"pest": "None",           "risk": "LOW",    "action": "Harvest residue removal"}
        },
        "grapes": {
            1:  {"pest": "Powdery Mildew", "risk": "HIGH",   "action": "Spray Sulfur after pruning"},
            2:  {"pest": "Downy Mildew",   "risk": "MEDIUM", "action": "Spray Copper fungicide"},
            3:  {"pest": "Thrips",         "risk": "HIGH",   "action": "Spray Spinosad"},
            4:  {"pest": "Black Rot",      "risk": "HIGH",   "action": "Spray Mancozeb"},
            5:  {"pest": "Mealybug",       "risk": "MEDIUM", "action": "Apply Buprofezin"},
            6:  {"pest": "None",           "risk": "LOW",    "action": "Rest period"},
            7:  {"pest": "None",           "risk": "LOW",    "action": "Pruning period"},
            8:  {"pest": "Downy Mildew",   "risk": "MEDIUM", "action": "Preventive spray"},
            9:  {"pest": "Powdery Mildew", "risk": "MEDIUM", "action": "Spray Hexaconazole"},
            10: {"pest": "Thrips",         "risk": "HIGH",   "action": "Spray before flowering"},
            11: {"pest": "Downy Mildew",   "risk": "HIGH",   "action": "Spray Metalaxyl"},
            12: {"pest": "Powdery Mildew", "risk": "HIGH",   "action": "Spray Sulfur"}
        },
        "pumpkin": {
            1:  {"pest": "Powdery Mildew",     "risk": "MEDIUM", "action": "Spray Neem Oil"},
            2:  {"pest": "Red Pumpkin Beetle", "risk": "HIGH",   "action": "Spray Carbaryl"},
            3:  {"pest": "Aphid",              "risk": "MEDIUM", "action": "Spray Dimethoate"},
            4:  {"pest": "Fruit Fly",          "risk": "HIGH",   "action": "Use bait traps"},
            5:  {"pest": "Downy Mildew",       "risk": "MEDIUM", "action": "Spray Mancozeb"},
            6:  {"pest": "Damping Off",        "risk": "HIGH",   "action": "Improve drainage"},
            7:  {"pest": "Powdery Mildew",     "risk": "HIGH",   "action": "Spray Sulfur"},
            8:  {"pest": "Fruit Fly",          "risk": "HIGH",   "action": "Cover fruit with bags"},
            9:  {"pest": "Aphid",              "risk": "MEDIUM", "action": "Spray neem oil"},
            10: {"pest": "Powdery Mildew",     "risk": "MEDIUM", "action": "Spray Hexaconazole"},
            11: {"pest": "None",               "risk": "LOW",    "action": "Land preparation"},
            12: {"pest": "None",               "risk": "LOW",    "action": "Soil improvement"}
        }
    }

    crop_alerts = alerts.get(crop.lower(), {})
    alert = crop_alerts.get(month, {"pest": "None", "risk": "LOW", "action": "Monitor regularly"})
    alert["month"] = month_name
    return alert


# ================================
# 🌱 CROP CALENDAR
# ================================
def get_crop_calendar(crop):
    calendars = {
        "mango": [
            {"month": "Jan", "activity": "🌸 Flowering",       "tip": "Apply micronutrients"},
            {"month": "Feb", "activity": "🐛 Pest monitoring", "tip": "Spray for hoppers"},
            {"month": "Mar", "activity": "🍋 Fruit setting",   "tip": "Irrigate regularly"},
            {"month": "Apr", "activity": "🥭 Fruit development","tip": "Apply potash"},
            {"month": "May", "activity": "🌾 Harvesting",      "tip": "Harvest at right maturity"},
            {"month": "Jun", "activity": "✂️ Pruning",         "tip": "Remove dead branches"},
            {"month": "Jul", "activity": "🌧️ Monsoon care",   "tip": "Avoid waterlogging"},
            {"month": "Aug", "activity": "🌿 Vegetative growth","tip": "Apply nitrogen"},
            {"month": "Sep", "activity": "🌿 Growth",          "tip": "Monitor for disease"},
            {"month": "Oct", "activity": "🌸 Bud initiation",  "tip": "Withhold irrigation"},
            {"month": "Nov", "activity": "🌸 Pre-flowering",   "tip": "Apply phosphorus"},
            {"month": "Dec", "activity": "🌸 Early flowering", "tip": "Spray boron"}
        ],
        "tomato": [
            {"month": "Jan", "activity": "🌱 Sowing",          "tip": "Use disease-free seeds"},
            {"month": "Feb", "activity": "🌿 Transplanting",   "tip": "60cm spacing"},
            {"month": "Mar", "activity": "🌸 Flowering",       "tip": "Apply DAP fertilizer"},
            {"month": "Apr", "activity": "🍅 Fruiting",        "tip": "Regular irrigation"},
            {"month": "May", "activity": "🌾 Harvesting",      "tip": "Harvest every 3 days"},
            {"month": "Jun", "activity": "🌧️ Off season",     "tip": "Prepare land"},
            {"month": "Jul", "activity": "🌱 Kharif sowing",   "tip": "Use raised beds"},
            {"month": "Aug", "activity": "🌿 Transplanting",   "tip": "Watch for blight"},
            {"month": "Sep", "activity": "🌸 Flowering",       "tip": "Spray fungicide"},
            {"month": "Oct", "activity": "🍅 Fruiting",        "tip": "Stake the plants"},
            {"month": "Nov", "activity": "🌾 Harvesting",      "tip": "Grade before selling"},
            {"month": "Dec", "activity": "🌱 Rabi prep",       "tip": "Soil testing"}
        ],
        "sugarcane": [
            {"month": "Jan", "activity": "🌱 Planting",        "tip": "Use healthy setts"},
            {"month": "Feb", "activity": "🌿 Early growth",    "tip": "Apply nitrogen"},
            {"month": "Mar", "activity": "🌾 Tillering",       "tip": "Gap filling"},
            {"month": "Apr", "activity": "☀️ Grand growth",    "tip": "Irrigate every 10 days"},
            {"month": "May", "activity": "🌿 Rapid growth",    "tip": "Earth up the crop"},
            {"month": "Jun", "activity": "🌧️ Monsoon",        "tip": "Ensure drainage"},
            {"month": "Jul", "activity": "🌿 Maturation",      "tip": "Stop nitrogen"},
            {"month": "Aug", "activity": "🎋 Maturing",        "tip": "Monitor for pests"},
            {"month": "Sep", "activity": "🎋 Pre-harvest",     "tip": "Arrange transport"},
            {"month": "Oct", "activity": "🌾 Harvesting",      "tip": "Cut at ground level"},
            {"month": "Nov", "activity": "🌱 Ratoon care",     "tip": "Apply fertilizer"},
            {"month": "Dec", "activity": "🌱 New planting",    "tip": "Prepare setts"}
        ],
        "cotton": [
            {"month": "Jan", "activity": "🏗️ Land prep",      "tip": "Deep ploughing"},
            {"month": "Feb", "activity": "🏗️ Land prep",      "tip": "Add compost"},
            {"month": "Mar", "activity": "🏗️ Pre-sowing",     "tip": "Soil testing"},
            {"month": "Apr", "activity": "🌱 Sowing starts",  "tip": "Treat seeds"},
            {"month": "May", "activity": "🌱 Sowing",         "tip": "90cm row spacing"},
            {"month": "Jun", "activity": "🌿 Seedling",       "tip": "First irrigation"},
            {"month": "Jul", "activity": "🌸 Flowering",      "tip": "Apply potash"},
            {"month": "Aug", "activity": "🌸 Boll formation", "tip": "Watch for bollworm"},
            {"month": "Sep", "activity": "🌾 Boll opening",   "tip": "Reduce irrigation"},
            {"month": "Oct", "activity": "🌾 Harvesting",     "tip": "Pick every 15 days"},
            {"month": "Nov", "activity": "🌾 Final picking",  "tip": "Grade the cotton"},
            {"month": "Dec", "activity": "🏗️ Post harvest",  "tip": "Remove crop residue"}
        ],
        "grapes": [
            {"month": "Jan", "activity": "✂️ Pruning",          "tip": "Keep 2-3 buds"},
            {"month": "Feb", "activity": "🌿 Bud burst",        "tip": "Apply copper spray"},
            {"month": "Mar", "activity": "🌸 Flowering",        "tip": "Spray boron"},
            {"month": "Apr", "activity": "🍇 Berry development","tip": "Thin clusters"},
            {"month": "May", "activity": "🍇 Ripening",         "tip": "Reduce irrigation"},
            {"month": "Jun", "activity": "🌾 Harvesting",       "tip": "Harvest in cool morning"},
            {"month": "Jul", "activity": "🌧️ Rest period",     "tip": "Withhold water"},
            {"month": "Aug", "activity": "✂️ Pre-pruning",      "tip": "Remove old wood"},
            {"month": "Sep", "activity": "✂️ Pruning prep",     "tip": "Apply fertilizer"},
            {"month": "Oct", "activity": "✂️ Main pruning",     "tip": "Prune to shape"},
            {"month": "Nov", "activity": "🌿 New shoots",       "tip": "Train shoots"},
            {"month": "Dec", "activity": "🌸 Flower init",      "tip": "Apply potash"}
        ],
        "pumpkin": [
            {"month": "Jan", "activity": "🌱 Sowing",          "tip": "Sow in warm soil"},
            {"month": "Feb", "activity": "🌿 Vine growth",     "tip": "Provide support"},
            {"month": "Mar", "activity": "🌸 Flowering",       "tip": "Hand pollinate"},
            {"month": "Apr", "activity": "🎃 Fruit set",       "tip": "Apply potash"},
            {"month": "May", "activity": "🌾 Harvesting",      "tip": "Harvest when stem dries"},
            {"month": "Jun", "activity": "🌱 Kharif sowing",   "tip": "Raised bed sowing"},
            {"month": "Jul", "activity": "🌿 Growth",          "tip": "Watch for mildew"},
            {"month": "Aug", "activity": "🌸 Flowering",       "tip": "Spray neem oil"},
            {"month": "Sep", "activity": "🎃 Fruiting",        "tip": "Reduce watering"},
            {"month": "Oct", "activity": "🌾 Harvest",         "tip": "Store in cool place"},
            {"month": "Nov", "activity": "🏗️ Land prep",      "tip": "Add organic matter"},
            {"month": "Dec", "activity": "🏗️ Resting",        "tip": "Soil improvement"}
        ]
    }
    return calendars.get(crop.lower(), [])


# ================================
# 💰 MANDI PRICES
# ================================
def get_mandi_prices():
    return {
        "Mango":     {"price": 4500, "change": "+200", "market": "Pune APMC"},
        "Tomato":    {"price": 1200, "change": "-150", "market": "Nashik APMC"},
        "Sugarcane": {"price": 3150, "change": "0",    "market": "Kolhapur APMC"},
        "Cotton":    {"price": 6800, "change": "+300", "market": "Akola APMC"},
        "Grapes":    {"price": 5200, "change": "+100", "market": "Sangli APMC"},
        "Pumpkin":   {"price": 800,  "change": "-50",  "market": "Mumbai APMC"},
        "Onion":     {"price": 1800, "change": "+400", "market": "Lasalgaon APMC"},
        "Wheat":     {"price": 2200, "change": "0",    "market": "Solapur APMC"},
        "Soybean":   {"price": 4100, "change": "+150", "market": "Latur APMC"},
    }

# ================================
# 📉 YIELD LOSS CALCULATOR
# ================================
def get_yield_loss(disease, confidence, crop):
    # Base yield loss % per disease
    base_loss = {
        'Anthracnose_Mango':      35,
        'Black_rot_Grape':        45,
        'Powdery_Mildew_Mango':   25,
        'Powdery_Mildew_Cotton':  20,
        'Powdery_Mildew_Pumpkin': 20,
        'RedRot_Sugarcane':       50,
        'Rust_Sugarcane':         30,
        'Sooty_Mould_Mango':      15,
        'Target_Spot_Tomato':     40,
        'Yellow_Sugarcane':       25,
        'Unknown_Disease':        20,
        'Healthy_Leaf':            0
    }

    # Average price per quintal (₹)
    crop_price = {
        'mango':     4500,
        'tomato':    1200,
        'cotton':    6800,
        'grapes':    5200,
        'sugarcane': 3150,
        'pumpkin':   800
    }

    # Average yield per acre (quintals)
    crop_yield = {
        'mango':     40,
        'tomato':    120,
        'cotton':    8,
        'grapes':    100,
        'sugarcane': 700,
        'pumpkin':   80
    }

    loss_pct     = base_loss.get(disease, 20) * (confidence / 100)
    price        = crop_price.get(crop.lower(), 2000)
    yield_q      = crop_yield.get(crop.lower(), 50)
    total_value  = price * yield_q
    money_loss   = round(total_value * loss_pct / 100)
    loss_pct     = round(loss_pct, 1)

    return {
        "loss_pct":    loss_pct,
        "money_loss":  money_loss,
        "total_value": total_value,
        "saved":       round(total_value - money_loss),
        "crop_yield":  yield_q,
        "crop_price":  price
    }


# ================================
# 🗓️ TREATMENT SCHEDULE GENERATOR
# ================================
def get_treatment_schedule(disease, medicine):
    schedules = {
        'Anthracnose_Mango': [
            {"day": "Day 1",  "emoji": "💊", "task": f"Apply {medicine} spray on all infected leaves and fruits", "type": "treatment"},
            {"day": "Day 2",  "emoji": "🌿", "task": "Remove and destroy all fallen infected leaves from ground", "type": "cleanup"},
            {"day": "Day 3",  "emoji": "🔍", "task": "Inspect treated areas — check if spots are drying up", "type": "check"},
            {"day": "Day 4",  "emoji": "💧", "task": "Water plants at base only, avoid wetting leaves", "type": "care"},
            {"day": "Day 5",  "emoji": "💊", "task": f"Apply second dose of {medicine} spray", "type": "treatment"},
            {"day": "Day 6",  "emoji": "🌱", "task": "Apply potassium fertilizer to boost plant immunity", "type": "care"},
            {"day": "Day 7",  "emoji": "📸", "task": "Take new leaf photo and re-scan with Agri AI", "type": "rescan"},
            {"day": "Day 10", "emoji": "💊", "task": f"Final preventive spray of {medicine} if disease persists", "type": "treatment"},
            {"day": "Day 14", "emoji": "✅", "task": "Final inspection — if clear, resume normal care routine", "type": "check"},
        ],
        'Black_rot_Grape': [
            {"day": "Day 1",  "emoji": "💊", "task": f"Spray {medicine} on all grape clusters and leaves", "type": "treatment"},
            {"day": "Day 2",  "emoji": "✂️", "task": "Remove and burn all infected clusters immediately", "type": "cleanup"},
            {"day": "Day 3",  "emoji": "🔍", "task": "Check remaining clusters for new black spots", "type": "check"},
            {"day": "Day 4",  "emoji": "💊", "task": f"Apply second {medicine} spray", "type": "treatment"},
            {"day": "Day 5",  "emoji": "🌿", "task": "Clear all fallen leaves and fruit debris from vineyard", "type": "cleanup"},
            {"day": "Day 7",  "emoji": "📸", "task": "Re-scan leaf with Agri AI to check improvement", "type": "rescan"},
            {"day": "Day 10", "emoji": "💊", "task": "Preventive spray if humid weather continues", "type": "treatment"},
            {"day": "Day 14", "emoji": "✅", "task": "Full vineyard inspection — verify disease is controlled", "type": "check"},
        ],
        'Powdery_Mildew_Mango': [
            {"day": "Day 1",  "emoji": "💊", "task": f"Spray {medicine} on all white powder affected leaves", "type": "treatment"},
            {"day": "Day 2",  "emoji": "🌬️", "task": "Improve air circulation by pruning dense branches", "type": "care"},
            {"day": "Day 3",  "emoji": "🔍", "task": "Check if white powder is reducing on treated leaves", "type": "check"},
            {"day": "Day 4",  "emoji": "💊", "task": f"Second spray of {medicine}", "type": "treatment"},
            {"day": "Day 5",  "emoji": "🌿", "task": "Remove heavily infected leaves from tree", "type": "cleanup"},
            {"day": "Day 7",  "emoji": "📸", "task": "Re-scan with Agri AI to verify recovery", "type": "rescan"},
            {"day": "Day 10", "emoji": "💊", "task": "Final preventive spray to stop reoccurrence", "type": "treatment"},
            {"day": "Day 14", "emoji": "✅", "task": "Confirm disease eliminated — resume normal schedule", "type": "check"},
        ],
        'Powdery_Mildew_Cotton': [
            {"day": "Day 1",  "emoji": "💊", "task": f"Spray {medicine} on all infected cotton leaves", "type": "treatment"},
            {"day": "Day 2",  "emoji": "🌿", "task": "Remove and destroy infected plant parts", "type": "cleanup"},
            {"day": "Day 3",  "emoji": "🔍", "task": "Inspect field for spread of white coating", "type": "check"},
            {"day": "Day 5",  "emoji": "💊", "task": f"Apply second {medicine} spray in evening", "type": "treatment"},
            {"day": "Day 7",  "emoji": "📸", "task": "Re-scan leaf with Agri AI to check progress", "type": "rescan"},
            {"day": "Day 10", "emoji": "✅", "task": "Final field inspection and preventive spray", "type": "check"},
        ],
        'Powdery_Mildew_Pumpkin': [
            {"day": "Day 1",  "emoji": "🌿", "task": f"Spray {medicine} (Neem Oil) on all white patch areas", "type": "treatment"},
            {"day": "Day 2",  "emoji": "✂️", "task": "Remove most infected leaves to stop spreading", "type": "cleanup"},
            {"day": "Day 3",  "emoji": "🔍", "task": "Inspect vines for new patches", "type": "check"},
            {"day": "Day 4",  "emoji": "🌿", "task": f"Apply second {medicine} spray in morning", "type": "treatment"},
            {"day": "Day 7",  "emoji": "📸", "task": "Re-scan leaf with Agri AI", "type": "rescan"},
            {"day": "Day 10", "emoji": "✅", "task": "Final inspection — continue preventive sprays", "type": "check"},
        ],
        'RedRot_Sugarcane': [
            {"day": "Day 1",  "emoji": "🔍", "task": "Identify and mark all infected sugarcane stalks", "type": "check"},
            {"day": "Day 1",  "emoji": "🗑️", "task": "Remove and burn ALL infected stalks immediately", "type": "cleanup"},
            {"day": "Day 2",  "emoji": "💊", "task": f"Drench soil with {medicine} solution around removed stalks", "type": "treatment"},
            {"day": "Day 3",  "emoji": "🌿", "task": "Check neighboring stalks for red discoloration", "type": "check"},
            {"day": "Day 5",  "emoji": "💊", "task": f"Second soil drench with {medicine}", "type": "treatment"},
            {"day": "Day 7",  "emoji": "📸", "task": "Re-scan remaining stalks with Agri AI", "type": "rescan"},
            {"day": "Day 14", "emoji": "✅", "task": "Full field survey — verify no new infections", "type": "check"},
        ],
        'Rust_Sugarcane': [
            {"day": "Day 1",  "emoji": "💊", "task": f"Spray {medicine} on all rust-spotted leaves", "type": "treatment"},
            {"day": "Day 2",  "emoji": "🌿", "task": "Remove and burn heavily infected leaves", "type": "cleanup"},
            {"day": "Day 3",  "emoji": "🔍", "task": "Check if orange pustules are drying up", "type": "check"},
            {"day": "Day 5",  "emoji": "💊", "task": f"Apply second {medicine} spray", "type": "treatment"},
            {"day": "Day 7",  "emoji": "📸", "task": "Re-scan with Agri AI to verify improvement", "type": "rescan"},
            {"day": "Day 14", "emoji": "✅", "task": "Final inspection and preventive treatment", "type": "check"},
        ],
        'Sooty_Mould_Mango': [
            {"day": "Day 1",  "emoji": "🐛", "task": "Spray insecticide to kill mealybugs/scale insects first", "type": "treatment"},
            {"day": "Day 2",  "emoji": "🌿", "task": f"Spray {medicine} (Neem Oil) on black sooty leaves", "type": "treatment"},
            {"day": "Day 3",  "emoji": "🔍", "task": "Check if black coating is reducing", "type": "check"},
            {"day": "Day 4",  "emoji": "💧", "task": "Wash leaves gently with water to remove mould", "type": "care"},
            {"day": "Day 5",  "emoji": "🌿", "task": f"Second {medicine} spray on affected areas", "type": "treatment"},
            {"day": "Day 7",  "emoji": "📸", "task": "Re-scan leaf with Agri AI", "type": "rescan"},
            {"day": "Day 14", "emoji": "✅", "task": "Final check — confirm mould and insects eliminated", "type": "check"},
        ],
        'Target_Spot_Tomato': [
            {"day": "Day 1",  "emoji": "💊", "task": f"Spray {medicine} on all spotted tomato leaves", "type": "treatment"},
            {"day": "Day 2",  "emoji": "✂️", "task": "Remove infected lower leaves from plants", "type": "cleanup"},
            {"day": "Day 3",  "emoji": "🔍", "task": "Inspect plants for new circular spots", "type": "check"},
            {"day": "Day 4",  "emoji": "💊", "task": f"Apply second {medicine} spray", "type": "treatment"},
            {"day": "Day 5",  "emoji": "🪵", "task": "Stake plants properly for better air circulation", "type": "care"},
            {"day": "Day 7",  "emoji": "📸", "task": "Re-scan with Agri AI to check recovery", "type": "rescan"},
            {"day": "Day 10", "emoji": "💊", "task": "Final preventive spray", "type": "treatment"},
            {"day": "Day 14", "emoji": "✅", "task": "Confirm disease controlled — monitor weekly", "type": "check"},
        ],
        'Yellow_Sugarcane': [
            {"day": "Day 1",  "emoji": "🧪", "task": "Apply Urea fertilizer (10kg/acre) around base of plants", "type": "treatment"},
            {"day": "Day 2",  "emoji": "💧", "task": "Irrigate field properly after fertilizer application", "type": "care"},
            {"day": "Day 3",  "emoji": "🔍", "task": "Check yellowing — is it improving or spreading?", "type": "check"},
            {"day": "Day 5",  "emoji": "🧪", "task": "Apply micronutrient spray (Zinc + Iron)", "type": "treatment"},
            {"day": "Day 7",  "emoji": "📸", "task": "Re-scan leaf with Agri AI", "type": "rescan"},
            {"day": "Day 10", "emoji": "🌱", "task": "Apply balanced NPK fertilizer", "type": "care"},
            {"day": "Day 14", "emoji": "✅", "task": "Final field inspection — confirm recovery", "type": "check"},
        ],
        'Unknown_Disease': [
            {"day": "Day 1",  "emoji": "📸", "task": "Take clearer photo in good daylight and re-scan", "type": "check"},
            {"day": "Day 1",  "emoji": "📞", "task": "Call Kisan Call Center: 1800-180-1551 for expert advice", "type": "check"},
            {"day": "Day 2",  "emoji": "🌿", "task": "Apply general Neem Oil spray as preventive measure", "type": "treatment"},
            {"day": "Day 3",  "emoji": "🔍", "task": "Monitor disease spread carefully", "type": "check"},
            {"day": "Day 5",  "emoji": "🏥", "task": "Visit nearest KVK (Krishi Vigyan Kendra) with sample", "type": "check"},
            {"day": "Day 7",  "emoji": "📸", "task": "Re-scan with Agri AI after expert consultation", "type": "rescan"},
        ],
    }

    return schedules.get(disease, schedules['Unknown_Disease'])
# ================================
# 🔹 ROUTES
# ================================
@app.route('/')
def home():
    weather = get_live_weather("Pune")
    return render_template("home.html", weather=weather)


@app.route('/select_crop')
def select_crop():
    return render_template("select_crop.html")


@app.route('/upload_page', methods=['POST'])
def upload_page():
    crop     = request.form['crop']
    weather  = get_live_weather("Pune")
    tips     = get_farming_tips(weather, crop)
    calendar = get_crop_calendar(crop)
    prices   = get_mandi_prices()
    pest     = get_pest_alerts(crop)
    return render_template("index.html",
        crop=crop, weather=weather, tips=tips,
        calendar=calendar, prices=prices, pest=pest,
        now_month=datetime.now().month
    )


@app.route('/crop/<crop>')
def load_crop(crop):
    weather  = get_live_weather("Pune")
    tips     = get_farming_tips(weather, crop)
    calendar = get_crop_calendar(crop)
    prices   = get_mandi_prices()
    pest     = get_pest_alerts(crop)
    return render_template("index.html",
        crop=crop, weather=weather, tips=tips,
        calendar=calendar, prices=prices, pest=pest,
        now_month=datetime.now().month
    )


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    crop = request.form['crop']

    os.makedirs("static/uploads", exist_ok=True)
    image_path = os.path.join("static/uploads", file.filename)
    file.save(image_path)

    disease, confidence = run_prediction(image_path)

    if confidence < 40.0:
        disease    = 'Unknown_Disease'
        confidence = round(confidence, 2)

    healthy_plant = is_healthy(disease)

    medicine, medicine_img = medicine_map.get(
        disease, ('Consult Expert', 'default.jpg')
    )

    disease_info_map['Healthy_Leaf'] = {
        "symptoms":   "No disease symptoms detected",
        "cause":      "Plant is perfectly healthy",
        "prevention": "Continue regular care and monitoring"
    }

    info = disease_info_map.get(disease, {
        "symptoms":   "No data",
        "cause":      "Unknown",
        "prevention": "Consult expert"
    })

    medicine_img_path = os.path.join("static", "medicines", medicine_img)
    if not os.path.exists(medicine_img_path):
        medicine_img = "default.jpg"

    # ✅ Yield loss calculator
    yield_data = get_yield_loss(disease, confidence, crop)

    # ✅ Treatment schedule
    schedule = get_treatment_schedule(disease, medicine)

    if healthy_plant:
        healthy      = round(confidence, 2)
        diseased     = round(100 - confidence, 2)
        price        = 0
        stock        = 0
        display_name = f"Healthy {crop.capitalize()} Leaf"
    else:
        healthy      = round(100 - confidence, 2)
        diseased     = round(confidence, 2)
        price        = 250
        stock        = 10
        display_name = disease.replace('_', ' ')

    return render_template("result.html",
        image_path    = "/" + image_path,
        disease       = disease,
        display_name  = display_name,
        crop          = crop,
        confidence    = confidence,
        medicine      = medicine,
        price         = price,
        stock         = stock,
        medicine_img  = medicine_img,
        info          = info,
        healthy       = healthy,
        diseased      = diseased,
        healthy_plant = healthy_plant,
        yield_data    = yield_data,       # ✅ NEW
        schedule      = schedule          # ✅ NEW
    )
@app.route('/buy/<medicine>')
def buy(medicine):
    return f"<h2>🛒 Purchasing {medicine}... Payment page coming soon!</h2>"


if __name__ == '__main__':
    app.run(debug=True)