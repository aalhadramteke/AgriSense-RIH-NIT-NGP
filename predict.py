import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ================================
# LOAD MODEL - SAFE WITH FULL PATH
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'model.h5')

MODEL_LOADED = False
model = None

if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        MODEL_LOADED = True
        print("✅ Model Loaded Successfully from:", model_path)
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        MODEL_LOADED = False
else:
    print(f"⚠️ model.h5 not found at: {model_path}")
    print("⚠️ Running with dummy predictions")
    MODEL_LOADED = False

# ================================
# CLASS LABELS
# ================================
class_labels = {
    0: 'Anthracnose_Mango',
    1: 'Black_rot_Grape',
    2: 'Powdery_Mildew_Mango',
    3: 'Powdery_Mildew_Cotton',
    4: 'Powdery_Mildew_Pumpkin',
    5: 'RedRot_Sugarcane',
    6: 'Rust_Sugarcane',
    7: 'Sooty_Mould_Mango',
    8: 'Target_Spot_Tomato',
    9: 'Unknown_Disease',
    10: 'Yellow_Sugarcane'
}

# ================================
# PREDICT FUNCTION
# ================================
def predict_disease(img_path, crop=None):
    if not MODEL_LOADED:
        print("⚠️ Model not loaded - returning dummy result")
        return 'Unknown_Disease', 75.0

    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        class_index = np.argmax(predictions)
        confidence  = float(predictions[class_index] * 100)
        disease     = class_labels.get(class_index, 'Unknown_Disease')

        print(f"✅ Predicted: {disease} ({confidence:.2f}%)")
        return disease, round(confidence, 2)

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return 'Unknown_Disease', 0.0

# ================================
# MEDICINE + IMAGE
# ================================
def get_medicine(disease):
    medicine_map = {
        'Anthracnose_Mango':      ('Carbendazim',   'carbendazim.jpg'),
        'Black_rot_Grape':        ('Mancozeb',       'mancozeb.jpg'),
        'Powdery_Mildew_Mango':   ('Sulfur',         'sulfur.jpg'),
        'Powdery_Mildew_Cotton':  ('Hexaconazole',   'hexaconazole.jpg'),
        'Powdery_Mildew_Pumpkin': ('Neem Oil',       'neem.jpg'),
        'RedRot_Sugarcane':       ('Carbendazim',    'carbendazim.jpg'),
        'Rust_Sugarcane':         ('Propiconazole',  'propiconazole.jpg'),
        'Sooty_Mould_Mango':      ('Neem Oil',       'neem.jpg'),
        'Target_Spot_Tomato':     ('Chlorothalonil', 'chlorothalonil.jpg'),
        'Yellow_Sugarcane':       ('Urea Spray',     'urea.jpg')
    }
    return medicine_map.get(disease, ('Consult Expert', 'default.jpg'))
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ================================
# LOAD MODEL
# ================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'model.h5')

MODEL_LOADED = False
model        = None

if os.path.exists(model_path):
    try:
        model        = tf.keras.models.load_model(model_path)
        MODEL_LOADED = True
        print("✅ Model Loaded Successfully from:", model_path)
    except Exception as e:
        print(f"❌ Model load failed: {e}")
else:
    print(f"⚠️ model.h5 not found - Running with dummy predictions")

# ================================
# ✅ CLASS LABELS - WITH HEALTHY_LEAF
# ================================
# ⚠️ IMPORTANT: These numbers must match your classes.json file!
# Open C:\project\model\classes.json and copy exact numbers
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

# ================================
# ✅ IS HEALTHY CHECK
# ================================
def is_healthy(disease):
    return disease == 'Healthy_Leaf'

# ================================
# PREDICT FUNCTION
# ================================
def predict_disease(img_path, crop=None):
    if not MODEL_LOADED:
        print("⚠️ Model not loaded - returning dummy result")
        return 'Unknown_Disease', 75.0

    try:
        img       = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        class_index = int(np.argmax(predictions))
        confidence  = float(predictions[class_index] * 100)
        disease     = class_labels.get(class_index, 'Unknown_Disease')

        print(f"✅ Predicted: {disease} ({confidence:.2f}%)")
        return disease, round(confidence, 2)

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return 'Unknown_Disease', 0.0

# ================================
# MEDICINE MAP
# ================================
def get_medicine(disease):
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
    return medicine_map.get(disease, ('Consult Expert', 'default.jpg'))

# ================================
# DISEASE INFO
# ================================
def get_disease_info(disease):
    info_map = {
        # ✅ Healthy info
        'Healthy_Leaf': {
            "symptoms": "No disease symptoms detected",
            "cause":    "Plant is perfectly healthy",
            "prevention": "Continue regular care and monitoring"
        },
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
        }
    }
    return info_map.get(disease, {
        "symptoms": "No data available",
        "cause": "Unknown",
        "prevention": "Consult an agriculture expert"
    })
# ================================
# DISEASE INFO
# ================================
def get_disease_info(disease):
    info_map = {
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
        }
    }
    return info_map.get(disease, {
        "symptoms": "No data available",
        "cause": "Unknown",
        "prevention": "Consult an agriculture expert"
    })

