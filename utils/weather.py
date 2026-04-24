import requests

API_KEY = "YOUR_API_KEY"   # 🔥 Put your real API key here

# 🌦️ Get weather
def get_weather(city="Nagpur"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
    data = response.json()

    temp = data['main']['temp']
    humidity = data['main']['humidity']
    condition = data['weather'][0]['main']

    return temp, humidity, condition


# ⚠️ Alert system (THIS WAS MISSING)
def get_alert(temp, humidity):
    if humidity > 70:
        return "⚠️ High humidity - Risk of fungal disease"
    elif temp > 35:
        return "⚠️ High temperature - Water stress risk"
    else:
        return "✅ Weather is normal"