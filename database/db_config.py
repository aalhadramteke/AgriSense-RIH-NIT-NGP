import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",   # 🔥 put correct password
    database="agri_system"
)

# ✅ important fix
cursor = conn.cursor(buffered=True)

print("✅ Database Connected Successfully")