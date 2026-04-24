from database.db_config import conn, cursor

# 🔷 =========================
# 🔐 LOGIN SYSTEM
# 🔷 =========================
def check_login(username, password):
    query = "SELECT role FROM users WHERE username=%s AND password=%s"
    cursor.execute(query, (username, password))
    return cursor.fetchone()


# 🔷 =========================
# 💊 GET MEDICINE DETAILS
# 🔷 =========================
def get_medicine_details(name):
    query = "SELECT name, price, stock, image FROM medicines WHERE name=%s"
    cursor.execute(query, (name,))
    return cursor.fetchone()


# 🔷 =========================
# 🛒 BUY MEDICINE
# 🔷 =========================
def buy_medicine(name):
    # Check stock first
    cursor.execute("SELECT stock FROM medicines WHERE name=%s", (name,))
    result = cursor.fetchone()

    if result and result[0] > 0:
        # Reduce stock
        cursor.execute("UPDATE medicines SET stock = stock - 1 WHERE name=%s", (name,))
        
        # Insert into sales
        cursor.execute(
            "INSERT INTO sales (medicine_name, quantity, total) VALUES (%s, %s, %s)",
            (name, 1, 0)
        )

        conn.commit()
        return True
    else:
        return False


# 🔷 =========================
# ➕ ADD MEDICINE (ADMIN)
# 🔷 =========================
def add_medicine(name, price, stock, image):
    query = "INSERT INTO medicines (name, price, stock, image) VALUES (%s,%s,%s,%s)"
    cursor.execute(query, (name, price, stock, image))
    conn.commit()


# 🔷 =========================
# 🔄 UPDATE STOCK (ADMIN)
# 🔷 =========================
def update_stock(name, stock):
    query = "UPDATE medicines SET stock=%s WHERE name=%s"
    cursor.execute(query, (stock, name))
    conn.commit()


# 🔷 =========================
# 📋 GET ALL MEDICINES (ADMIN VIEW)
# 🔷 =========================
def get_all_medicines():
    query = "SELECT * FROM medicines"
    cursor.execute(query)
    return cursor.fetchall()

def check_login(username, password):
    print("INPUT:", username, password)

    cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
    user = cursor.fetchone()

    print("DB RESULT:", user)

    return user
def insert_crop_result(crop, result):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO crop_analysis (crop, result) VALUES (%s, %s)",
        (crop, result)
    )

    conn.commit()
    conn.close()


def get_crop_stats(crop):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM crop_analysis WHERE crop=%s", (crop,))
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM crop_analysis WHERE crop=%s AND result='Healthy'", (crop,))
    healthy = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM crop_analysis WHERE crop=%s AND result='Diseased'", (crop,))
    diseased = cursor.fetchone()[0]

    conn.close()

    if total == 0:
        return 0, 0

    return round((healthy/total)*100,2), round((diseased/total)*100,2)