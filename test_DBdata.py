import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="uiuc_chatbot",
    user="postgres",
    password="dev123"
)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM uiuc_chatbot;")
print(f"Local Postgres records: {cursor.fetchone()[0]}")