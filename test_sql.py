"""
Database Connection Tester
Tests connection to your database and identifies the type
"""
import sys

print("=" * 70)
print("🔍 DATABASE CONNECTION TESTER")
print("=" * 70)

DB_HOST = "98.70.12.116"
DB_PORT = 1642
DB_USER = "sa"  # or "user" - you mentioned both
DB_PASSWORD = input("Enter database password: ")
DB_NAME = "DBAgriPilot-dev"

print(f"\n📋 Testing connection to:")
print(f"   Host: {DB_HOST}")
print(f"   Port: {DB_PORT}")
print(f"   Database: {DB_NAME}")
print(f"   User: {DB_USER}")

# Test 1: Try PostgreSQL
print("\n" + "=" * 70)
print("🐘 Test 1: PostgreSQL")
print("=" * 70)
try:
    import psycopg2
    print("✅ psycopg2 library found")
    
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=5
    )
    print("✅ ✅ ✅ CONNECTION SUCCESSFUL - This is PostgreSQL!")
    
    # Get version
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    row = cursor.fetchone()
    if row and row[0] is not None:
        version = row[0]
    else:
        version = "unknown"
    print(f"   Version: {version}")
    
    cursor.close()
    conn.close()
    
    print("\n🎯 RESULT: Use PostgreSQL configuration")
    sys.exit(0)
    
except ImportError:
    print("⚠️  psycopg2 not installed")
    print("   Install with: pip install psycopg2-binary")
except Exception as e:
    print(f"❌ PostgreSQL connection failed: {e}")

# Test 2: Try MySQL
print("\n" + "=" * 70)
print("🐬 Test 2: MySQL")
print("=" * 70)
try:
    import pymysql
    print("✅ pymysql library found")
    
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        connect_timeout=5
    )
    cursor = conn.cursor()
    cursor.execute("SELECT VERSION();")
    row = cursor.fetchone()
    if row and row[0] is not None:
        version = row[0]
    else:
        version = "unknown"
    print(f"   Version: {version}")
    
    cursor.close()
    conn.close()
    
    print("\n🎯 RESULT: This is MySQL")
    sys.exit(0)
    
except ImportError:
    print("⚠️  pymysql not installed")
    print("   Install with: pip install pymysql")
except Exception as e:
    print(f"❌ MySQL connection failed: {e}")

# Test 3: Try SQL Server (MSSQL)
print("\n" + "=" * 70)
print("🏢 Test 3: SQL Server (MSSQL)")
print("=" * 70)
try:
    import pyodbc
    print("✅ pyodbc library found")
    
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={DB_HOST},{DB_PORT};"
        f"DATABASE={DB_NAME};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD}"
    )
    conn = pyodbc.connect(conn_str, timeout=5)
    cursor = conn.cursor()
    cursor.execute("SELECT @@VERSION;")
    row = cursor.fetchone()
    if row and row[0] is not None:
        version = str(row[0])
    else:
        version = "unknown"
    print(f"   Version: {version[:100]}...")
    
    cursor.close()
    conn.close()
    
    print("\n🎯 RESULT: This is SQL Server")
    sys.exit(0)
    
except ImportError:
    print("⚠️  pyodbc not installed")
    print("   Install with: pip install pyodbc")
except Exception as e:
    print(f"❌ SQL Server connection failed: {e}")

# Summary
print("\n" + "=" * 70)
print("❌ Could not connect to database with any driver")
print("=" * 70)
print("\n💡 Recommendations:")
print("   1. Install database drivers:")
print("      pip install psycopg2-binary pymysql pyodbc")
print("   2. Check if the database is accessible from your network")
print("   3. Verify credentials are correct")
print("\n   OR use local Docker PostgreSQL instead:")
print("      docker run -d --name postgres -e POSTGRES_PASSWORD=dev123 -e POSTGRES_DB=uiuc_chatbot -p 5432:5432 postgres:15")
print("=" * 70)