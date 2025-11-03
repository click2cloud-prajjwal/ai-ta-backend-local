"""
List all available API endpoints in the Flask application
"""
import requests
import json

print("=" * 70)
print("🔍 TESTING AI-TA BACKEND ENDPOINTS")
print("=" * 70)

BASE_URL = "http://localhost:8000"

# Common endpoints to test
endpoints = [
    "/",
    "/health",
    "/healthcheck",
    "/api/health",
    "/getTopContexts",
    "/getContexts",
    "/chat",
    "/ingest",
    "/delete",
    "/projects",
    "/courses",
]

print(f"\n🌐 Testing endpoints at: {BASE_URL}")
print("-" * 70)

available_endpoints = []
for endpoint in endpoints:
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=2)
        status = response.status_code
        
        if status == 200:
            print(f"✅ {endpoint:30} - Status: {status} (Working!)")
            available_endpoints.append(endpoint)
        elif status == 404:
            print(f"❌ {endpoint:30} - Status: {status} (Not found)")
        elif status == 405:
            print(f"⚠️  {endpoint:30} - Status: {status} (Method not allowed - try POST)")
            available_endpoints.append(f"{endpoint} (POST)")
        else:
            print(f"⚠️  {endpoint:30} - Status: {status}")
            available_endpoints.append(endpoint)
    except requests.exceptions.Timeout:
        print(f"⏱️  {endpoint:30} - Timeout")
    except requests.exceptions.ConnectionError:
        print(f"❌ {endpoint:30} - Connection refused")
    except Exception as e:
        print(f"❌ {endpoint:30} - Error: {str(e)[:30]}")

print("\n" + "=" * 70)
print(f"✅ AVAILABLE ENDPOINTS: {len(available_endpoints)}")
print("=" * 70)

if available_endpoints:
    print("\n📋 Working endpoints:")
    for ep in available_endpoints:
        print(f"   • {BASE_URL}{ep}")
else:
    print("\n⚠️  No endpoints found. Check Flask routes in the code.")

print("\n💡 TIP: For full API documentation, visit:")
print("   📚 https://docs.uiuc.chat/")
print("=" * 70)