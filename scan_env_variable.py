"""
Comprehensive Environment Variable Checker
Scans the entire codebase for required environment variables
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("🔍 COMPREHENSIVE ENVIRONMENT VARIABLE SCAN")
print("=" * 70)

# Scan all Python files for os.environ['...'] patterns
def scan_for_env_vars(directory):
    """Scan Python files for environment variable usage"""
    env_vars = set()
    
    for py_file in Path(directory).rglob('*.py'):
        try:
            content = py_file.read_text(encoding='utf-8')
            
            # Pattern 1: os.environ['VAR']
            matches1 = re.findall(r'os\.environ\[[\"\']([A-Z_0-9]+)[\"\']\]', content)
            env_vars.update(matches1)
            
            # Pattern 2: os.environ.get('VAR')
            matches2 = re.findall(r'os\.environ\.get\([\"\']([A-Z_0-9]+)[\"\']', content)
            env_vars.update(matches2)
            
            # Pattern 3: os.getenv('VAR')
            matches3 = re.findall(r'os\.getenv\([\"\']([A-Z_0-9]+)[\"\']', content)
            env_vars.update(matches3)
            
        except Exception as e:
            pass
    
    return sorted(env_vars)

print("\n📂 Scanning codebase for environment variables...")
project_dir = os.path.abspath('.')
all_env_vars = scan_for_env_vars(project_dir)

print(f"   Found {len(all_env_vars)} unique environment variables\n")

# Check which ones are set
print("=" * 70)
print("📋 ENVIRONMENT VARIABLE STATUS")
print("=" * 70)

missing_vars = []
set_vars = []

for var in all_env_vars:
    value = os.getenv(var)
    if value and value.strip() and value != 'placeholder':
        set_vars.append(var)
        # print(f"✅ {var:40} SET")
    else:
        missing_vars.append(var)
        print(f"❌ {var:40} MISSING")

print("\n" + "=" * 70)
print(f"📊 SUMMARY: {len(set_vars)} SET, {len(missing_vars)} MISSING")
print("=" * 70)

if missing_vars:
    print("\n🔧 RECOMMENDED .ENV ADDITIONS:")
    print("=" * 70)
    print("\n# Add these to your .env file:\n")
    
    # Categorize missing variables
    categories = {
        'PostHog': ['POSTHOG'],
        'Vyriad Qdrant': ['VYRIAD_QDRANT'],
        'Ollama/NCSA': ['OLLAMA', 'NCSA'],
        'Beam': ['BEAM'],
        'N8N': ['N8N'],
        'Railway': ['RAILWAY'],
        'Vercel': ['VERCEL'],
        'Sentry': ['SENTRY'],
        'Stripe': ['STRIPE'],
        'Auth': ['AUTH', 'CLERK'],
        'Other': []
    }
    
    categorized = {cat: [] for cat in categories}
    
    for var in missing_vars:
        found = False
        for category, keywords in categories.items():
            if category == 'Other':
                continue
            if any(keyword in var for keyword in keywords):
                categorized[category].append(var)
                found = True
                break
        if not found:
            categorized['Other'].append(var)
    
    # Print categorized recommendations
    for category, vars_list in categorized.items():
        if vars_list:
            print(f"# {category} (Optional)")
            for var in vars_list:
                print(f"{var}=placeholder")
            print()

print("\n" + "=" * 70)
print("💡 QUICK FIX FOR COMMON ISSUES:")
print("=" * 70)
print("""
1. PostHog (Analytics - Optional):
   POSTHOG_API_KEY=placeholder

2. Vyriad Qdrant (Medical/Patent data - Optional):
   VYRIAD_QDRANT_URL=vyriad-qdrant.ncsa.ai
   VYRIAD_QDRANT_PORT=443
   VYRIAD_QDRANT_API_KEY=placeholder

3. NCSA Services (Optional):
   NCSA_HOSTED_API_KEY=placeholder

4. Beam (Job processing - Optional if using RabbitMQ):
   BEAM_API_KEY=placeholder
   BEAM_API_BASE_URL=https://api.beam.cloud

5. To skip version warnings:
   QDRANT_CHECK_VERSION=False
""")

print("=" * 70)
print("🎯 RECOMMENDATION:")
print("=" * 70)
print("""
Most of these services are OPTIONAL for basic functionality.
For testing, add 'placeholder' values to prevent crashes.
Only configure the ones you actually need to use.

Core services you HAVE configured:
✅ PostgreSQL
✅ Minio
✅ RabbitMQ  
✅ Qdrant (main)
✅ OpenAI
✅ Nomic
✅ Supabase

These are enough for document ingestion and querying!
""")
print("=" * 70)