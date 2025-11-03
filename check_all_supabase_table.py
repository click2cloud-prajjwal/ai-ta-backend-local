# check_all_supabase.py
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_API_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_API_KEY must be set in environment variables")

client = create_client(supabase_url, supabase_key)

# Check multiple possible table names
tables = ["documents", "uiuc_chatbot", "materials", "llm_convo_monitor"]

print("Checking all Supabase tables...")
print("="*60)

for table_name in tables:
    try:
        result = client.table(table_name).select("*").limit(3).execute()
        
        if result.data:
            print(f"\n✅ Table: {table_name}")
            print(f"   Rows: {len(result.data)}")
            for i, row in enumerate(result.data, 1):
                print(f"\n   Row {i}:")
                # Show key fields
                for key in ['course_name', 'readable_filename', 's3_path', 'url']:
                    if key in row:
                        print(f"     {key}: {row[key]}")
        else:
            print(f"\n⚠️  Table '{table_name}' exists but is EMPTY")
            
    except Exception as e:
        print(f"\n❌ Table '{table_name}': {str(e)[:100]}")

print("\n" + "="*60)