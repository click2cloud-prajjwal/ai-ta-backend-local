import os
import supabase

# Simple env vars (replace with your actual values)
os.environ['SUPABASE_URL'] = 'http://ovwqzgbduxnxxqpqiewy.supabase.co'
os.environ['SUPABASE_API_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im92d3F6Z2JkdXhueHhxcHFpZXd5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA2MzY2NjAsImV4cCI6MjA3NjIxMjY2MH0.MBAyqmmrJ8pPil63N7-mdezULLkdOhgfcmxZ-uAi7kg'
os.environ['MATERIALS_SUPABASE_TABLE'] = 'uiuc_chatbot'

client = supabase.create_client(
    os.environ['SUPABASE_URL'],
    os.environ['SUPABASE_API_KEY']
)

table = os.environ.get('MATERIALS_SUPABASE_TABLE', 'documents')

print(f"Checking table: {table}")
print("="*50)

try:
    result = client.table(table).select('*').limit(10).execute()
    
    if result.data:
        print(f"Found {len(result.data)} documents:")
        for doc in result.data:
            print(f"  - {doc.get('readable_filename')} ({doc.get('course_name')})")
    else:
        print("  (no documents yet)")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
