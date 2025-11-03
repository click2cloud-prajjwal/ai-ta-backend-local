import os
import json
from concurrent.futures import as_completed
import requests
import psycopg2
from dotenv import load_dotenv

from ai_ta_backend.executors.thread_pool_executor import ThreadPoolExecutorAdapter

load_dotenv()


def send_request(webcrawl_url, payload):
    response = requests.post(webcrawl_url, json=payload)
    try:
        return response.json()
    except Exception:
        return {"error": response.text, "status_code": response.status_code}


def fetch_base_urls(project_name: str):
    """
    Tries to fetch base URLs from Supabase.
    If Supabase is unavailable, falls back to local PostgreSQL.
    Returns a dict: { base_url: [document_groups] }
    """

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_API_KEY")

    # Try Supabase first
    if supabase_url and supabase_key:
        try:
            from supabase import create_client
            supabase_client = create_client(supabase_url, supabase_key)
            response = supabase_client.rpc("get_base_url_with_doc_groups", {"p_course_name": project_name}).execute()
            if response.data:
                print("✅ Retrieved base URLs from Supabase")
                return response.data
        except Exception as e:
            print("⚠️ Supabase unavailable, falling back to local Postgres:", e)

    # Fallback: Local PostgreSQL connection
    print("🔄 Using local PostgreSQL fallback")

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_ENDPOINT", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USERNAME", "ai_user"),
        password=os.getenv("POSTGRES_PASSWORD", "ai_password"),
        dbname=os.getenv("POSTGRES_DATABASE", "uiuc_chatbot")
    )
    cursor = conn.cursor()

    # Example query equivalent to your Supabase RPC
    query = """
        SELECT base_url, document_groups
        FROM document_sources
        WHERE course_name = %s;
    """
    cursor.execute(query, (project_name,))
    rows = cursor.fetchall()

    # Convert to dict
    base_urls = {}
    for base_url, document_groups in rows:
        try:
            if isinstance(document_groups, str):
                document_groups = json.loads(document_groups)
        except Exception:
            document_groups = [document_groups]
        base_urls[base_url] = document_groups

    cursor.close()
    conn.close()
    print(f"✅ Retrieved {len(base_urls)} base URLs from local Postgres")
    return base_urls


def webscrape_documents(project_name: str):
    print(f"Scraping documents for project: {project_name}")

    base_urls = fetch_base_urls(project_name)
    if not base_urls:
        print("❌ No base URLs found for project.")
        return "No URLs to scrape."

    webcrawl_url = "https://crawlee-production.up.railway.app/crawl"
    payload_template = {
        "params": {
            "url": "",
            "scrapeStrategy": "same-hostname",
            "maxPagesToCrawl": 15000,
            "maxTokens": 2000000,
            "courseName": project_name
        }
    }

    processed_file_name = f"processed_urls_{''.join(e if e.isalnum() else '_' for e in project_name.lower())}.txt"
    if not os.path.exists(processed_file_name):
        open(processed_file_name, 'w').close()

    print(f"Processed file name: {processed_file_name}")

    tasks = []
    count = 0
    batch_size = 10

    with ThreadPoolExecutorAdapter(max_workers=batch_size) as executor:
        for base_url, doc_groups in base_urls.items():
            payload = payload_template.copy()
            payload["params"]["url"] = base_url
            payload["params"]["documentGroups"] = doc_groups

            with open(processed_file_name, 'r') as file:
                skip_urls = set(line.strip() for line in file)

            if base_url in skip_urls:
                print(f"Skipping URL: {base_url}")
                continue

            with open(processed_file_name, 'a') as file:
                file.write(base_url + '\n')

            print("Payload:", json.dumps(payload, indent=2))
            tasks.append(executor.submit(send_request, webcrawl_url, payload.copy()))
            count += 1

            if count % batch_size == 0:
                for future in as_completed(tasks):
                    response = future.result()
                    print("Response from crawl:", response)
                tasks = []

        # Process remaining tasks
        for future in as_completed(tasks):
            response = future.result()
            print("Response from crawl:", response)

    return "Webscrape done."
