# create_collection_with_index.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

print("🔄 Creating Qdrant collection with proper indexes...")

client = QdrantClient(
    url="https://13c93d79-8178-4532-aa95-3c0bf9831a23.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.xmei39ZyBDKYuZwpNUOnnGkhai2R0EFjCsQQcXWafKo"
)

collection_name = "uiuc-chatbot"

try:
    # Delete old collection if exists
    try:
        client.delete_collection(collection_name)
        print(f"🗑️ Deleted old collection: {collection_name}")
    except:
        print(f"ℹ️ No existing collection to delete")
    
    # Create collection with proper vector size (1536 for text-embedding-3-small)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{collection_name}' created!")
    
    # Create payload index for conversation_id (required for filtering)
    client.create_payload_index(
        collection_name=collection_name,
        field_name="conversation_id",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("✅ Created index on 'conversation_id'")
    
    # Create other useful indexes
    client.create_payload_index(
        collection_name=collection_name,
        field_name="course_name",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("✅ Created index on 'course_name'")
    
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_groups",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("✅ Created index on 'doc_groups'")
    
    # Verify
    collection_info = client.get_collection(collection_name)
    print(f"\n📊 Collection Info:")
    print(f"   Vectors: {collection_info.vectors_count}")
    print(f"   Status: {collection_info.status}")
    
    print("\n✅ All done! Collection is ready for use.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()