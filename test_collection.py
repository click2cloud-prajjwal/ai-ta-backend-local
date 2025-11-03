# check_qdrant_data.py
from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://13c93d79-8178-4532-aa95-3c0bf9831a23.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.xmei39ZyBDKYuZwpNUOnnGkhai2R0EFjCsQQcXWafKo"
)

collection_name = "uiuc-chatbot"

# 1. Check collection stats
info = client.get_collection(collection_name)
print(f"📊 Total vectors in collection: {info.vectors_count}")
print(f"📊 Points count: {info.points_count}")

# 2. Scroll through first 10 points to see actual data
points = client.scroll(
    collection_name=collection_name,
    limit=10,
    with_payload=True,
    with_vectors=False
)[0]

print(f"\n🔍 Found {len(points)} sample points:")
for i, point in enumerate(points, 1):
    print(f"\n--- Point {i} (ID: {point.id}) ---")
    payload = point.payload if isinstance(point.payload, dict) else {}
    print(f"Payload keys: {list(payload.keys())}")
    print(f"course_name: {payload.get('course_name')}")
    print(f"conversation_id: {payload.get('conversation_id')}")
    print(f"doc_groups: {payload.get('doc_groups')}")
    print(f"text preview: {payload.get('text', '')[:100]}...")

# 3. Check for test-course specifically
from qdrant_client.models import Filter, FieldCondition, MatchValue

test_course_points = client.scroll(
    collection_name=collection_name,
    scroll_filter=Filter(
        must=[
            FieldCondition(
                key="course_name",
                match=MatchValue(value="test-course")
            )
        ]
    ),
    limit=5,
    with_payload=True,
    with_vectors=False
)[0]

print(f"\n🎯 Points with course_name='test-course': {len(test_course_points)}")