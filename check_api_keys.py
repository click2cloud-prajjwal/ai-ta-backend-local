"""
Check Required API Keys for Ingestion
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("🔑 CHECKING REQUIRED API KEYS")
print("=" * 70)

required_keys = {
    "OPENAI_API_KEY": "OpenAI API (for embeddings/processing)",
    "NOMIC_API_KEY": "Nomic API (for embeddings)",
    "EMBEDDING_MODEL": "Embedding model configuration"
}

optional_keys = {
    "EMBEDDING_API_BASE": "Custom embedding API endpoint",
    "NCSA_HOSTED_API_KEY": "NCSA hosted services"
}

print("\n📋 Required Keys:")
print("-" * 70)

all_good = True
for key, description in required_keys.items():
    value = os.getenv(key)
    if value and value.strip() and value != "placeholder":
        # Mask the key
        if len(value) > 8:
            masked = f"{value[:4]}...{value[-4:]}"
        else:
            masked = "***"
        print(f"✅ {key:25} = {masked:20} ({description})")
    else:
        print(f"❌ {key:25} = NOT SET          ({description})")
        all_good = False

print("\n📋 Optional Keys:")
print("-" * 70)
for key, description in optional_keys.items():
    value = os.getenv(key)
    if value and value.strip() and value != "placeholder":
        masked = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
        print(f"✅ {key:25} = {masked:20} ({description})")
    else:
        print(f"⚠️  {key:25} = NOT SET          ({description})")

print("\n" + "=" * 70)
if all_good:
    print("✅ ALL REQUIRED KEYS ARE SET!")
    print("You should be able to ingest documents now.")
else:
    print("❌ SOME REQUIRED KEYS ARE MISSING!")
    print("\n🔧 To fix:")
    print("1. Get OpenAI API key from: https://platform.openai.com/api-keys")
    print("2. Get Nomic API key from: https://atlas.nomic.ai/")
    print("3. Add them to your .env file:")
    print("   OPENAI_API_KEY=sk-your-key-here")
    print("   NOMIC_API_KEY=nk-your-key-here")
    print("   EMBEDDING_MODEL=nomic-embed-text-v1.5")
print("=" * 70)