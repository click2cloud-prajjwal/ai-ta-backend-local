"""
Minio Connection Test Script
Run this to verify your Minio setup is working correctly
"""
import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from datetime import datetime

def test_minio_connection():
    """Test connection to Minio and perform basic operations"""
    
    print("=" * 60)
    print("🧪 MINIO CONNECTION TEST")
    print("=" * 60)
    
    # Configuration
    endpoint_url = os.getenv('MINIO_URL', 'http://localhost:9000')
    access_key = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')
    bucket_name = os.getenv('S3_BUCKET_NAME', 'uiuc-chat')
    
    print(f"\n📋 Configuration:")
    print(f"   Endpoint: {endpoint_url}")
    print(f"   Access Key: {access_key}")
    print(f"   Bucket: {bucket_name}")
    
    try:
        # Create S3 client
        print("\n🔌 Connecting to Minio...")
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        print("✅ Connected successfully!")
        
        # Check if bucket exists
        print(f"\n🪣 Checking bucket '{bucket_name}'...")
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ Bucket '{bucket_name}' exists and is accessible")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"❌ Bucket '{bucket_name}' does not exist")
                print(f"   Please create it in the Minio console at http://localhost:9001")
                return False
            else:
                raise
        
        # Test file upload
        print("\n📤 Testing file upload...")
        test_filename = f"test_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        test_content = f"Hello from Minio! Test performed at {datetime.now()}"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=test_filename,
            Body=test_content.encode('utf-8'),
            ContentType='text/plain'
        )
        print(f"✅ Uploaded: {test_filename}")
        
        # Test file download
        print("\n📥 Testing file download...")
        response = s3_client.get_object(Bucket=bucket_name, Key=test_filename)
        downloaded_content = response['Body'].read().decode('utf-8')
        
        if downloaded_content == test_content:
            print("✅ Downloaded and verified content matches!")
        else:
            print("❌ Content mismatch!")
            return False
        
        # List files in bucket
        print(f"\n📂 Listing files in '{bucket_name}'...")
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
        
        if 'Contents' in response:
            print(f"   Found {len(response['Contents'])} file(s):")
            for obj in response['Contents'][:5]:  # Show first 5
                size_kb = obj['Size'] / 1024
                print(f"   • {obj['Key']} ({size_kb:.2f} KB)")
            if len(response['Contents']) > 5:
                print(f"   ... and {len(response['Contents']) - 5} more")
        else:
            print("   Bucket is empty (besides our test file)")
        
        # Test file deletion
        print(f"\n🗑️  Cleaning up test file...")
        s3_client.delete_object(Bucket=bucket_name, Key=test_filename)
        print(f"✅ Deleted: {test_filename}")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED! Minio is working correctly!")
        print("=" * 60)
        return True
        
    except ClientError as e:
        print(f"\n❌ AWS/Minio Error: {e}")
        print(f"   Error Code: {e.response['Error']['Code']}")
        print(f"   Error Message: {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {type(e).__name__}")
        print(f"   {str(e)}")
        return False

if __name__ == "__main__":
    # Load environment variables from .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("📝 Loaded .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed, using default values")
        print("   Install it with: pip install python-dotenv")
    
    success = test_minio_connection()
    exit(0 if success else 1)