#!/usr/bin/env python3
"""
Test script for Colab integration
"""
import os
import sys
import requests
import json
from PIL import Image
import io
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_colab_connection():
    """Test basic connection to Colab service"""
    colab_url = os.getenv('COLAB_API_URL')
    
    if not colab_url:
        print("❌ COLAB_API_URL not set in environment variables")
        return False
    
    print(f"🔗 Testing connection to: {colab_url}")
    
    try:
        response = requests.get(f"{colab_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Colab service is healthy!")
            print(f"   Device: {health_data.get('device', 'Unknown')}")
            print(f"   Model loaded: {health_data.get('model_loaded', False)}")
            print(f"   CUDA available: {health_data.get('cuda_available', False)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

def test_local_webapp():
    """Test local web app health"""
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Local web app is healthy!")
            
            colab_info = health_data.get('colab_service', {})
            print(f"   Colab available: {colab_info.get('is_available', False)}")
            print(f"   Processing modes: {health_data.get('processing_modes', {})}")
            return True
        else:
            print(f"❌ Local web app health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Local web app connection failed: {str(e)}")
        return False

def create_test_image():
    """Create a simple test image"""
    # Create a simple colored square
    img = Image.new('RGB', (224, 224), color='green')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

def test_image_processing():
    """Test image processing through web app"""
    print("\n🖼️  Testing image processing...")
    
    try:
        # Create test image
        test_image = create_test_image()
        
        # Upload to web app
        files = {'image': ('test.png', test_image, 'image/png')}
        response = requests.post("http://localhost:5000/analyze", files=files, timeout=60)
        
        if response.status_code == 200:
            # Check if it's JSON response
            if response.headers.get('content-type', '').startswith('application/json'):
                result = response.json()
                print("✅ Image processing successful!")
                print(f"   Predicted class: {result.get('predicted_class', 'Unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.2%}")
                print(f"   Processing source: {result.get('processing_source', 'Unknown')}")
                print(f"   Background removed: {result.get('has_background_removal', False)}")
                return True
            else:
                print("✅ Image processing successful (HTML response)")
                return True
        else:
            print(f"❌ Image processing failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Image processing error: {str(e)}")
        return False

def test_colab_config():
    """Test Colab configuration endpoint"""
    print("\n⚙️  Testing Colab configuration...")
    
    try:
        # Get current config
        response = requests.get("http://localhost:5000/api/colab/config", timeout=5)
        if response.status_code == 200:
            config = response.json()
            print("✅ Colab configuration accessible!")
            print(f"   URL: {config.get('url', 'Not set')}")
            print(f"   Available: {config.get('is_available', False)}")
            print(f"   Last health check: {config.get('last_health_check', 'Never')}")
            return True
        else:
            print(f"❌ Colab config failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Colab config error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Colab Integration\n")
    
    results = []
    
    # Test 1: Colab connection
    print("1. Testing Colab service connection...")
    results.append(test_colab_connection())
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Local web app
    print("2. Testing local web app...")
    results.append(test_local_webapp())
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Colab configuration
    results.append(test_colab_config())
    
    print("\n" + "="*50 + "\n")
    
    # Test 4: Image processing
    results.append(test_image_processing())
    
    print("\n" + "="*50 + "\n")
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Colab integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        
        if not results[0]:  # Colab connection failed
            print("\n💡 Tips:")
            print("   - Make sure your Colab notebook is running")
            print("   - Check that the ngrok URL is correct in .env")
            print("   - Verify the Colab notebook has finished installing packages")
        
        if not results[1]:  # Local web app failed
            print("\n💡 Tips:")
            print("   - Make sure your Flask app is running (python app.py)")
            print("   - Check that port 5000 is available")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)