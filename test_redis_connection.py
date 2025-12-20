"""
Test script to verify the new Redis connection
"""
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Redis import RedisManager

def test_redis_connection():
    """Test the new Redis connection"""
    print("Testing new Redis connection...")
    print("Connection details:")
    print("Host: redis-13713.c212.ap-south-1-1.ec2.redns.redis-cloud.com")
    print("Port: 13713")
    print("Username: default")
    print("Password: ghHRoTYqi8LcLiaBv9qbZDUFRIcfSBnA")
    print("-" * 50)
    
    try:
        # Initialize Redis manager
        redis_manager = RedisManager()
        
        if redis_manager.is_available:
            print("✅ Redis connection successful!")
            
            # Test basic operations
            test_key = "test_connection_key"
            test_value = "test_connection_value"
            
            # Test SET operation
            set_result = redis_manager.set(test_key, test_value)
            if set_result:
                print("✅ SET operation successful")
            else:
                print("❌ SET operation failed")
                return False
            
            # Test GET operation
            get_result = redis_manager.get(test_key)
            if get_result == test_value:
                print("✅ GET operation successful")
                print(f"Retrieved value: {get_result}")
            else:
                print("❌ GET operation failed")
                print(f"Expected: {test_value}, Got: {get_result}")
                return False
            
            # Test SETEX operation (with expiration)
            setex_result = redis_manager.setex(f"{test_key}_expiry", 60, "expiring_value")
            if setex_result:
                print("✅ SETEX operation successful")
            else:
                print("❌ SETEX operation failed")
                return False
            
            print("\n🎉 All Redis operations completed successfully!")
            print("The new Redis connection is working properly.")
            return True
            
        else:
            print("❌ Failed to establish Redis connection")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Redis connection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Redis Connection Test")
    print("=" * 50)
    
    success = test_redis_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ TEST PASSED: Redis connection is working!")
    else:
        print("❌ TEST FAILED: Redis connection issues detected")
    
    print("\nConnection string used:")
    print("redis://default:ghHRoTYqi8LcLiaBv9qbZDUFRIcfSBnA@redis-13713.c212.ap-south-1-1.ec2.redns.redis-cloud.com:13713")
