import redis

class RedisManager:
    
    """Safe Redis client manager with availability checks"""
    
    def __init__(self, host="redis-14494.c330.asia-south1-1.gce.cloud.redislabs.com", port=14494, decode_responses=True):
        self.host = host
        self.port = port
        self.decode_responses = decode_responses
        self.client = None
        self.is_available = False
        self.connect()
    
    def connect(self):
        """Attempt to connect to Redis"""
        try:
            self.client = redis.StrictRedis(
                host=self.host, 
                port=self.port, 
                decode_responses=self.decode_responses,
                username="default",
                password="UlAWjg62BmBM91qHyVDLWR74g96ErhcC",
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test the connection
            self.client.ping()
            self.is_available = True
            print("✅ Redis connection established successfully")
        except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
            print(f"❌ Redis connection failed: {e}")
            self.is_available = False
            self.client = None
    
    def check_availability(self):
        """Check if Redis is available and attempt reconnection if needed"""
        if not self.is_available or not self.client:
            self.connect()
        
        if self.is_available:
            try:
                self.client.ping()
                return True
            except (redis.ConnectionError, redis.TimeoutError, Exception):
                print("❌ Redis ping failed, marking as unavailable")
                self.is_available = False
                return False
        return False
    
    def get(self, key):
        """Safe Redis GET operation"""
        if not self.check_availability():
            print(f"⚠️ Redis unavailable - skipping GET operation for key: {key}")
            return None
        
        try:
            return self.client.get(key)
        except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
            print(f"❌ Redis GET failed for key {key}: {e}")
            self.is_available = False
            return None
    
    def set(self, key, value):
        """Safe Redis SET operation"""
        if not self.check_availability():
            print(f"⚠️ Redis unavailable - skipping SET operation for key: {key}")
            return False
        
        try:
            result = self.client.set(key, value)
            return result
        except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
            print(f"❌ Redis SET failed for key {key}: {e}")
            self.is_available = False
            return False
    
    def setex(self, key, seconds, value):
        """Safe Redis SETEX operation"""
        if not self.check_availability():
            print(f"⚠️ Redis unavailable - skipping SETEX operation for key: {key}")
            return False
        
        try:
            result = self.client.setex(key, seconds, value)
            return result
        except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
            print(f"❌ Redis SETEX failed for key {key}: {e}")
            self.is_available = False
            return False