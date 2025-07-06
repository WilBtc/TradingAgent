#!/usr/bin/env python3
"""
Redis Caching Layer for Trading Bot
High-performance caching to improve API response times
"""

import redis
import json
import time
import logging
from typing import Any, Optional, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradingCache:
    """Redis-based cache for trading data"""
    
    def __init__(self, host='localhost', port=6380, password=None):
        self.redis_available = False
        self.in_memory_cache = {}
        self.cache_ttls = {}
        
        try:
            self.redis_client = redis.Redis(
                host=host, 
                port=port, 
                password=password,
                decode_responses=True,
                socket_timeout=2
            )
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info("✅ Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable, using in-memory cache: {e}")
            self.redis_client = None
    
    def _get_key(self, category: str, symbol: str = None) -> str:
        """Generate cache key"""
        if symbol:
            return f"trading:{category}:{symbol.replace('/', '_')}"
        return f"trading:{category}"
    
    def set(self, category: str, data: Any, ttl: int = 300, symbol: str = None) -> bool:
        """Set cache data with TTL"""
        key = self._get_key(category, symbol)
        
        try:
            if self.redis_available and self.redis_client:
                # Use Redis
                serialized = json.dumps(data)
                result = self.redis_client.setex(key, ttl, serialized)
                logger.debug(f"📝 Redis cache set: {key} (TTL: {ttl}s)")
                return result
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            self.redis_available = False
        
        # Fallback to in-memory cache
        expiry = time.time() + ttl
        self.in_memory_cache[key] = {
            'data': data,
            'expiry': expiry
        }
        logger.debug(f"📝 Memory cache set: {key} (TTL: {ttl}s)")
        return True
    
    def get(self, category: str, symbol: str = None) -> Optional[Any]:
        """Get cache data"""
        key = self._get_key(category, symbol)
        
        try:
            if self.redis_available and self.redis_client:
                # Try Redis first
                cached = self.redis_client.get(key)
                if cached:
                    logger.debug(f"🎯 Redis cache hit: {key}")
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            self.redis_available = False
        
        # Fallback to in-memory cache
        if key in self.in_memory_cache:
            cached_item = self.in_memory_cache[key]
            if time.time() < cached_item['expiry']:
                logger.debug(f"🎯 Memory cache hit: {key}")
                return cached_item['data']
            else:
                # Expired
                del self.in_memory_cache[key]
                logger.debug(f"⏰ Cache expired: {key}")
        
        logger.debug(f"❌ Cache miss: {key}")
        return None
    
    def delete(self, category: str, symbol: str = None) -> bool:
        """Delete cache entry"""
        key = self._get_key(category, symbol)
        
        try:
            if self.redis_available and self.redis_client:
                self.redis_client.delete(key)
        except Exception:
            pass
        
        if key in self.in_memory_cache:
            del self.in_memory_cache[key]
        
        logger.debug(f"🗑️ Cache deleted: {key}")
        return True
    
    def clear_all(self) -> bool:
        """Clear all cache"""
        try:
            if self.redis_available and self.redis_client:
                # Delete all keys matching pattern
                keys = self.redis_client.keys("trading:*")
                if keys:
                    self.redis_client.delete(*keys)
        except Exception:
            pass
        
        self.in_memory_cache.clear()
        logger.info("🧹 All cache cleared")
        return True
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            'redis_available': self.redis_available,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if self.redis_available and self.redis_client:
                info = self.redis_client.info()
                stats.update({
                    'redis_memory_used': info.get('used_memory_human', 'N/A'),
                    'redis_keys': info.get('db0', {}).get('keys', 0),
                    'redis_hits': info.get('keyspace_hits', 0),
                    'redis_misses': info.get('keyspace_misses', 0)
                })
        except Exception:
            pass
        
        stats['memory_cache_size'] = len(self.in_memory_cache)
        return stats

# Cache TTL configurations (in seconds)
CACHE_TTLS = {
    'market_data': 10,      # Market prices update frequently
    'account_data': 30,     # Account balance doesn't change often
    'trading_signals': 15,  # Signals need to be relatively fresh
    'multi_timeframe': 20,  # Technical analysis can be cached briefly
    'indicators': 300,      # Technical indicators are stable
    'ai_analysis': 30,      # AI analysis can be cached
    'positions': 5,         # Position data should be very fresh
    'sniper_entry': 10      # Entry signals need to be fresh
}

# Global cache instance
trading_cache = TradingCache()

def cache_data(category: str, data: Any, symbol: str = None) -> bool:
    """Helper function to cache data with appropriate TTL"""
    ttl = CACHE_TTLS.get(category, 60)  # Default 1 minute
    return trading_cache.set(category, data, ttl, symbol)

def get_cached_data(category: str, symbol: str = None) -> Optional[Any]:
    """Helper function to get cached data"""
    return trading_cache.get(category, symbol)

def clear_cache(category: str = None, symbol: str = None) -> bool:
    """Helper function to clear cache"""
    if category:
        return trading_cache.delete(category, symbol)
    else:
        return trading_cache.clear_all()

if __name__ == "__main__":
    # Test the cache
    cache = TradingCache()
    
    # Test set/get
    test_data = {"price": 109500, "timestamp": time.time()}
    cache.set("test", test_data, 10)
    
    retrieved = cache.get("test")
    print(f"Test data: {retrieved}")
    
    # Test stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")