#!/usr/bin/env python3
"""
LNMarkets Time Synchronization Module
Handles time offset calculation and patching for API requests
"""

import time
import requests
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LNMarketsTimeSync:
    """
    Handles time synchronization for LNMarkets API
    Calculates time offset between local system and server
    """
    
    def __init__(self):
        self.time_offset = 0
        self.last_sync = 0
        self.sync_interval = 3600  # Re-sync every hour
        
    def get_server_time(self) -> Optional[int]:
        """Get current time from a reliable time server"""
        try:
            # Try multiple time servers
            time_servers = [
                'https://worldtimeapi.org/api/timezone/UTC',
                'https://api.binance.com/api/v3/time',  
                'https://api.kraken.com/0/public/Time'
            ]
            
            for server in time_servers:
                try:
                    response = requests.get(server, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Parse based on server
                        if 'worldtimeapi' in server:
                            return int(data['unixtime'] * 1000)
                        elif 'binance' in server:
                            return data['serverTime']
                        elif 'kraken' in server:
                            return int(data['result']['unixtime'] * 1000)
                            
                except Exception as e:
                    logger.debug(f"Failed to get time from {server}: {e}")
                    continue
                    
            # If all fail, try getting time from HTTP headers
            try:
                response = requests.head('https://www.google.com', timeout=5)
                if 'Date' in response.headers:
                    server_time = datetime.strptime(
                        response.headers['Date'],
                        '%a, %d %b %Y %H:%M:%S %Z'
                    )
                    return int(server_time.timestamp() * 1000)
            except:
                pass
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting server time: {e}")
            return None
    
    def calculate_offset(self) -> int:
        """Calculate time offset between local and server time"""
        try:
            # Get server time
            server_time = self.get_server_time()
            if server_time is None:
                logger.warning("Could not get server time, using 0 offset")
                return 0
                
            # Get local time
            local_time = int(datetime.now().timestamp() * 1000)
            
            # Calculate offset
            offset = server_time - local_time
            
            logger.info(f"Time offset calculated: {offset}ms (server ahead by {offset/1000:.2f}s)")
            
            return offset
            
        except Exception as e:
            logger.error(f"Error calculating time offset: {e}")
            return 0
    
    def sync_time(self, force: bool = False) -> int:
        """Synchronize time offset if needed"""
        current_time = time.time()
        
        # Check if sync is needed
        if not force and (current_time - self.last_sync) < self.sync_interval:
            return self.time_offset
            
        # Calculate new offset
        self.time_offset = self.calculate_offset()
        self.last_sync = current_time
        
        return self.time_offset
    
    def get_synced_timestamp(self) -> str:
        """Get current timestamp adjusted for server time"""
        # Ensure we have a recent sync
        self.sync_time()
        
        # Get current time with offset
        adjusted_time = int((datetime.now().timestamp() * 1000) + self.time_offset)
        
        return str(adjusted_time)


# Global instance
_time_sync = LNMarketsTimeSync()


def get_synced_timestamp() -> str:
    """Get synchronized timestamp for LNMarkets API"""
    return _time_sync.get_synced_timestamp()


def sync_time(force: bool = False) -> int:
    """Force time synchronization"""
    return _time_sync.sync_time(force=force)


def patch_lnmarkets_library():
    """
    Monkey patch the LNMarkets library to use synchronized timestamps
    This modifies the library's timestamp generation in-memory
    """
    try:
        import lnmarkets.rest
        
        # Store original method
        original_request_options = lnmarkets.rest.LNMarketsRest._request_options
        
        def patched_request_options(self, **options):
            """Patched version that uses synchronized timestamp"""
            credentials = options.get('credentials')
            method = options.get('method')
            path = options.get('path')
            params = options.get('params')
            opts = { 'headers': {} }

            if method != 'DELETE':
                opts['headers']['Content-Type'] = 'application/json'
      
            if self.custom_headers:
              opts['headers'].update(**self.custom_headers)

            if method in ['GET', 'DELETE']:
                from urllib.parse import urlencode
                data = urlencode(params)
            elif method in ['POST', 'PUT']:
                import json
                data = json.dumps(params, separators=(',', ':'))
                
            if credentials and not self.skip_api_key:
                if not self.key:
                    'You need an API key to use an authenticated route'
                elif not self.secret:
                    'You need an API secret to use an authenticated route'
                elif not self.passphrase:
                    'You need an API passphrase to use an authenticated route'
                
                # Use synchronized timestamp instead of local time
                ts = get_synced_timestamp()
                
                import hashlib
                import hmac
                from base64 import b64encode
                         
                payload = ts + method + '/' + self.version + path + data
                hashed = hmac.new(bytes(self.secret, 'utf-8'), bytes(payload, 'utf-8'), hashlib.sha256).digest()
                signature = b64encode(hashed)
                
                opts['headers']['LNM-ACCESS-KEY'] = self.key
                opts['headers']['LNM-ACCESS-PASSPHRASE'] = self.passphrase
                opts['headers']['LNM-ACCESS-TIMESTAMP'] = ts
                opts['headers']['LNM-ACCESS-SIGNATURE'] = signature
                
            opts['ressource'] = 'https://' + self.hostname + '/' + self.version + path

            if method in ['GET', 'DELETE'] and params:
                opts['ressource'] += '?' + data
                
            return opts
        
        # Apply patch
        lnmarkets.rest.LNMarketsRest._request_options = patched_request_options
        
        logger.info("✅ LNMarkets library patched for time synchronization")
        
        # Force initial sync
        sync_time(force=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch LNMarkets library: {e}")
        return False


if __name__ == "__main__":
    # Test the time sync
    print("Testing LNMarkets Time Synchronization...")
    
    # Get server time
    server_time = _time_sync.get_server_time()
    if server_time:
        local_time = int(datetime.now().timestamp() * 1000)
        print(f"Local time:  {local_time}")
        print(f"Server time: {server_time}")
        print(f"Difference:  {(server_time - local_time) / 1000:.2f} seconds")
    
    # Test sync
    offset = sync_time(force=True)
    print(f"Time offset: {offset}ms")
    
    # Test timestamp generation
    ts1 = get_synced_timestamp()
    print(f"Synced timestamp: {ts1}")
    
    # Patch library
    if patch_lnmarkets_library():
        print("✅ Library patched successfully")
    else:
        print("❌ Failed to patch library")