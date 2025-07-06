#!/usr/bin/env python3
"""
LNMarkets Official Client Integration
Using the official ln-markets Python library for real trading data
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # If python-dotenv is not installed, manually load .env
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

from lnmarkets import rest, websockets
from robust_logging import get_logger
from lnmarkets_time_sync import patch_lnmarkets_library, sync_time

logger = get_logger(__name__)

# Apply time synchronization patch on import
patch_lnmarkets_library()


def handle_timestamp_errors(func):
    """Decorator to handle timestamp errors and retry with time sync"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # Check if result contains timestamp error
            if isinstance(result, str) and 'outdated timestamp' in result.lower():
                logger = get_logger(func.__module__)
                logger.warning(f"⏰ Timestamp error in {func.__name__}, syncing time...")
                sync_time(force=True)
                # Retry the function
                result = func(*args, **kwargs)
                
            return result
            
        except Exception as e:
            if 'timestamp' in str(e).lower():
                logger = get_logger(func.__module__)
                logger.warning(f"⏰ Timestamp exception in {func.__name__}, syncing time...")
                sync_time(force=True)
                # Retry the function
                try:
                    return func(*args, **kwargs)
                except:
                    raise
            else:
                raise
                
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class LNMarketsOfficialClient:
    """
    Official LNMarkets API client for real trading data
    Uses the official ln-markets Python library
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 api_passphrase: str = None, network: str = None):
        
        # Get credentials from environment
        self.api_key = api_key or os.getenv('LNMARKETS_API_KEY', '')
        self.api_secret = api_secret or os.getenv('LNMARKETS_API_SECRET', '')
        self.api_passphrase = api_passphrase or os.getenv('LNMARKETS_API_PASSPHRASE', '')
        
        # Determine network from environment
        testnet_env = os.getenv('LNMARKETS_TESTNET', 'true').lower()
        if network is None:
            self.network = 'testnet' if testnet_env == 'true' else 'mainnet'
        else:
            self.network = network
        
        # Configure LNMarkets client
        self.options = {
            'key': self.api_key,
            'secret': self.api_secret,
            'passphrase': self.api_passphrase,
            'network': self.network  # 'testnet' or 'mainnet'
        }
        
        # Initialize clients
        self.rest_client = None
        self.ws_client = None
        self.is_connected = False
        
        logger.info(f"🔧 LNMarkets client configured for {self.network.upper()}")
    
    def initialize(self) -> bool:
        """Initialize the LNMarkets REST client"""
        try:
            if not self.api_key or not self.api_secret:
                logger.error("❌ LNMarkets API credentials not configured")
                return False
            
            # Initialize REST client
            self.rest_client = rest.LNMarketsRest(**self.options)
            
            # Test connection
            user_info = self.rest_client.get_user()
            if user_info:
                # Check for timestamp error
                if isinstance(user_info, str) and 'outdated timestamp' in user_info:
                    logger.warning("⏰ Timestamp sync error detected, forcing time sync...")
                    sync_time(force=True)
                    # Retry after sync
                    user_info = self.rest_client.get_user()
                
                if user_info and not (isinstance(user_info, str) and 'error' in user_info.lower()):
                    logger.info(f"✅ Connected to LNMarkets {self.network.upper()}")
                    
                    # Handle both dict and string responses
                    if isinstance(user_info, dict):
                        logger.info(f"👤 User ID: {user_info.get('uid', 'Unknown')}")
                        logger.info(f"💰 Balance: {user_info.get('balance', 0)} sats")
                    else:
                        logger.info(f"👤 Response: {user_info}")
                    
                    self.is_connected = True
                    return True
                else:
                    logger.error(f"❌ Failed to connect to LNMarkets: {user_info}")
                    return False
            else:
                logger.error("❌ Failed to connect to LNMarkets")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error initializing LNMarkets client: {e}")
            # Check if it's a timestamp error
            if 'timestamp' in str(e).lower():
                logger.warning("⏰ Timestamp error detected, forcing time sync...")
                sync_time(force=True)
            return False
    
    def get_user_info(self) -> Optional[Dict]:
        """Get user account information"""
        if not self.rest_client:
            return None
        
        try:
            result = self.rest_client.get_user()
            
            # Check for timestamp error and retry
            if isinstance(result, str) and 'outdated timestamp' in result:
                logger.warning("⏰ Timestamp error in get_user_info, syncing time...")
                sync_time(force=True)
                result = self.rest_client.get_user()
            
            # Handle both dict and string responses
            if isinstance(result, dict):
                return result
            elif isinstance(result, str):
                # Try to parse as JSON
                try:
                    import json
                    return json.loads(result)
                except:
                    return {'raw_response': result}
            return None
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            if 'timestamp' in str(e).lower():
                sync_time(force=True)
            return None
    
    def get_account_balance(self) -> Optional[Dict]:
        """Get account balance and equity"""
        user_info = self.get_user_info()
        if user_info:
            balance_sats = user_info.get('balance', 0)
            balance_btc = balance_sats / 100000000  # Convert sats to BTC
            
            # Get current BTC price for USD conversion
            ticker = self.get_ticker()
            btc_price = ticker.get('lastPrice', 50000) if ticker else 50000
            balance_usd = balance_btc * btc_price
            
            return {
                'balance_sats': balance_sats,
                'balance_btc': balance_btc,
                'balance_usd': balance_usd,
                'equity': balance_btc,
                'available_margin': balance_btc,
                'used_margin': 0,  # Will calculate from positions
                'timestamp': datetime.now().isoformat()
            }
        return None
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        if not self.rest_client:
            return []
        
        try:
            # Get running trades (positions) - need to specify type=running
            result = self.rest_client.futures_get_trades({'type': 'running'})
            
            # Handle both dict and string responses
            positions = []
            if isinstance(result, list):
                positions = result
            elif isinstance(result, dict):
                positions = result.get('data', [])
            elif isinstance(result, str):
                try:
                    import json
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        positions = parsed
                    else:
                        positions = parsed.get('data', [])
                except:
                    positions = []
            
            formatted_positions = []
            for pos in positions:
                formatted_positions.append({
                    'id': pos.get('id'),
                    'side': pos.get('side'),
                    'quantity': pos.get('quantity', 0),
                    'entry_price': pos.get('price', 0),
                    'leverage': pos.get('leverage', 1),
                    'margin': pos.get('margin', 0),
                    'pnl': pos.get('pl', 0),
                    'liquidation_price': pos.get('liquidation', 0),
                    'status': 'open',
                    'timestamp': datetime.fromtimestamp(
                        pos.get('creation_ts', 0) / 1000
                    ).isoformat() if pos.get('creation_ts') else datetime.now().isoformat()
                })
            
            return formatted_positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_ticker(self) -> Optional[Dict]:
        """Get current market ticker data"""
        if not self.rest_client:
            return None
        
        try:
            result = self.rest_client.futures_get_ticker()
            # Handle both dict and string responses
            if isinstance(result, dict):
                return result
            elif isinstance(result, str):
                # Try to parse as JSON
                try:
                    import json
                    return json.loads(result)
                except:
                    return {'raw_response': result}
            return None
        except Exception as e:
            logger.error(f"Error getting ticker: {e}")
            return None
    
    def get_market_data(self) -> Optional[Dict]:
        """Get formatted market data"""
        ticker = self.get_ticker()
        if ticker:
            # Map LNMarkets response fields to our format
            last_price = ticker.get('lastPrice', ticker.get('last', 0))
            bid_price = ticker.get('bidPrice', ticker.get('bid', 0))
            ask_price = ticker.get('askPrice', ticker.get('ask', 0))
            index_price = ticker.get('index', 0)
            
            return {
                'symbol': 'BTCUSD',
                'price': float(last_price),
                'bid': float(bid_price),
                'ask': float(ask_price),
                'high_24h': float(index_price),  # Use index as high for now
                'low_24h': float(index_price),   # Use index as low for now
                'volume_24h': 0,  # Not provided in this endpoint
                'change_24h': 0,  # Calculate if needed
                'change_24h_percent': 0,  # Calculate if needed
                'funding_rate': ticker.get('carryFeeRate', 0),
                'index_price': float(index_price),
                'carry_fee_rate': ticker.get('carryFeeRate', 0),
                'timestamp': datetime.now().isoformat()
            }
        return None
    
    def place_market_order(self, side: str, quantity: int, leverage: int = 1, 
                          take_profit: Optional[float] = None, stop_loss: Optional[float] = None) -> Optional[str]:
        """Place a market order using futures_new_trade"""
        if not self.rest_client:
            return None
        
        try:
            logger.info(f"📈 Placing {side} market order: {quantity} USD at {leverage}x leverage")
            
            # Build parameters for new trade
            params = {
                'type': 'm',  # market order
                'side': 'b' if side.lower() in ['buy', 'long'] else 's',
                'quantity': quantity,
                'leverage': leverage
            }
            
            # Add optional parameters
            if take_profit:
                params['takeprofit'] = take_profit
            if stop_loss:
                params['stoploss'] = stop_loss
                
            # Use the correct method name
            result = self.rest_client.futures_new_trade(params)
            
            # Handle response
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except:
                    pass
                    
            if result and isinstance(result, dict):
                order_id = result.get('id')
                logger.info(f"✅ Order placed successfully: {order_id}")
                logger.info(f"   Entry: {result.get('price', 'N/A')}")
                logger.info(f"   Liquidation: {result.get('liquidation', 'N/A')}")
                return order_id
            else:
                logger.error(f"❌ Failed to place order. API Response: {result}")
                logger.error(f"   Response type: {type(result)}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def close_position(self, position_id: str) -> bool:
        """Close a specific position using futures_close"""
        if not self.rest_client:
            return False
        
        try:
            logger.info(f"🔒 Closing position: {position_id}")
            
            # Use the correct method with params
            result = self.rest_client.futures_close({'id': position_id})
            
            # Handle response
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except:
                    pass
                    
            if result:
                logger.info(f"✅ Position {position_id} closed successfully")
                if isinstance(result, dict):
                    logger.info(f"   Final P&L: {result.get('pl', 'N/A')}")
                return True
            else:
                logger.error(f"❌ Failed to close position {position_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def close_all_positions(self) -> Dict[str, Any]:
        """Close all open positions using futures_close_all"""
        try:
            logger.info("🔒 Closing all open positions...")
            
            # Use the dedicated close all method
            result = self.rest_client.futures_close_all()
            
            if result:
                logger.info("✅ All positions closed successfully")
                return {'success': True, 'result': result}
            else:
                logger.error("❌ Failed to close all positions")
                return {'success': False, 'result': None}
                
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get recent trade history"""
        if not self.rest_client:
            return []
        
        try:
            # Get all trades (both running and closed)
            all_trades_result = self.rest_client.futures_get_trades({})
            
            # Handle different response types
            all_trades = []
            if isinstance(all_trades_result, list):
                all_trades = all_trades_result
            elif isinstance(all_trades_result, dict):
                all_trades = all_trades_result.get('data', [])
            elif isinstance(all_trades_result, str):
                try:
                    import json
                    parsed = json.loads(all_trades_result)
                    if isinstance(parsed, list):
                        all_trades = parsed
                    else:
                        all_trades = parsed.get('data', [])
                except:
                    all_trades = []
            
            # Sort by creation timestamp
            sorted_trades = sorted(
                all_trades,
                key=lambda x: x.get('creation_ts', 0),
                reverse=True
            )
            
            return sorted_trades[:limit]
            
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
    
    def add_margin_to_position(self, position_id: str, amount: int) -> bool:
        """Add margin to an existing position"""
        if not self.rest_client:
            return False
            
        try:
            logger.info(f"💰 Adding {amount} sats margin to position {position_id}")
            
            result = self.rest_client.futures_add_margin({
                'id': position_id,
                'amount': amount
            })
            
            if result:
                logger.info(f"✅ Margin added successfully")
                return True
            else:
                logger.error(f"❌ Failed to add margin")
                return False
                
        except Exception as e:
            logger.error(f"Error adding margin: {e}")
            return False
    
    def update_position(self, position_id: str, take_profit: Optional[float] = None, 
                       stop_loss: Optional[float] = None) -> bool:
        """Update position with new TP/SL"""
        if not self.rest_client:
            return False
            
        try:
            updated = False
            
            if take_profit is not None:
                logger.info(f"📊 Updating take profit to {take_profit}")
                result = self.rest_client.futures_update_trade({
                    'id': position_id,
                    'type': 'takeprofit',
                    'value': take_profit
                })
                if result:
                    updated = True
                    
            if stop_loss is not None:
                logger.info(f"🛡️ Updating stop loss to {stop_loss}")
                result = self.rest_client.futures_update_trade({
                    'id': position_id,
                    'type': 'stoploss',
                    'value': stop_loss
                })
                if result:
                    updated = True
                    
            return updated
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            return False
    
    def place_limit_order(self, side: str, price: float, quantity: int, leverage: int = 1,
                         take_profit: Optional[float] = None, stop_loss: Optional[float] = None) -> Optional[str]:
        """Place a limit order"""
        if not self.rest_client:
            return None
            
        try:
            logger.info(f"📊 Placing {side} limit order: {quantity} USD @ {price} with {leverage}x leverage")
            
            params = {
                'type': 'l',  # limit order
                'side': 'b' if side.lower() in ['buy', 'long'] else 's',
                'quantity': quantity,
                'leverage': leverage,
                'price': price
            }
            
            if take_profit:
                params['takeprofit'] = take_profit
            if stop_loss:
                params['stoploss'] = stop_loss
                
            result = self.rest_client.futures_new_trade(params)
            
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except:
                    pass
                    
            if result and isinstance(result, dict):
                order_id = result.get('id')
                logger.info(f"✅ Limit order placed: {order_id}")
                return order_id
            else:
                logger.error(f"❌ Failed to place limit order: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if not self.rest_client:
            return False
            
        try:
            logger.info(f"❌ Cancelling order: {order_id}")
            
            result = self.rest_client.futures_cancel({'id': order_id})
            
            if result:
                logger.info(f"✅ Order cancelled successfully")
                return True
            else:
                logger.error(f"❌ Failed to cancel order")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_market_info(self) -> Optional[Dict]:
        """Get market configuration and limits"""
        if not self.rest_client:
            return None
            
        try:
            result = self.rest_client.futures_get_market()
            
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except:
                    pass
                    
            return result
            
        except Exception as e:
            logger.error(f"Error getting market info: {e}")
            return None
    
    def get_leaderboard(self) -> Optional[List]:
        """Get trading leaderboard"""
        if not self.rest_client:
            return None
            
        try:
            result = self.rest_client.futures_get_leaderboard()
            
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except:
                    pass
                    
            return result
            
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return None
    
    async def start_websocket(self, on_message_callback=None):
        """Start WebSocket connection for real-time data"""
        try:
            logger.info("🔌 Starting LNMarkets WebSocket connection...")
            
            # Initialize WebSocket client  
            self.ws_client = websockets.LNMarketsWebsocket(**self.options)
            
            # Define message handler
            def handle_message(message):
                try:
                    data = json.loads(message) if isinstance(message, str) else message
                    logger.debug(f"📡 WebSocket message: {data}")
                    
                    if on_message_callback:
                        on_message_callback(data)
                        
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
            
            # Connect and subscribe
            self.ws_client.connect()
            self.ws_client.subscribe(['index'], handle_message)
            
            logger.info("✅ WebSocket connected and subscribed to index")
            
        except Exception as e:
            logger.error(f"Error starting WebSocket: {e}")
    
    def stop_websocket(self):
        """Stop WebSocket connection"""
        if self.ws_client:
            try:
                self.ws_client.close()
                logger.info("🔌 WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")


# Global instance
_lnmarkets_official_client = None

def get_lnmarkets_official_client() -> LNMarketsOfficialClient:
    """Get or create LNMarkets official client instance"""
    global _lnmarkets_official_client
    if _lnmarkets_official_client is None:
        _lnmarkets_official_client = LNMarketsOfficialClient()
    return _lnmarkets_official_client


async def main():
    """Test the official LNMarkets client"""
    print("🚀 Testing LNMarkets Official Client")
    print("=" * 50)
    
    client = get_lnmarkets_official_client()
    
    # Test connection
    connected = client.initialize()
    
    if connected:
        print("\n📊 Getting Live Data from LNMarkets...")
        
        # Get account data
        balance = client.get_account_balance()
        if balance:
            print(f"💰 Balance: {balance['balance_btc']:.8f} BTC (${balance['balance_usd']:,.2f})")
        
        # Get positions
        positions = client.get_positions()
        print(f"📈 Open Positions: {len(positions)}")
        for pos in positions:
            print(f"   {pos['side'].upper()} {pos['quantity']} @ {pos['entry_price']} (P&L: {pos['pnl']})")
        
        # Get market data
        market = client.get_market_data()
        if market:
            print(f"💰 BTC Price: ${market['price']:,.2f}")
            print(f"📈 24h Change: {market['change_24h_percent']:.2f}%")
        
        # Test WebSocket (briefly)
        print("\n🔌 Testing WebSocket connection...")
        
        def on_ws_message(data):
            print(f"📡 WebSocket data: {data}")
        
        await client.start_websocket(on_ws_message)
        
        # Let it run for a few seconds
        await asyncio.sleep(5)
        
        client.stop_websocket()
        
        print("\n✅ LNMarkets Official Client Test Complete!")
        
    else:
        print("\n❌ Could not connect to LNMarkets API")
        print("\n🔑 To configure LNMarkets API credentials:")
        print("1. Sign up at https://lnmarkets.com/")
        print("2. Go to API settings and create API key")
        print("3. Set environment variables:")
        print("   export LNMARKETS_API_KEY='your_key'")
        print("   export LNMARKETS_API_SECRET='your_secret'")
        print("   export LNMARKETS_API_PASSPHRASE='your_passphrase'")


if __name__ == '__main__':
    asyncio.run(main())