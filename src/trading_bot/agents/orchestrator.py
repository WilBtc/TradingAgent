import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

from selfhosted_multi_agent_system import SelfHostedMultiAgentSystem, TradingDecision
from lnmarkets_data_provider import LNMarketsDataProvider
from lnmarkets_official_client import LNMarketsOfficialClient
from redis_cache import TradingCache


class SelfHostedTradingOrchestrator:
    """Self-hosted trading orchestrator for LNMarkets - no external dependencies"""
    
    def __init__(self, config_file: str = None):
        self.logger = logging.getLogger("selfhosted_orchestrator")
        self.cache = TradingCache()
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_history = []
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration optimized for self-hosted Bitcoin trading"""
        default_config = {
            # LNMarkets credentials from environment
            'lnmarkets_api_key': os.getenv('LNMARKETS_API_KEY'),
            'lnmarkets_api_secret': os.getenv('LNMARKETS_API_SECRET'),
            'lnmarkets_passphrase': os.getenv('LNMARKETS_API_PASSPHRASE'),
            'lnmarkets_testnet': os.getenv('LNMARKETS_TESTNET', 'false').lower() == 'true',
            
            # DeepSeek for AI analysis (optional)
            'deepseek_api_key': os.getenv('DEEPSEEK_API_KEY'),
            
            # Trading parameters optimized for Bitcoin
            'max_position_size': 0.02,  # 2% max position
            'max_daily_loss': 0.03,     # 3% max daily loss
            'max_leverage': 33,         # 33x max leverage
            'min_confidence_threshold': 0.6,  # 60% confidence minimum
            
            # Self-hosted specific settings
            'trading_mode': 'conservative',  # conservative, balanced, aggressive
            'enable_ai_analysis': True,      # Use DeepSeek for enhanced analysis
            'update_interval_minutes': 3,   # How often to run analysis
            'enable_auto_trading': False,   # Require manual approval by default
            
            # Bitcoin-specific settings
            'symbol': 'BTCUSD',
            'focus_timeframes': ['1m', '5m', '15m', '1h'],  # LNMarkets timeframes
            'news_sources': 'crypto_only',  # Focus on crypto news
            
            # Risk management
            'max_concurrent_positions': 3,
            'stop_loss_percentage': 0.02,   # 2% stop loss
            'take_profit_percentage': 0.04, # 4% take profit
            'emergency_stop_loss': 0.05,    # 5% emergency stop
            
            # Data sources (all free/self-hosted)
            'use_lnmarkets_data': True,
            'use_coingecko_data': True,
            'use_fear_greed_index': True,
            'use_crypto_news': True,
            'cache_duration_minutes': 5
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
                self.logger.info(f"Loaded config from {config_file}")
            except Exception as e:
                self.logger.error(f"Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def _initialize_components(self):
        """Initialize self-hosted trading system components"""
        try:
            # Self-hosted multi-agent system
            self.multi_agent_system = SelfHostedMultiAgentSystem(self.config)
            self.logger.info("Self-hosted multi-agent system initialized")
            
            # Self-hosted data provider
            self.data_provider = LNMarketsDataProvider()
            self.logger.info("LNMarkets data provider initialized")
            
            # Trading client
            self.trading_client = LNMarketsOfficialClient(
                api_key=self.config.get('lnmarkets_api_key'),
                api_secret=self.config.get('lnmarkets_api_secret'),
                api_passphrase=self.config.get('lnmarkets_passphrase'),
                network='testnet' if self.config.get('lnmarkets_testnet') else 'mainnet'
            )
            self.logger.info("LNMarkets trading client initialized")
            
            self.logger.info("All self-hosted components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    async def run_selfhosted_analysis_cycle(self, symbol: str = None) -> Dict[str, Any]:
        """Run complete self-hosted analysis cycle"""
        if symbol is None:
            symbol = self.config.get('symbol', 'BTCUSD')
            
        try:
            self.logger.info(f"Starting self-hosted analysis cycle for {symbol}")
            
            # Step 1: Gather comprehensive market data from self-hosted sources
            market_data = await self._gather_selfhosted_data(symbol)
            
            # Step 2: Run multi-agent analysis
            trading_decision = await self._run_agent_analysis(symbol)
            
            # Step 3: Validate with risk management
            validation_result = await self._validate_decision(symbol, trading_decision, market_data)
            
            # Step 4: Execute if approved and auto-trading enabled
            execution_result = None
            if validation_result.get('approved', False) and self.config.get('enable_auto_trading', False):
                execution_result = await self._execute_trade(trading_decision)
            elif validation_result.get('approved', False):
                execution_result = {
                    'status': 'approved_pending_manual',
                    'message': 'Trade approved but manual execution required',
                    'decision': trading_decision.__dict__ if trading_decision else None
                }
            
            # Compile comprehensive results
            cycle_result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'system_type': 'self_hosted_lnmarkets',
                'market_data_summary': self._summarize_market_data(market_data),
                'trading_decision': trading_decision.__dict__ if trading_decision else None,
                'validation_result': validation_result,
                'execution_result': execution_result,
                'data_sources': ['LNMarkets', 'CoinGecko', 'Crypto News', 'Fear & Greed'],
                'performance_metrics': self._calculate_cycle_metrics(),
                'account_status': await self._get_account_status()
            }
            
            # Store in performance history
            self.performance_history.append(cycle_result)
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            self.logger.info(f"Self-hosted analysis cycle completed for {symbol}")
            return cycle_result
            
        except Exception as e:
            self.logger.error(f"Self-hosted analysis cycle failed: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'system_type': 'self_hosted_lnmarkets',
                'error': str(e),
                'status': 'failed'
            }
    
    async def _gather_selfhosted_data(self, symbol: str) -> Dict[str, Any]:
        """Gather all data using only self-hosted and free sources"""
        try:
            # Use self-hosted data provider
            comprehensive_data = await self.data_provider.get_comprehensive_data(symbol)
            return comprehensive_data
        except Exception as e:
            self.logger.error(f"Self-hosted data gathering failed: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    async def _run_agent_analysis(self, symbol: str) -> Optional[TradingDecision]:
        """Run self-hosted multi-agent analysis"""
        try:
            return await self.multi_agent_system.generate_trading_decision(symbol)
        except Exception as e:
            self.logger.error(f"Agent analysis failed: {e}")
            return None
    
    async def _validate_decision(self, symbol: str, decision: TradingDecision, market_data: Dict) -> Dict[str, Any]:
        """Validate trading decision with self-hosted risk management"""
        validation_result = {
            'approved': False,
            'reasons': [],
            'adjustments': {},
            'risk_level': 'high'
        }
        
        try:
            if not decision:
                validation_result['reasons'].append('No trading decision generated')
                return validation_result
            
            # Check confidence threshold
            min_confidence = self.config.get('min_confidence_threshold', 0.6)
            if decision.confidence < min_confidence:
                validation_result['reasons'].append(f'Confidence {decision.confidence:.2f} below threshold {min_confidence}')
                return validation_result
            
            # Check position size limits
            max_position = self.config.get('max_position_size', 0.02)
            if decision.position_size > max_position:
                validation_result['adjustments']['position_size'] = max_position
                validation_result['reasons'].append(f'Position size reduced from {decision.position_size:.4f} to {max_position:.4f}')
            
            # Check market conditions
            market_quality = market_data.get('data_quality', {}).get('overall_quality', 0)
            if market_quality < 0.5:
                validation_result['reasons'].append('Poor market data quality')
                validation_result['risk_level'] = 'high'
            
            # Check volatility
            volatility = market_data.get('technical_analysis', {}).get('volatility', 0)
            if volatility > 0.05:  # 5% volatility threshold
                validation_result['reasons'].append(f'High volatility detected: {volatility:.3f}')
                validation_result['adjustments']['reduce_leverage'] = True
                validation_result['risk_level'] = 'high'
            
            # Approve if no major issues
            if not any('below threshold' in reason or 'Poor market data' in reason for reason in validation_result['reasons']):
                validation_result['approved'] = True
                validation_result['risk_level'] = 'medium' if validation_result['reasons'] else 'low'
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Decision validation failed: {e}")
            validation_result['reasons'].append(f'Validation error: {e}')
            return validation_result
    
    async def _execute_trade(self, decision: TradingDecision) -> Dict[str, Any]:
        """Execute trade on LNMarkets"""
        try:
            if decision.action == "HOLD":
                return {
                    'status': 'no_action',
                    'reason': 'HOLD decision',
                    'timestamp': datetime.now()
                }
            
            # Execute through self-hosted multi-agent system
            return await self.multi_agent_system.execute_decision(decision)
                
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    async def _get_account_status(self) -> Dict[str, Any]:
        """Get current LNMarkets account status"""
        try:
            # Initialize client if needed
            if not self.trading_client.is_connected:
                self.trading_client.initialize()
            
            user_data = self.trading_client.get_user_info()
            positions = self.trading_client.get_positions()
            
            # Calculate total margin used in open positions
            total_margin_used = 0
            if isinstance(positions, list):
                for pos in positions:
                    total_margin_used += pos.get('margin', 0)
            
            total_balance = user_data.get('balance', 0)
            available_balance = total_balance - total_margin_used
            
            return {
                'balance': total_balance,
                'available_balance': available_balance,
                'margin_used': total_margin_used,
                'username': user_data.get('username', ''),
                'position_count': len(positions) if isinstance(positions, list) else 0,
                'network': 'testnet' if self.config.get('lnmarkets_testnet') else 'mainnet'
            }
        except Exception as e:
            self.logger.error(f"Account status fetch failed: {e}")
            return {'error': str(e)}
    
    def _summarize_market_data(self, market_data: Dict) -> Dict[str, Any]:
        """Create a summary of market data for logging"""
        try:
            summary = {
                'symbol': market_data.get('symbol', 'BTCUSD'),
                'current_price': market_data.get('market_data', {}).get('price', 0),
                'price_change_24h': market_data.get('market_data', {}).get('change_percent_24h', 0),
                'overall_sentiment': market_data.get('sentiment_analysis', {}).get('overall_sentiment', 0),
                'fear_greed_index': market_data.get('fundamentals', {}).get('fear_greed_index', 50),
                'news_count': len(market_data.get('news_data', [])),
                'technical_trend': market_data.get('technical_analysis', {}).get('trend', 'neutral'),
                'data_quality': market_data.get('data_quality', {}).get('overall_quality', 0)
            }
            return summary
        except Exception as e:
            self.logger.error(f"Market data summary failed: {e}")
            return {'error': str(e)}
    
    def _calculate_cycle_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for self-hosted system"""
        try:
            if len(self.performance_history) < 2:
                return {'cycles_completed': len(self.performance_history)}
            
            recent_cycles = self.performance_history[-10:]  # Last 10 cycles
            
            # Calculate success rates
            successful_cycles = sum(1 for cycle in recent_cycles 
                                  if cycle.get('execution_result', {}).get('status') == 'executed')
            approved_cycles = sum(1 for cycle in recent_cycles 
                                if cycle.get('validation_result', {}).get('approved', False))
            error_cycles = sum(1 for cycle in recent_cycles if 'error' in cycle)
            
            # Calculate average confidence
            confidences = []
            for cycle in recent_cycles:
                decision = cycle.get('trading_decision')
                if decision and 'confidence' in decision:
                    confidences.append(decision['confidence'])
            
            return {
                'cycles_completed': len(self.performance_history),
                'recent_cycles': len(recent_cycles),
                'execution_rate': successful_cycles / len(recent_cycles) if recent_cycles else 0,
                'approval_rate': approved_cycles / len(recent_cycles) if recent_cycles else 0,
                'error_rate': error_cycles / len(recent_cycles) if recent_cycles else 0,
                'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
                'system_type': 'self_hosted'
            }
            
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            return {'error': str(e)}
    
    async def run_continuous_monitoring(self, interval_minutes: int = None):
        """Run continuous self-hosted monitoring"""
        if interval_minutes is None:
            interval_minutes = self.config.get('update_interval_minutes', 3)
        
        self.logger.info(f"Starting continuous self-hosted monitoring (interval: {interval_minutes} minutes)")
        
        symbol = self.config.get('symbol', 'BTCUSD')
        
        while True:
            try:
                self.logger.info(f"Running self-hosted analysis cycle for {symbol}")
                
                result = await self.run_selfhosted_analysis_cycle(symbol)
                
                # Log key metrics
                if 'error' not in result:
                    decision = result.get('trading_decision')
                    if decision:
                        self.logger.info(
                            f"{symbol}: {decision.get('action', 'UNKNOWN')} "
                            f"(confidence: {decision.get('confidence', 0):.2f}) "
                            f"[Self-hosted System]"
                        )
                    
                    # Log account status
                    account = result.get('account_status', {})
                    if 'balance' in account:
                        self.logger.info(f"Account balance: {account['balance']} sats")
                
                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Continuous monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive self-hosted system status"""
        return {
            'system_type': 'self_hosted_lnmarkets',
            'components': {
                'multi_agent_system': self.multi_agent_system is not None,
                'data_provider': self.data_provider is not None,
                'trading_client': self.trading_client is not None,
                'cache_system': self.cache is not None
            },
            'config': {
                'symbol': self.config.get('symbol'),
                'trading_mode': self.config.get('trading_mode'),
                'min_confidence_threshold': self.config.get('min_confidence_threshold'),
                'max_position_size': self.config.get('max_position_size'),
                'max_leverage': self.config.get('max_leverage'),
                'auto_trading_enabled': self.config.get('enable_auto_trading'),
                'network': 'testnet' if self.config.get('lnmarkets_testnet') else 'mainnet'
            },
            'data_sources': {
                'lnmarkets': True,
                'coingecko': self.config.get('use_coingecko_data', True),
                'fear_greed_index': self.config.get('use_fear_greed_index', True),
                'crypto_news': self.config.get('use_crypto_news', True),
                'external_apis': False  # No paid APIs
            },
            'performance': self._calculate_cycle_metrics(),
            'timestamp': datetime.now()
        }
    
    def save_config(self, filename: str = "selfhosted_config.json"):
        """Save current configuration to file"""
        try:
            # Remove sensitive data before saving
            safe_config = self.config.copy()
            sensitive_keys = ['lnmarkets_api_key', 'lnmarkets_api_secret', 'lnmarkets_passphrase', 'deepseek_api_key']
            for key in sensitive_keys:
                if key in safe_config:
                    safe_config[key] = "***REDACTED***"
            
            with open(filename, 'w') as f:
                json.dump(safe_config, f, indent=2, default=str)
            
            self.logger.info(f"Configuration saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")


if __name__ == "__main__":
    # Example usage for self-hosted deployment
    async def main():
        # Initialize self-hosted orchestrator
        orchestrator = SelfHostedTradingOrchestrator()
        
        # Save default config template
        orchestrator.save_config("selfhosted_config_template.json")
        
        # Get system status
        status = orchestrator.get_system_status()
        print("Self-Hosted System Status:")
        print(json.dumps(status, indent=2, default=str))
        
        # Run single analysis cycle
        print("\nRunning self-hosted analysis cycle...")
        result = await orchestrator.run_selfhosted_analysis_cycle("BTCUSD")
        print(json.dumps(result, indent=2, default=str))
        
        # Uncomment to run continuous monitoring
        # await orchestrator.run_continuous_monitoring(interval_minutes=3)
    
    asyncio.run(main())
