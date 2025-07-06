#!/usr/bin/env python3
"""
Profit Maximizer - Aggressive Auto-Trading System
Designed to stack sats and maximize profits on LNMarkets
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

from selfhosted_orchestrator import SelfHostedTradingOrchestrator
from selfhosted_multi_agent_system import SelfHostedMultiAgentSystem, AgentType
from lnmarkets_data_provider import LNMarketsDataProvider
from redis_cache import TradingCache


class ProfitMaximizer:
    """Aggressive auto-trading system focused on profit maximization"""
    
    def __init__(self, config_file: str = "auto_trading_config.json"):
        self.logger = logging.getLogger("profit_maximizer")
        self.cache = TradingCache()
        
        # Load aggressive trading configuration
        self.config = self._load_config(config_file)
        
        # Initialize core trading system
        self.orchestrator = SelfHostedTradingOrchestrator()
        
        # Override orchestrator config with profit-focused settings
        self.orchestrator.config.update(self.config)
        
        # Also update the multi-agent system config if it exists
        if hasattr(self.orchestrator, 'multi_agent_system') and self.orchestrator.multi_agent_system:
            self.orchestrator.multi_agent_system.config.update(self.config)
            # Update risk agent's max position size
            if hasattr(self.orchestrator.multi_agent_system, 'agents'):
                risk_agent = self.orchestrator.multi_agent_system.agents.get(AgentType.RISK_MANAGER)
                if risk_agent:
                    risk_agent.max_position_size = self.config.get('max_position_size', 0.003)
                    risk_agent.max_leverage = self.config.get('max_leverage', 33)
        
        # Performance tracking
        self.profit_history = []
        self.trade_count = 0
        self.daily_profit = 0.0
        self.start_balance = 0
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load profit-maximization configuration"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"✅ Loaded profit maximizer config from {config_file}")
                # Flatten nested config structures
                flattened_config = self._flatten_config(config)
                return flattened_config
            else:
                self.logger.warning(f"Config file {config_file} not found, using aggressive defaults")
                return self._get_aggressive_defaults()
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_aggressive_defaults()
    
    def _flatten_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested config structures for easier access"""
        flattened = {}
        
        # Copy top-level items
        for key, value in config.items():
            if isinstance(value, dict) and key in ['trading_parameters', 'profit_optimization', 'risk_management']:
                # Flatten these nested structures
                for nested_key, nested_value in value.items():
                    flattened[nested_key] = nested_value
            else:
                flattened[key] = value
        
        return flattened
    
    def _get_aggressive_defaults(self) -> Dict[str, Any]:
        """Get aggressive profit-focused default configuration"""
        return {
            'auto_trading_enabled': True,
            'trading_mode': 'aggressive_profit',
            'min_confidence_threshold': 0.45,
            'max_position_size': 0.003,  # 0.3% for smaller USD amounts
            'max_leverage': 33,
            'update_interval_minutes': 2,
            'target_daily_profit': 0.05,
            'stop_loss_percentage': 0.015,
            'take_profit_percentage': 0.025,
            'quick_scalp_mode': True,
            'sentiment_boost': True
        }
    
    async def start_profit_engine(self):
        """Start the aggressive profit maximization engine"""
        self.logger.info("🚀 STARTING PROFIT MAXIMIZATION ENGINE 🚀")
        self.logger.info("💰 MISSION: STACK SATS AND MAXIMIZE PROFITS 💰")
        
        # Get initial balance
        account_status = await self.orchestrator._get_account_status()
        self.start_balance = account_status.get('balance', 0)
        self.start_available_balance = account_status.get('available_balance', self.start_balance)
        
        # Initialize technical analyzer for position monitoring
        from real_technical_indicators import RealTechnicalAnalyzer
        self.technical_analyzer = RealTechnicalAnalyzer()
        
        self.logger.info(f"💳 Starting Balance: {self.start_balance:,} sats")
        self.logger.info(f"🎯 Daily Profit Target: {self.config.get('target_daily_profit', 0.05)*100:.1f}%")
        
        # Display interval in appropriate units
        interval_minutes = self.config.get('update_interval_minutes', 2)
        if interval_minutes < 1:
            interval_seconds = int(interval_minutes * 60)
            self.logger.info(f"⚡ Update Interval: {interval_seconds} seconds")
        else:
            self.logger.info(f"⚡ Update Interval: {interval_minutes} minutes")
        
        # Start continuous profit generation
        await self._run_profit_loop()
    
    async def _run_profit_loop(self):
        """Main profit generation loop"""
        update_interval = self.config.get('update_interval_minutes', 2)
        
        while True:
            try:
                loop_start = datetime.now()
                
                self.logger.info("⚡ SCANNING FOR PROFIT OPPORTUNITIES ⚡")
                
                # PRIORITY 1: Check and manage existing positions for profit-taking
                await self._monitor_and_close_profitable_positions()
                
                # PRIORITY 2: Run enhanced analysis cycle for new positions
                result = await self.orchestrator.run_selfhosted_analysis_cycle("BTCUSD")
                
                # Process the trading decision
                await self._process_trading_result(result)
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Log current status
                await self._log_status_update()
                
                # Wait for next cycle
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, (update_interval * 60) - loop_duration)
                
                if sleep_time > 0:
                    # Display wait time appropriately
                    if sleep_time < 60:
                        self.logger.info(f"💤 Waiting {sleep_time:.1f}s for next profit opportunity...")
                    else:
                        self.logger.info(f"💤 Waiting {sleep_time/60:.1f}m for next profit opportunity...")
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"❌ Profit loop error: {e}")
                await asyncio.sleep(30)  # Brief pause before retry
    
    async def _process_trading_result(self, result: Dict[str, Any]):
        """Process trading result and execute if profitable"""
        try:
            trading_decision = result.get('trading_decision')
            validation_result = result.get('validation_result', {})
            
            if not trading_decision:
                self.logger.warning("⚠️ No trading decision generated")
                return
            
            action = trading_decision.get('action', 'HOLD')
            confidence = trading_decision.get('confidence', 0)
            
            # Enhanced profit-focused decision logic
            if action != 'HOLD' and validation_result.get('approved', False):
                # Apply profit maximization enhancements
                enhanced_decision = await self._enhance_for_profit(trading_decision, result)
                
                if enhanced_decision:
                    # Execute the enhanced trade
                    execution_result = await self._execute_profit_trade(enhanced_decision)
                    
                    if execution_result.get('status') == 'executed':
                        self.trade_count += 1
                        self.logger.info(f"✅ TRADE #{self.trade_count} EXECUTED: {action} (confidence: {confidence:.1%})")
                        self.logger.info(f"💰 Target Profit: {enhanced_decision.get('target_profit_sats', 0):,} sats")
                    else:
                        self.logger.warning(f"⚠️ Trade execution failed: {execution_result.get('error', 'Unknown')}")
                else:
                    self.logger.info(f"📊 Trading opportunity found: {action} @ {confidence:.1%}")
                    # Execute original decision without enhancement
                    execution_result = await self._execute_profit_trade(trading_decision)
                    if execution_result.get('status') == 'executed':
                        self.trade_count += 1
                        self.logger.info(f"✅ TRADE #{self.trade_count} EXECUTED: {action} (confidence: {confidence:.1%})")
                    else:
                        self.logger.warning(f"⚠️ Trade execution failed: {execution_result.get('error', 'Unknown')}")
            else:
                self.logger.info(f"📊 Market analysis: {action} @ {confidence:.1%} - Waiting for better opportunity")
                
        except Exception as e:
            self.logger.error(f"Error processing trading result: {e}")
    
    async def _enhance_for_profit(self, decision: Dict[str, Any], market_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhance trading decision for maximum profit"""
        try:
            # Get market data
            market_data = market_result.get('market_data_summary', {})
            current_price = market_data.get('current_price', 0)
            sentiment = market_data.get('overall_sentiment', 0)
            
            # Calculate enhanced position size based on confidence and sentiment
            base_size = decision.get('position_size', 0.01)
            confidence = decision.get('confidence', 0.5)
            
            # Sentiment boost
            if self.config.get('sentiment_boost') and abs(sentiment) > 0.3:
                sentiment_multiplier = 1 + (abs(sentiment) * 0.5)
                enhanced_size = base_size * sentiment_multiplier
                self.logger.info(f"💡 Sentiment boost applied: {sentiment_multiplier:.2f}x")
            else:
                enhanced_size = base_size
            
            # Confidence scaling
            if confidence > 0.7:
                confidence_multiplier = 1 + ((confidence - 0.7) * 2)
                enhanced_size *= confidence_multiplier
                self.logger.info(f"🔥 High confidence boost: {confidence_multiplier:.2f}x")
            
            # Cap at maximum position size
            max_size = self.config.get('max_position_size', 0.003)  # Use smaller default
            enhanced_size = min(enhanced_size, max_size)
            
            # Calculate profit targets
            take_profit_pct = self.config.get('take_profit_percentage', 0.025)
            stop_loss_pct = self.config.get('stop_loss_percentage', 0.015)
            
            if decision['action'] == 'BUY':
                take_profit_price = current_price * (1 + take_profit_pct)
                stop_loss_price = current_price * (1 - stop_loss_pct)
            else:  # SELL
                take_profit_price = current_price * (1 - take_profit_pct)
                stop_loss_price = current_price * (1 + stop_loss_pct)
            
            # Get current available balance
            account_status = await self.orchestrator._get_account_status()
            available_balance = account_status.get('available_balance', self.start_balance)
            
            # Calculate expected profit in sats using available balance
            position_value_sats = enhanced_size * available_balance
            target_profit_sats = position_value_sats * take_profit_pct
            
            # Trade every opportunity - no minimum profit threshold
            # Removed profit filtering to trade all signals as requested
            
            enhanced_decision = decision.copy()
            enhanced_decision.update({
                'position_size': enhanced_size,
                'take_profit': take_profit_price,
                'stop_loss': stop_loss_price,
                'target_profit_sats': target_profit_sats,
                'enhanced_for_profit': True,
                'enhancement_reason': f"Sentiment: {sentiment:.2f}, Confidence: {confidence:.1%}"
            })
            
            return enhanced_decision
            
        except Exception as e:
            self.logger.error(f"Error enhancing decision for profit: {e}")
            return None
    
    async def _execute_profit_trade(self, enhanced_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade with profit maximization parameters"""
        try:
            # Use the orchestrator's multi-agent system for execution
            from selfhosted_multi_agent_system import TradingDecision
            
            # Convert dict back to TradingDecision object
            trading_decision = TradingDecision(
                symbol=enhanced_decision['symbol'],
                action=enhanced_decision['action'],
                confidence=enhanced_decision['confidence'],
                position_size=enhanced_decision['position_size'],
                stop_loss=enhanced_decision.get('stop_loss'),
                take_profit=enhanced_decision.get('take_profit'),
                reasoning=enhanced_decision['reasoning'] + " [PROFIT ENHANCED]",
                risk_assessment=enhanced_decision.get('risk_assessment', {}),
                agent_consensus=enhanced_decision.get('agent_consensus', {}),
                timestamp=datetime.now()
            )
            
            # Execute through the multi-agent system
            result = await self.orchestrator.multi_agent_system.execute_decision(trading_decision)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Profit trade execution failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _update_performance_metrics(self):
        """Update profit tracking metrics"""
        try:
            # Get current account status
            account_status = await self.orchestrator._get_account_status()
            current_balance = account_status.get('balance', 0)
            
            # Calculate profit
            if self.start_balance > 0:
                total_profit = current_balance - self.start_balance
                profit_percentage = (total_profit / self.start_balance) * 100
                
                self.daily_profit = profit_percentage
                
                # Store profit history
                profit_entry = {
                    'timestamp': datetime.now(),
                    'balance': current_balance,
                    'profit_sats': total_profit,
                    'profit_percentage': profit_percentage,
                    'trade_count': self.trade_count
                }
                
                self.profit_history.append(profit_entry)
                
                # Keep only last 100 entries
                if len(self.profit_history) > 100:
                    self.profit_history = self.profit_history[-100:]
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _log_status_update(self):
        """Log current profit status"""
        try:
            account_status = await self.orchestrator._get_account_status()
            current_balance = account_status.get('balance', 0)
            available_balance = account_status.get('available_balance', current_balance)
            margin_used = account_status.get('margin_used', 0)
            
            if self.start_balance > 0:
                total_profit = current_balance - self.start_balance
                profit_pct = (total_profit / self.start_balance) * 100
                
                self.logger.info("📊 === PROFIT STATUS UPDATE ===")
                self.logger.info(f"💰 Current Balance: {current_balance:,} sats")
                self.logger.info(f"💵 Available Balance: {available_balance:,} sats")
                self.logger.info(f"🔒 Margin Used: {margin_used:,} sats")
                self.logger.info(f"📈 Total Profit: {total_profit:,} sats ({profit_pct:+.2f}%)")
                self.logger.info(f"🔥 Trades Executed: {self.trade_count}")
                self.logger.info(f"🎯 Daily Target: {self.config.get('target_daily_profit', 0.05)*100:.1f}%")
                
                if profit_pct > 0:
                    self.logger.info(f"🚀 PROFIT ACHIEVED! STACKING SATS! 🚀")
                
                # Check if daily target reached
                daily_target = self.config.get('target_daily_profit', 0.05) * 100
                if profit_pct >= daily_target:
                    self.logger.info(f"🎉 DAILY TARGET REACHED! {profit_pct:.2f}% >= {daily_target:.1f}% 🎉")
                
        except Exception as e:
            self.logger.error(f"Error logging status: {e}")
    
    def get_profit_summary(self) -> Dict[str, Any]:
        """Get comprehensive profit summary"""
        try:
            if not self.profit_history:
                return {'status': 'no_data'}
            
            latest = self.profit_history[-1]
            
            return {
                'system_type': 'profit_maximizer',
                'status': 'active',
                'start_balance': self.start_balance,
                'current_balance': latest['balance'],
                'total_profit_sats': latest['profit_sats'],
                'total_profit_percentage': latest['profit_percentage'],
                'trades_executed': self.trade_count,
                'daily_target': self.config.get('target_daily_profit', 0.05) * 100,
                'target_achieved': latest['profit_percentage'] >= (self.config.get('target_daily_profit', 0.05) * 100),
                'avg_profit_per_trade': latest['profit_sats'] / max(1, self.trade_count),
                'config': {
                    'aggressive_mode': True,
                    'auto_trading': True,
                    'max_leverage': self.config.get('max_leverage', 10),
                    'update_interval': self.config.get('update_interval_minutes', 2)
                },
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating profit summary: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _monitor_and_close_profitable_positions(self):
        """Monitor existing positions and close them when profitable or when reversal detected"""
        try:
            # Get current positions from LNMarkets
            positions = await self._get_open_positions()
            
            if not positions:
                self.logger.info("📊 No open positions to monitor")
                return
            
            self.logger.info(f"📊 Monitoring {len(positions)} open position(s) for profit-taking")
            
            # Get current market data and technical indicators
            market_data = await self._get_current_market_data()
            current_price = market_data.get('price', 0)
            
            if current_price == 0:
                self.logger.warning("⚠️ Unable to get current price for position monitoring")
                return
            
            # Analyze each position
            for position in positions:
                await self._analyze_and_manage_position(position, current_price, market_data)
                
        except Exception as e:
            self.logger.error(f"❌ Error monitoring positions: {e}")
    
    async def _get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions from LNMarkets"""
        try:
            # Use the orchestrator's LNMarkets client
            if hasattr(self.orchestrator, 'lnmarkets_client') and self.orchestrator.lnmarkets_client:
                positions = self.orchestrator.lnmarkets_client.get_positions()
                return positions
            
            # Fallback: Try to get from the multi-agent system
            if hasattr(self.orchestrator, 'multi_agent_system') and hasattr(self.orchestrator.multi_agent_system, 'lnmarkets_client'):
                positions = self.orchestrator.multi_agent_system.lnmarkets_client.get_positions()
                return positions
            
            # Initialize our own client if needed
            from lnmarkets_official_client import get_lnmarkets_official_client
            client = get_lnmarkets_official_client()
            if client.initialize():
                positions = client.get_positions()
                return positions
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    async def _get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data and technical indicators"""
        try:
            # Try to get from the orchestrator's data provider
            if hasattr(self.orchestrator, 'data_provider'):
                ticker = await self.orchestrator.data_provider.get_ticker("BTCUSD")
                if ticker:
                    return {
                        'price': ticker.get('last', 0),
                        'bid': ticker.get('bid', 0),
                        'ask': ticker.get('ask', 0),
                        'timestamp': datetime.now()
                    }
            
            # Fallback: Use technical analyzer for market data
            if hasattr(self, 'technical_analyzer'):
                analyzer = self.technical_analyzer
            else:
                from real_technical_indicators import RealTechnicalAnalyzer
                analyzer = RealTechnicalAnalyzer()
            df = await analyzer.get_market_data("BTC/USDT", "1m", limit=50)
            
            if not df.empty:
                indicators = analyzer.calculate_all_indicators(df)
                return {
                    'price': df['close'].iloc[-1],
                    'indicators': indicators,
                    'dataframe': df,
                    'timestamp': datetime.now()
                }
            
            return {'price': 0}
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {'price': 0}
    
    async def _analyze_and_manage_position(self, position: Dict[str, Any], current_price: float, market_data: Dict[str, Any]):
        """Analyze a single position and decide whether to close it for profit"""
        try:
            position_id = position.get('id')
            side = position.get('side', '').lower()
            entry_price = position.get('entry_price', 0)
            quantity = position.get('quantity', 0)
            pnl = position.get('pnl', 0)
            
            # Calculate profit percentage
            if side == 'buy':
                profit_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # sell/short
                profit_pct = ((entry_price - current_price) / entry_price) * 100
            
            self.logger.info(f"📊 Position {position_id}: {side.upper()} @ {entry_price:,.2f}, Current: {current_price:,.2f}, PnL: {pnl:,} sats ({profit_pct:+.2f}%)")
            
            # Decision logic for closing positions
            should_close = False
            close_reason = ""
            
            # 1. Check if position reached profit target
            target_profit_pct = self.config.get('take_profit_percentage', 0.025) * 100  # Convert to percentage
            if profit_pct >= target_profit_pct:
                should_close = True
                close_reason = f"Target profit reached: {profit_pct:.2f}% >= {target_profit_pct:.2f}%"
            
            # 2. Quick scalp exit for fast profits
            elif self.config.get('quick_scalp_mode', True) and profit_pct >= 1.0:
                # Quick exit on 1%+ profit in scalp mode
                should_close = True
                close_reason = f"Quick scalp profit taken: {profit_pct:.2f}%"
            
            # 3. Check for smaller profits with reversal signals
            elif profit_pct > 0.5:  # If in profit (even small)
                reversal_detected = await self._check_for_reversal(side, market_data)
                if reversal_detected:
                    should_close = True
                    close_reason = f"Reversal detected with profit: {profit_pct:.2f}%"
            
            # 4. Dynamic profit taking based on market conditions
            elif profit_pct > 0:
                # Get technical indicators if available
                indicators = market_data.get('indicators')
                if indicators:
                    # Check momentum weakening
                    if side == 'buy' and indicators.momentum_score < -0.2:
                        should_close = True
                        close_reason = f"Momentum weakening with profit: {profit_pct:.2f}%"
                    elif side == 'sell' and indicators.momentum_score > 0.2:
                        should_close = True
                        close_reason = f"Momentum reversing with profit: {profit_pct:.2f}%"
                    
                    # Check RSI extremes
                    if side == 'buy' and indicators.rsi > 75:
                        should_close = True
                        close_reason = f"RSI overbought ({indicators.rsi:.1f}) with profit: {profit_pct:.2f}%"
                    elif side == 'sell' and indicators.rsi < 25:
                        should_close = True
                        close_reason = f"RSI oversold ({indicators.rsi:.1f}) with profit: {profit_pct:.2f}%"
            
            # Execute close if decided
            if should_close:
                self.logger.info(f"🎯 CLOSING POSITION FOR PROFIT: {close_reason}")
                await self._close_position(position_id, close_reason)
            else:
                # Log why we're keeping the position open
                if profit_pct > 0:
                    self.logger.info(f"✅ Keeping position open: {profit_pct:.2f}% profit, waiting for better exit")
                else:
                    self.logger.info(f"⏳ Position at {profit_pct:.2f}% loss, holding for recovery")
                    
        except Exception as e:
            self.logger.error(f"Error analyzing position: {e}")
    
    async def _check_for_reversal(self, position_side: str, market_data: Dict[str, Any]) -> bool:
        """Check for potential market reversal using technical indicators"""
        try:
            indicators = market_data.get('indicators')
            df = market_data.get('dataframe')
            
            if not indicators:
                return False
            
            reversal_signals = 0
            
            # Check MACD crossover
            if position_side == 'buy':
                # For long positions, check for bearish crossover
                if indicators.macd < indicators.macd_signal and indicators.macd < 0:
                    reversal_signals += 1
                    self.logger.info("⚠️ Bearish MACD crossover detected")
            else:
                # For short positions, check for bullish crossover
                if indicators.macd > indicators.macd_signal and indicators.macd > 0:
                    reversal_signals += 1
                    self.logger.info("⚠️ Bullish MACD crossover detected")
            
            # Check momentum shift
            if position_side == 'buy' and indicators.momentum_score < -0.3:
                reversal_signals += 1
                self.logger.info("⚠️ Strong negative momentum detected")
            elif position_side == 'sell' and indicators.momentum_score > 0.3:
                reversal_signals += 1
                self.logger.info("⚠️ Strong positive momentum detected")
            
            # Check price vs moving averages
            if df is not None and len(df) > 20:
                current_price = df['close'].iloc[-1]
                sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
                
                if position_side == 'buy' and current_price < sma_20:
                    reversal_signals += 1
                    self.logger.info("⚠️ Price broke below SMA20")
                elif position_side == 'sell' and current_price > sma_20:
                    reversal_signals += 1
                    self.logger.info("⚠️ Price broke above SMA20")
            
            # Check Bollinger Bands
            if position_side == 'buy' and market_data.get('price', 0) >= indicators.bb_upper:
                reversal_signals += 1
                self.logger.info("⚠️ Price at upper Bollinger Band")
            elif position_side == 'sell' and market_data.get('price', 0) <= indicators.bb_lower:
                reversal_signals += 1
                self.logger.info("⚠️ Price at lower Bollinger Band")
            
            # Need at least 2 reversal signals to confirm
            return reversal_signals >= 2
            
        except Exception as e:
            self.logger.error(f"Error checking for reversal: {e}")
            return False
    
    async def _close_position(self, position_id: str, reason: str):
        """Close a position on LNMarkets"""
        try:
            # Try to close through the orchestrator
            if hasattr(self.orchestrator, 'lnmarkets_client') and self.orchestrator.lnmarkets_client:
                result = self.orchestrator.lnmarkets_client.close_position(position_id)
                if result:
                    self.logger.info(f"✅ Position {position_id} closed successfully: {reason}")
                    self.logger.info(f"💰 Closed position result: {result}")
                    return True
            
            # Fallback: Use our own client
            from lnmarkets_official_client import get_lnmarkets_official_client
            client = get_lnmarkets_official_client()
            if client.initialize():
                result = client.close_position(position_id)
                if result:
                    self.logger.info(f"✅ Position {position_id} closed successfully: {reason}")
                    self.logger.info(f"💰 Closed position result: {result}")
                    return True
            
            self.logger.error(f"❌ Failed to close position {position_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {e}")
            return False


async def main():
    """Main function to start profit maximization"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        profit_maximizer = ProfitMaximizer()
        
        # Show initial status
        summary = profit_maximizer.get_profit_summary()
        print("\n🚀 PROFIT MAXIMIZER INITIALIZED 🚀")
        print(json.dumps(summary, indent=2, default=str))
        
        # Start the profit engine
        await profit_maximizer.start_profit_engine()
        
    except KeyboardInterrupt:
        print("\n⚠️ Profit maximizer stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
