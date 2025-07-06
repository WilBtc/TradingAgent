# 🔥 BITCOIN TRADING AGENT - QUICK COMMANDS 🔥
# Made by WilBTC

.PHONY: help install run stop status logs money

# Default - show the money-making commands
help:
	@echo "🔥 BITCOIN TRADING AGENT - MONEY MAKING COMMANDS 🔥"
	@echo ""
	@echo "💰 QUICK START:"
	@echo "  make install    Install the money printer"
	@echo "  make run        START MAKING MONEY 💸"
	@echo "  make status     Check if the AI is working"
	@echo "  make logs       Watch the profits roll in"
	@echo "  make stop       Stop the money printer (but why?)"
	@echo ""
	@echo "🚀 ADVANCED:"
	@echo "  make test       Test with paper trading first"
	@echo "  make config     Show current configuration"
	@echo "  make backup     Backup your profitable setup"
	@echo ""
	@echo "💎 Ready to stack sats? Run: make install && make run"

# Install the trading agent
install:
	@echo "🔥 Installing the Bitcoin Money Printer..."
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "✅ Installation complete!"
	@echo "🔥 Next: Copy .env.example to .env and add your API keys"
	@echo "🚀 Then run: make run"

# START MAKING MONEY
run:
	@echo "🚀 STARTING THE MONEY PRINTER..."
	@echo "🤖 AI agents initializing..."
	@echo "⚡ Scanning for profit opportunities..."
	python -m trading_bot.agents.profit_maximizer

# Test mode (paper trading)
test:
	@echo "🧪 Starting in TEST MODE (no real money)"
	@echo "📝 This is safe - no actual trades will be placed"
	LNMARKETS_TESTNET=true python -m trading_bot.agents.profit_maximizer

# Check if the money printer is working
status:
	@echo "💰 CHECKING MONEY PRINTER STATUS..."
	@ps aux | grep "profit_maximizer" | head -5 || echo "❌ Not running - use 'make run' to start"
	@echo ""
	@echo "📊 Recent activity:"
	@tail -10 logs/profit_maximizer.log 2>/dev/null || echo "No logs yet - start with 'make run'"

# Watch the money roll in
logs:
	@echo "💸 WATCHING THE PROFITS..."
	@echo "(Press Ctrl+C to stop watching)"
	@tail -f logs/profit_maximizer.log

# Stop the money printer (but why would you?)
stop:
	@echo "😢 Stopping the money printer..."
	@pkill -f "profit_maximizer" || echo "Already stopped"
	@echo "💔 Money printing stopped. Run 'make run' to restart."

# Show current configuration  
config:
	@echo "⚙️ CURRENT CONFIGURATION:"
	@echo "🎯 Mode: $${TRADING_MODE:-aggressive_profit}"
	@echo "💎 Max Leverage: $${MAX_LEVERAGE:-33}x"
	@echo "🎯 Daily Target: $${DAILY_PROFIT_TARGET:-5}%"
	@echo "🛡️ Stop Loss: $${STOP_LOSS_PERCENTAGE:-1.5}%"
	@echo "⚡ Scan Interval: $${UPDATE_INTERVAL_SECONDS:-10} seconds"

# Backup your profitable setup
backup:
	@echo "💾 Creating backup of your money printer..."
	@mkdir -p backups
	@tar -czf backups/trading-agent-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		config/ logs/ .env 2>/dev/null || echo "⚠️  Some files missing - that's normal for new installs"
	@echo "✅ Backup created in backups/"

# Show recent profits (if any)
money:
	@echo "💰 RECENT PROFIT REPORT:"
	@echo "========================"
	@grep -E "(PROFIT|profit|PnL)" logs/profit_maximizer.log | tail -10 2>/dev/null || echo "Start trading to see profits here!"
	@echo ""
	@echo "🚀 Not seeing profits yet? The AI needs time to find opportunities!"

# Emergency stop (for paper hands)
panic:
	@echo "🚨 EMERGENCY STOP ACTIVATED"
	@echo "😱 Stopping all trading immediately..."
	@pkill -f "python.*trading" || true
	@echo "🛑 All trading stopped."
	@echo "🧘 Take a breath. Remember: the AI doesn't panic, you shouldn't either."
	@echo "🚀 When ready, restart with: make run"