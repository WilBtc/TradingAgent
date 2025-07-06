#!/bin/bash

# 🔥 BITCOIN TRADING AGENT - GITHUB PUSH SCRIPT 🔥
# Made by WilBTC

echo "🔥 PUSHING THE BITCOIN MONEY PRINTER TO GITHUB 🔥"
echo "================================================"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please run from the marketing repo directory."
    exit 1
fi

# Check if we have changes to push
if git diff --quiet && git diff --cached --quiet; then
    echo "✅ Everything is already pushed!"
    exit 0
fi

echo "📦 Repository contents ready to push:"
echo "✅ Epic Sales Pitch README"
echo "✅ Quick Start Guide"
echo "✅ Profit-Focused Features"
echo "✅ Money-Making Commands"
echo "✅ Complete Trading System"
echo "✅ Docker Deployment Ready"
echo ""

echo "🔑 Authentication Options:"
echo "1. SSH (recommended)"
echo "2. HTTPS with Personal Access Token"
echo "3. GitHub CLI"
echo ""

read -p "Choose authentication method (1-3): " auth_method

case $auth_method in
    1)
        echo "🔐 Using SSH authentication..."
        git remote remove origin 2>/dev/null || true
        git remote add origin git@github.com:WilBtc/TradingAgent.git
        ;;
    2)
        echo "🔐 Using HTTPS with Personal Access Token..."
        echo "ℹ️  You'll need to enter your GitHub username and Personal Access Token"
        git remote remove origin 2>/dev/null || true
        git remote add origin https://github.com/WilBtc/TradingAgent.git
        ;;
    3)
        echo "🔐 Using GitHub CLI..."
        if ! command -v gh &> /dev/null; then
            echo "❌ GitHub CLI not installed. Please install it or choose another method."
            exit 1
        fi
        gh auth status || gh auth login
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "🚀 Pushing to GitHub..."
echo "Repository: https://github.com/WilBtc/TradingAgent"
echo ""

# Push to GitHub
if git push -u origin main; then
    echo ""
    echo "🎉 SUCCESS! BITCOIN TRADING AGENT DEPLOYED TO GITHUB! 🎉"
    echo "=================================================="
    echo ""
    echo "🔗 Your killer marketing repository is live at:"
    echo "   https://github.com/WilBtc/TradingAgent"
    echo ""
    echo "💰 What's live on GitHub:"
    echo "   ✅ Late-night sales pitch README"
    echo "   ✅ One-command installation"
    echo "   ✅ Professional source code"
    echo "   ✅ Docker deployment ready"
    echo "   ✅ Conversion-optimized documentation"
    echo ""
    echo "🎯 Your repository is now a CONVERSION MACHINE!"
    echo "   - Grabs attention with epic headlines"
    echo "   - Uses urgency and social proof"
    echo "   - Makes setup stupidly easy"
    echo "   - Positions as 'the AI that actually works'"
    echo ""
    echo "📈 Next steps to maximize conversions:"
    echo "   1. Add repository description on GitHub"
    echo "   2. Create releases with version tags"
    echo "   3. Add social media links to profile"
    echo "   4. Consider creating demo videos"
    echo ""
    echo "⚡ Ready to convert GitHub visitors into customers! ⚡"
    echo "Made by WilBTC - Stack Sats, Maximize Profits"
else
    echo ""
    echo "❌ PUSH FAILED"
    echo "=============="
    echo ""
    echo "Common solutions:"
    echo "1. Check your GitHub credentials"
    echo "2. Verify the repository exists: https://github.com/WilBtc/TradingAgent"
    echo "3. Try a different authentication method"
    echo "4. Check your internet connection"
    echo ""
    echo "Need help? The repository is ready - just need to get it uploaded!"
fi