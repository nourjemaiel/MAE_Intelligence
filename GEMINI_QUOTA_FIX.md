# Gemini API 429 Quota Error - Solutions & Alternatives

## ✅ Quick Fix Applied

Your `agent.py` now has:
- **Automatic rate limiting** - Max 15 requests/minute (free tier limit)
- **Exponential backoff retry** - Auto-retries on 429 errors (wait 5s, 10s, 20s)
- **Better error messages** - Guides users on quota issues

### How to Test
```bash
cd mae_backend
python main.py  # or uvicorn main:app --reload
```

If you still hit quota limits, wait a few hours as free tier quotas reset periodically.

---

## 📊 Problem Analysis

Your free tier Gemini API has limits:
- **Requests per minute:** 15 (currently hit)
- **Requests per day:** Some limit 
- **Tokens per minute:** 32,000

The error suggests you're making too many API calls too quickly.

---

## 🛠️ Long-Term Solutions

### **Option 1: Reduce API Calls (Best for Free Tier)**

Add caching to avoid redundant queries:

```python
from functools import lru_cache
import hashlib

# In agent.py
class CachedAgent:
    def __init__(self, tools_map):
        self.cache = {}
        self.agent = MAEAgent(tools_map)
    
    def run(self, user_message, history=None):
        # Cache key = hash of message + last history item
        cache_key = hashlib.md5(
            f"{user_message}{history[-1] if history else ''}".encode()
        ).hexdigest()
        
        if cache_key in self.cache:
            logging.info(f"📦 Cache hit for: {user_message[:50]}")
            return self.cache[cache_key]
        
        result = self.agent.run(user_message, history)
        self.cache[cache_key] = result
        return result
```

### **Option 2: Switch to Claude API (Recommended)**

Replace Gemini with Claude (faster, better reasoning):

```python
# Install: pip install anthropic

import anthropic

class ClaudeAgent:
    def __init__(self, tools_map):
        self.tools_map = tools_map
        self.client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
    
    def run(self, user_message, history=None):
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",  # More calls per minute
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        return {"answer": response.content[0].text}
```

**Advantages:**
- Higher free tier limits (3,500 requests/day)
- Better reasoning for complex questions
- No quota issues for typical usage

### **Option 3: Upgrade to Paid Gemini Tier**

Visit: https://aistudio.google.com/app/apikey

- **Free:** 15 requests/min (current)
- **Paid:** Up to 10,000 requests/min
- **Cost:** ~$0.075 per 1M input tokens (very cheap)

### **Option 4: Reduce Tool Calls**

Modify agent prompt to make fewer tool calls:

```python
SYSTEM_PROMPT = """...
Règles :
- Utilise au MAXIMUM 2 outils par requête (au lieu de tous les outils)
- Combine les résultats pour répondre plutôt que faire plusieurs appels
- Cache les réponses communes...
"""
```

---

## 🔧 Configuration Changes

### Update `.env` to test alternatives:

```bash
# Current (Gemini free)
GEMINI_API_KEY=your_key
# CLAUDE_API_KEY=your_key  # Uncomment to use Claude

# Frontend
VITE_AGENT_URL=http://localhost:8000
```

### Update `requirements.txt` (if using Claude):

```txt
# Add this line
anthropic>=0.39.0
```

---

## 📈 Monitoring

Monitor API usage:

**Google AI Studio:**
https://aistudio.google.com/app/apikey

**Check logs:**
```bash
tail -f mae_backend/logs/agent.log | grep "429\|quota\|rate"
```

---

## 🎯 Recommended Action Plan

1. **Today:** Test the rate-limiting fix (already applied)
2. **This week:** Implement option 1 (caching) if Gemini keeps failing
3. **Next phase:** Try Claude (Option 2) for better performance
4. **Long-term:** Evaluate which API works best for your project

---

## 📞 Support

If you need help:
- Check MLflow dashboard: `http://localhost:5000`
- View agent logs: `mae_backend/logs/agent.log`
- Test API manually: `curl http://localhost:8000/docs`
