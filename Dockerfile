# Multi-stage: supports both Python and Node.js bots
# Uncomment the section for your language

# --- Python Bot ---
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bot.py .
COPY CLAUDE.md .
EXPOSE 5001
CMD ["uvicorn", "bot:app", "--host", "0.0.0.0", "--port", "5001"]

# --- Node.js Bot (uncomment below, comment above) ---
# FROM node:20-slim
# WORKDIR /app
# COPY package*.json .
# RUN npm install --production
# COPY bot.js .
# COPY CLAUDE.md .
# EXPOSE 5001
# CMD ["node", "bot.js"]
