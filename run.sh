# RSA only (no API key needed)
python reference_game.py

# With Anthropic Claude
export ANTHROPIC_API_KEY=sk-...
python reference_game.py --llm anthropic

# With OpenAI
export OPENAI_API_KEY=sk-...
python reference_game.py --llm openai

# Tweak RSA parameters
python reference_game.py --alpha 8.0 --cost 0.2
