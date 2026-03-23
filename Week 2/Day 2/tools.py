import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun

# --- 1. Currency Exchange Tool ---
class CurrencyInput(BaseModel):
    amount: float = Field(description="The amount of money to convert")
    from_currency: str = Field(description="3-letter currency code to convert from (e.g., USD, EUR, INR)")
    to_currency: str = Field(description="3-letter currency code to convert to (e.g., USD, EUR, INR)")

@tool("currency_converter", args_schema=CurrencyInput)
async def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts a specific amount from one currency to another using current exchange rates."""
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    
    url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency}&to={to_currency}"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            converted_amount = data["rates"][to_currency]
            return f"{amount} {from_currency} is currently equal to {converted_amount} {to_currency}."
    except httpx.HTTPStatusError as e:
        return f"Tool Error: The currency API returned an error. (Status: {e.response.status_code}). Please inform the user gracefully."
    except Exception as e:
        return f"Tool Error: Unable to fetch currency data at this moment ({str(e)}). Please apologize to the user."

# --- 2. Weather Tool ---
class WeatherInput(BaseModel):
    city: str = Field(description="The name of the city to get the weather for (e.g., Tokyo, San Francisco, London)")

@tool("get_weather", args_schema=WeatherInput)
async def get_weather(city: str) -> str:
    """Fetches the current weather conditions for a specified city."""
    # Using wttr.in for a simple, free JSON weather API that accepts city names directly
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Extracting basic info to prevent token bloat
            current = data['current_condition'][0]
            temp = current['temp_C']
            desc = current['weatherDesc'][0]['value']
            feels_like = current['FeelsLikeC']
            
            return f"The current weather in {city} is {desc} with a temperature of {temp}°C (feels like {feels_like}°C)."
    except Exception as e:
        return f"Tool Error: Unable to fetch weather data for {city}. It might not be a valid city name or the API is down. Inform the user."

# --- 3. Web Search Tool ---
# DuckDuckGo is already wrapped beautifully by LangChain, but we will wrap it to ensure it catches errors.
class SearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the internet.")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str) -> str:
    """Searches the internet for current events, facts, or general knowledge."""
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result
    except Exception as e:
        return f"Tool Error: Web search failed ({str(e)}). Ask the user to try again later."

# --- List of tools to pass to our agent ---
tools = [currency_converter, get_weather, web_search]