# AI Travel Designer Agent 🌍

A multi-agent AI system that plans complete travel experiences by coordinating between specialized agents with intelligent handoffs.

## What It Does

Plan a full travel experience by coordinating between specialized agents:
- **Suggests destinations** based on mood or interests
- **Uses Tools**: `get_flights()`, `suggest_hotels()` with mock data
- **Hands off between**:
  - **DestinationAgent** (finds places)
  - **BookingAgent** (simulates travel booking)
  - **ExploreAgent** (suggests attractions & food)

## Features

- 🤖 **Multi-Agent Architecture**: Three specialized agents working together
- 🔄 **Intelligent Handoffs**: Seamless switching between agents based on user needs
- 🛠️ **Custom Tools**: Travel info generator and hotel picker
- 💬 **Interactive Chat**: Chainlit-powered conversational interface
- ✈️ **Complete Travel Planning**: From destination selection to booking and exploration

## Technology Stack

- **OpenAI Agent SDK + Runner**: Core agent framework
- **Chainlit**: Chat interface and session management
- **Gemini API**: AI model backend
- **Python 3.13+**: Modern Python features

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Environment setup**:
   Create a `.env` file with:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Run the application**:
   ```bash
   chainlit run main.py
   ```

## Agent Roles

### TravelAgent (Main)
- Coordinates conversations and user experience
- Routes queries to appropriate specialized agents
- Provides general travel planning guidance

### DestinationAgent
- Suggests travel destinations based on user preferences
- Asks follow-up questions to understand interests
- Recommends places based on mood and interests

### BookingAgent
- Simulates flight and hotel bookings
- Uses `get_flights()` and `suggest_hotels()` tools
- Provides pricing and availability information

### ExploreAgent
- Suggests local attractions and experiences
- Recommends restaurants and food options
- Provides cultural and entertainment insights

## Usage

1. Start a conversation with the Travel Designer
2. Share your travel preferences or interests
3. The system will automatically route you to the best agent
4. Get personalized travel recommendations and booking options

## Supported Features

- **Destination Discovery**: Find perfect travel destinations
- **Flight Booking**: Mock flight options with pricing
- **Hotel Selection**: Accommodation recommendations
- **Local Exploration**: Attractions, food, and experiences
- **Complete Trip Planning**: End-to-end travel assistance

## Project Structure

```
Travel-Agent/
├── main.py          # Main application with agent definitions
├── pyproject.toml   # Dependencies and project config
├── chainlit.md      # Welcome screen content
├── README.md        # This file
└── .env             # Environment variables (create this)
```

## Tools

### get_flights(destination: str)
Returns mock flight options for a given destination with pricing.

### suggest_hotels(destination: str)
Returns mock hotel recommendations for a given destination with ratings and pricing.
