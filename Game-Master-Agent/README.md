# Game Master Agent (Fantasy Adventure Game) 🧙⚔️

A multi-agent AI system that runs an immersive text-based fantasy adventure game using specialized agents and intelligent handoffs.

## What It Does

Run a text-based adventure game using multiple AI agents:
- **Narrates an adventure story** based on player choices
- **Uses Tools**: `roll_dice()` and `generate_event()` to control game flow
- **Hands off between**:
  - **NarratorAgent** (story progress)
  - **MonsterAgent** (combat phase)
  - **ItemAgent** (inventory & rewards)

## Features

- 🎮 **Immersive Fantasy Adventure**: Rich storytelling and interactive gameplay
- 🤖 **Multi-Agent Architecture**: Three specialized agents working together
- 🔄 **Intelligent Handoffs**: Seamless switching between agents based on gameplay
- 🛠️ **Custom Tools**: Dice roller and event generator for dynamic gameplay
- 💬 **Interactive Chat**: Chainlit-powered conversational interface
- 🎲 **Dice-Based Combat**: Random outcomes for exciting battles

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

### GameMasterAgent (Main)
- Orchestrates the entire adventure experience
- Routes players to appropriate specialized agents
- Creates immersive fantasy atmosphere

### NarratorAgent
- Narrates the adventure story and progression
- Provides vivid descriptions of locations and events
- Guides players through the story

### MonsterAgent
- Handles combat encounters and battles
- Uses `roll_dice()` tool for random outcomes
- Manages player actions (attack, defend, run)

### ItemAgent
- Manages inventory and item collection
- Uses `generate_event()` tool for random discoveries
- Distributes rewards after events or combat

## Gameplay

1. **Start Your Adventure**: Begin in a fantasy world with multiple locations
2. **Explore Locations**: Visit forests, dungeons, or villages
3. **Combat Encounters**: Fight monsters with dice-based mechanics
4. **Collect Rewards**: Find items and treasures throughout your journey
5. **Story Progression**: Experience an evolving narrative

## Tools

### roll_dice(sides: int = 20)
Returns a random number for combat outcomes and game mechanics.

### generate_event(context: str)
Generates random events based on the current location (forest, dungeon, village).

## Supported Locations

- **Forest**: Mysterious encounters, ancient trees, traveling merchants
- **Dungeon**: Traps, skeleton warriors, treasure chests
- **Village**: NPCs, blacksmiths, local rumors

## Project Structure

```
Game-Master-Agent/
├── main.py          # Main application with agent definitions
├── pyproject.toml   # Dependencies and project config
├── chainlit.md      # Welcome screen content
├── README.md        # This file
└── .env             # Environment variables (create this)
```

## Adventure Commands

- **Explore**: "I want to explore the forest"
- **Combat**: "I attack the monster"
- **Inventory**: "Check my inventory"
- **Story**: "What happens next?"

Embark on your epic fantasy adventure! 🗡️🛡️
