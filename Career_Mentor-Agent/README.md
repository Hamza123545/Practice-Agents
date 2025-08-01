# Career Mentor Agent 🎓

A multi-agent AI system that guides students through career exploration using specialized agents and intelligent handoffs.

## What It Does

Guide students through career exploration using multi-agent support:
- **Recommends career paths** based on interests
- **Uses Tool**: `get_career_roadmap()` to show skills needed for a chosen field
- **Hands off between agents**:
  - **CareerAgent** (suggests fields)
  - **SkillAgent** (shows skill-building plans)
  - **JobAgent** (shares real-world job roles)

## Features

- 🤖 **Multi-Agent Architecture**: Three specialized agents working together
- 🔄 **Intelligent Handoffs**: Seamless switching between agents based on user needs
- 🛠️ **Custom Tools**: Career roadmap generator for different fields
- 💬 **Interactive Chat**: Chainlit-powered conversational interface
- 🎯 **Career Guidance**: Step-by-step skill development plans

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

### CareerAgent (Main)
- Coordinates conversations and user experience
- Routes queries to appropriate specialized agents
- Provides general career guidance

### SkillAgent
- Generates detailed skill roadmaps
- Uses `get_career_roadmap()` tool
- Covers software, data science, medicine, and more

### JobAgent
- Suggests specific job roles and titles
- Provides salary information and requirements
- Shares real-world career preparation advice

## Usage

1. Start a conversation with the Career Mentor
2. Share your career interests or goals
3. The system will automatically route you to the best agent
4. Get personalized guidance and roadmaps

## Supported Career Fields

- **Software Engineering**: Programming, development, tech roles
- **Data Science**: Analytics, machine learning, statistics
- **Medicine**: Healthcare, medical school, clinical practice
- **Custom Fields**: General guidance for any career path

## Project Structure

```
Career_Mentor-Agent/
├── main.py          # Main application with agent definitions
├── pyproject.toml   # Dependencies and project config
├── chainlit.md      # Welcome screen content
├── README.md        # This file
└── .env             # Environment variables (create this)
```
