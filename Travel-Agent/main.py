import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, handoff
from agents.run import RunConfig, RunContextWrapper

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

# Setup Gemini model and client (OpenAI-compatible)
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# === Tools ===
def get_flights(destination: str) -> str:
    return f"✈️ Flights to {destination}:\n- AirX: $300\n- FlyJet: $280\n- SkyHigh: $320"

def suggest_hotels(destination: str) -> str:
    return f"🏨 Hotels in {destination}:\n- GrandView Hotel: 4⭐ ($120/night)\n- CozyStay Inn: 3⭐ ($80/night)"

# === Specialized Agents ===
DestinationAgent = Agent(
    name="DestinationAgent",
    instructions="Suggest travel destinations based on user's mood or interests. Ask follow-up questions if unclear.",
    model=model
)

BookingAgent = Agent(
    name="BookingAgent",
    instructions="Simulate booking flights and hotels using tools. Use get_flights() and suggest_hotels() when user wants to book.",
    model=model,
    tools={
        "get_flights": get_flights,
        "suggest_hotels": suggest_hotels
    }
)

ExploreAgent = Agent(
    name="ExploreAgent",
    instructions="Suggest local attractions, foods, and experiences in the selected destination.",
    model=model
)

# === Main Travel Agent ===
TravelAgent = Agent(
    name="TravelAgent",
    instructions="""You are a friendly AI travel designer that helps users plan trips. 

You have access to three specialized agents:
1. DestinationAgent - for suggesting travel destinations based on interests
2. BookingAgent - for booking flights and hotels 
3. ExploreAgent - for suggesting attractions, food, and experiences

Use the appropriate handoff tool when:
- User asks about destinations or where to go → use handoff_to_destination
- User asks about booking flights or hotels → use handoff_to_booking  
- User asks about attractions, food, or what to do at a destination → use handoff_to_explore

Always be helpful and guide users to the right specialist.""",
    model=model,
    handoffs=[
        handoff(DestinationAgent, tool_name_override="handoff_to_destination", tool_description_override="Handoff to DestinationAgent for destination suggestions"),
        handoff(BookingAgent, tool_name_override="handoff_to_booking", tool_description_override="Handoff to BookingAgent for flight and hotel bookings"),
        handoff(ExploreAgent, tool_name_override="handoff_to_explore", tool_description_override="Handoff to ExploreAgent for attractions and experiences"),
    ]
)

# === Chainlit Start ===
@cl.on_chat_start
async def start():
    cl.user_session.set("agent", TravelAgent)
    cl.user_session.set("config", config)
    cl.user_session.set("chat_history", [])
    await cl.Message(content="🌍 Welcome to the AI Travel Designer!\nTell me what you like, and I'll design a dream trip for you.").send()

# === Chainlit Message Handler ===
@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="💭 Thinking...")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))
    history = cl.user_session.get("chat_history") or []

    history.append({"role": "user", "content": message.content})

    try:
        result = Runner.run_sync(agent, history, run_config=config)
        final = result.final_output

        # Show handoff message if agent changed
        try:
            if hasattr(result, 'final_agent') and result.final_agent and result.final_agent != agent:
                agent_info = {
                    "DestinationAgent": {
                        "emoji": "🗺️",
                        "description": "I'll help you discover amazing destinations based on your interests and preferences!"
                    },
                    "BookingAgent": {
                        "emoji": "✈️",
                        "description": "I'll help you find the best flights and hotels for your trip!"
                    },
                    "ExploreAgent": {
                        "emoji": "🎯",
                        "description": "I'll suggest exciting attractions, delicious food, and unforgettable experiences!"
                    }
                }

                # ✅ Safe extraction of agent name
                agent_name = str(getattr(result.final_agent, "name", result.final_agent)).strip('"')
                info = agent_info.get(agent_name, {
                    "emoji": "🤖",
                    "description": "I'll help you with your request!"
                })

                await cl.Message(
                    content=f"{info['emoji']} **Switching to {agent_name}**\n\n{info['description']}",
                    author="System"
                ).send()
        except Exception as handoff_error:
            print(f"Handoff message error: {handoff_error}")

        # Final assistant response
        msg.content = final
        await msg.update()

        history.append({"role": "assistant", "content": final})
        cl.user_session.set("chat_history", history)

    except Exception as e:
        msg.content = f"❌ Error: {str(e)}"
        await msg.update()
        print(f"Error: {e}")
