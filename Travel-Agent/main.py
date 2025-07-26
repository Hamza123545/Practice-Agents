import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

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

def get_flights(destination: str) -> str:
    return f"✈️ Flights to {destination}:\n- AirX: $300\n- FlyJet: $280\n- SkyHigh: $320"

def suggest_hotels(destination: str) -> str:
    return f"🏨 Hotels in {destination}:\n- GrandView Hotel: 4⭐ ($120/night)\n- CozyStay Inn: 3⭐ ($80/night)"

DestinationAgent = Agent(
    name="DestinationAgent",
    instructions="Suggest travel destinations based on user's mood or interests. Ask follow-up questions if unclear."
)

BookingAgent = Agent(
    name="BookingAgent",
    instructions="Simulate booking flights and hotels using tools.",
    tools={
        "get_flights": get_flights,
        "suggest_hotels": suggest_hotels
    }
)

ExploreAgent = Agent(
    name="ExploreAgent",
    instructions="Suggest local attractions, foods, and experiences in the selected destination."
)

@cl.on_chat_start
async def start():
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)
    cl.user_session.set("current_agent", DestinationAgent)
    await cl.Message(content="🌍 Welcome to the AI Travel Designer!\nTell me what you like, and I’ll design a dream trip for you.").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})
    user_input = message.content.lower()

    if any(word in user_input for word in ["book", "hotel", "flight"]):
        agent = BookingAgent
    elif any(word in user_input for word in ["see", "explore", "eat", "do there"]):
        agent = ExploreAgent
    else:
        agent = DestinationAgent

    cl.user_session.set("current_agent", agent)

    msg = cl.Message(content="")
    await msg.send()

    try:
        if agent == BookingAgent:
            destination = None
            for word in ["paris", "london", "tokyo", "karachi", "istanbul"]:
                if word in user_input:
                    destination = word.title()
                    break

            if destination:
                flights = get_flights(destination)
                hotels = suggest_hotels(destination)
                await msg.update(content=f"📍 Your Travel Plan for *{destination}*:\n\n{flights}\n\n{hotels}")
                history.append({"role": "assistant", "content": msg.content})
                cl.user_session.set("chat_history", history)
                return

        result = Runner.run_streamed(agent, history, run_config=cast(RunConfig, config))
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                await msg.stream_token(event.data.delta)

        history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("chat_history", history)

    except Exception as e:
        await msg.update(content=f"❌ Something went wrong: {str(e)}")
