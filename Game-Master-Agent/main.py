import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, handoff
from agents.run import RunConfig, RunContextWrapper
import random

# === Load environment variables ===
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

# === Gemini-compatible client ===
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

# === Tool Functions ===

def roll_dice(sides: int = 20) -> int:
    return random.randint(1, sides)

def generate_event(context: str) -> str:
    events = {
        "forest": [
            "You hear rustling in the bushes. A goblin appears!",
            "You find an ancient tree with glowing runes.",
            "A traveling merchant offers you a mysterious potion."
        ],
        "dungeon": [
            "A trap triggers beneath your feet!",
            "A skeleton warrior blocks your path.",
            "You discover a chest filled with gold... or is it a mimic?"
        ],
        "village": [
            "A child runs up to you, asking for help.",
            "The blacksmith offers to upgrade your weapon.",
            "You overhear talk of a dragon nearby."
        ]
    }
    return random.choice(events.get(context.lower(), ["Nothing unusual happens..."]))

# === Agents ===

NarratorAgent = Agent(
    name="NarratorAgent",
    instructions="Narrate the fantasy adventure based on player decisions. Use vivid descriptions and advance the story.",
    model=model
)

MonsterAgent = Agent(
    name="MonsterAgent",
    instructions="Control monster behavior during combat. Ask the user what action they take (attack, defend, run), then narrate outcome using dice roll.",
    model=model,
    tools={"roll_dice": roll_dice}
)

ItemAgent = Agent(
    name="ItemAgent",
    instructions="Describe items found by the player and manage inventory. Assign rewards after events or combat.",
    model=model,
    tools={"generate_event": generate_event}
)

# === Main Game Master Agent with Proper Handoffs ===
GameMasterAgent = Agent(
    name="GameMasterAgent",
    instructions="""You are a fantasy adventure game master that orchestrates an epic quest. 

You have access to three specialized agents:
1. NarratorAgent - for story narration and adventure progression
2. MonsterAgent - for combat encounters and dice-based battles
3. ItemAgent - for inventory management and reward distribution

Use the appropriate handoff tool when:
- User wants to explore, move, or progress story → use handoff_to_narrator
- User wants to fight, attack, or engage in combat → use handoff_to_monster
- User wants to check inventory, collect items, or get rewards → use handoff_to_item

Create an immersive fantasy experience and guide players through their adventure!""",
    model=model,
    handoffs=[
        handoff(NarratorAgent, tool_name_override="handoff_to_narrator", tool_description_override="Handoff to NarratorAgent for story progression"),
        handoff(MonsterAgent, tool_name_override="handoff_to_monster", tool_description_override="Handoff to MonsterAgent for combat encounters"),
        handoff(ItemAgent, tool_name_override="handoff_to_item", tool_description_override="Handoff to ItemAgent for inventory and rewards"),
    ]
)

# === Chat Start ===
@cl.on_chat_start
async def start():
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)
    cl.user_session.set("current_agent", NarratorAgent)
    await cl.Message(content="🧙 Welcome, adventurer! Your quest begins now...\n\nTell me what you'd like to do — explore a forest, enter a dungeon, or visit a village?").send()

# === Message Handling ===
@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})
    user_input = message.content.lower()

    # Agent Handoff Logic with Beautiful Messages
    previous_agent = cl.user_session.get("current_agent")
    
    if any(word in user_input for word in ["attack", "defend", "monster", "fight", "battle", "combat", "enemy"]):
        agent = MonsterAgent
    elif any(word in user_input for word in ["item", "chest", "reward", "loot", "inventory", "collect", "treasure"]):
        agent = ItemAgent
    else:
        agent = NarratorAgent

    # Show handoff message if agent changed
    if previous_agent != agent:
        agent_info = {
            "NarratorAgent": {
                "emoji": "📖",
                "description": "I'll narrate your adventure and guide you through the story!"
            },
            "MonsterAgent": {
                "emoji": "⚔️",
                "description": "I'll handle combat encounters and dice-based battles!"
            },
            "ItemAgent": {
                "emoji": "🎁",
                "description": "I'll manage your inventory and distribute rewards!"
            }
        }
        
        info = agent_info.get(agent.name, {"emoji": "🤖", "description": "I'll help you with your adventure!"})
        await cl.Message(
            content=f"{info['emoji']} **Switching to {agent.name}**\n\n{info['description']}",
            author="System"
        ).send()

    cl.user_session.set("current_agent", agent)

    msg = cl.Message(content="")
    await msg.send()

    try:
        # Manual tool trigger for ItemAgent (event generator)
        if agent == ItemAgent:
            context = None
            for area in ["forest", "dungeon", "village"]:
                if area in user_input:
                    context = area
                    break

            if context:
                event = generate_event(context)
                msg.content = f"🎁 You discover:\n\n{event}"
                await msg.update()
                history.append({"role": "assistant", "content": msg.content})
                cl.user_session.set("chat_history", history)
                return

        # Manual tool trigger for MonsterAgent (dice roller)
        if agent == MonsterAgent:
            roll = roll_dice()
            outcome = "🗡️ Critical Hit!" if roll > 15 else "💢 Weak strike..." if roll < 5 else "⚔️ You strike the enemy."
            msg.content = f"You rolled a {roll}.\n{outcome}"
            await msg.update()
            history.append({"role": "assistant", "content": msg.content})
            cl.user_session.set("chat_history", history)
            return

        # Run synchronous response for all other agents
        result = Runner.run_sync(agent, history, run_config=cast(RunConfig, config))
        final = result.final_output

        msg.content = final
        await msg.update()

        history.append({"role": "assistant", "content": final})
        cl.user_session.set("chat_history", history)

    except Exception as e:
        msg.content = f"❌ Error: {str(e)}"
        await msg.update()
        print(f"Error: {e}")