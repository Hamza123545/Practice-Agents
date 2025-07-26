import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, handoff
from agents.run import RunConfig, RunContextWrapper

# Load .env
load_dotenv()

# Validate API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

# === Skill roadmap tool ===
def get_skill_roadmap(field: str) -> str:
    field = field.lower()
    if "software" in field:
        return "🧑‍💻 Software Engineering Roadmap:\n1. Learn Python or Java\n2. Study Data Structures\n3. Build full-stack apps\n4. Version control (Git)\n5. Interview prep"
    elif "data" in field:
        return "📊 Data Science Roadmap:\n1. Python & Statistics\n2. Pandas, NumPy, Scikit-learn\n3. ML Models\n4. Kaggle projects\n5. Portfolio & Jobs"
    elif "medicine" in field:
        return "🩺 Medical Field Roadmap:\n1. Pre-med subjects\n2. Medical entrance tests\n3. MBBS studies\n4. Clinical rotations\n5. Specialization"
    else:
        return f"⚠️ No roadmap found for '{field}'. Try software, data, or medicine."

# === on_handoff function ===
def on_handoff(agent: Agent, ctx: RunContextWrapper[None]):
    cl.Message(
        content=f"🔄 **Switching to `{agent.name}`** to help you better.",
        author="System"
    ).send()

# === on_chat_start ===
@cl.on_chat_start
async def start():
    client = AsyncOpenAI(
        api_key=gemini_api_key,
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

    # === Specialized Agents ===
    skill_agent = Agent(
        name="Skill Advisor",
        instructions="You provide step-by-step skill roadmaps based on the user's career interest. Ask for their target field and use get_skill_roadmap().",
        model=model,
        tools={"get_skill_roadmap": get_skill_roadmap}
    )

    job_agent = Agent(
        name="Job Expert",
        instructions="You suggest popular job roles, responsibilities, and how to prepare for them.",
        model=model
    )

    # === Main Agent with Handoff ===
    career_agent = Agent(
        name="Career Mentor",
        instructions="You are a friendly AI that helps users explore careers. If the user asks about skills or learning, handoff to Skill Advisor. If they ask about job titles or salaries, handoff to Job Expert.",
        model=model,
        handoffs=[
            handoff(skill_agent, on_handoff=lambda ctx: on_handoff(skill_agent, ctx)),
            handoff(job_agent, on_handoff=lambda ctx: on_handoff(job_agent, ctx)),
        ]
    )

    # Store in session
    cl.user_session.set("agent", career_agent)
    cl.user_session.set("config", config)
    cl.user_session.set("chat_history", [])

    await cl.Message(
        content="🎓 **Welcome to Career Mentor AI!**\n\nTell me your career goals or interests, and I’ll guide you through next steps."
    ).send()

# === on_message ===
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

        msg.content = final
        await msg.update()

        history.append({"role": "developer", "content": final})
        cl.user_session.set("chat_history", history)

    except Exception as e:
        msg.content = f"❌ Error: {e}"
        await msg.update()
        print(f"Error: {e}")
