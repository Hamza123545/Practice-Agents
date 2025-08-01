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

# === Career roadmap tool ===
def get_career_roadmap(field: str) -> str:
    field = field.lower()
    if "software" in field:
        return "🧑‍💻 Software Engineering Roadmap:\n1. Learn Python or Java\n2. Study Data Structures\n3. Build full-stack apps\n4. Version control (Git)\n5. Interview prep"
    elif "data" in field:
        return "📊 Data Science Roadmap:\n1. Python & Statistics\n2. Pandas, NumPy, Scikit-learn\n3. ML Models\n4. Kaggle projects\n5. Portfolio & Jobs"
    elif "medicine" in field:
        return "🩺 Medical Field Roadmap:\n1. Pre-med subjects\n2. Medical entrance tests\n3. MBBS studies\n4. Clinical rotations\n5. Specialization"
    else:
        return f"⚠️ No roadmap found for '{field}'. Try software, data, or medicine."

# === Handoff Functions ===
def on_handoff_to_skill(ctx: RunContextWrapper[None]) -> Agent:
    # Note: We can't use async/await here due to handoff function signature
    print("📚 Switching to SkillAgent - I'll create a detailed skill roadmap!")
    return SkillAgent

def on_handoff_to_job(ctx: RunContextWrapper[None]) -> Agent:
    print("💼 Switching to JobAgent - I'll help you explore job roles and salaries!")
    return JobAgent

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
        name="SkillAgent",
        instructions="You provide step-by-step skill roadmaps based on the user's career interest. Ask for their target field and use get_career_roadmap().",
        model=model,
        tools={"get_career_roadmap": get_career_roadmap}
    )

    job_agent = Agent(
        name="JobAgent",
        instructions="You suggest popular job roles, responsibilities, and how to prepare for them.",
        model=model
    )

    # === Main Agent with Proper Handoffs ===
    career_agent = Agent(
        name="CareerAgent",
        instructions="""You are a friendly AI that helps users explore careers. 

You have access to two specialized agents:
1. SkillAgent - for creating skill roadmaps and learning paths
2. JobAgent - for exploring job roles, salaries, and career preparation

Use the appropriate handoff tool when:
- User asks about skills, learning, or skill development → use handoff_to_skill
- User asks about job titles, salaries, or career preparation → use handoff_to_job

Always be helpful and guide users to the right specialist.""",
        model=model,
        handoffs=[
            handoff(skill_agent, tool_name_override="handoff_to_skill", tool_description_override="Handoff to SkillAgent for skill roadmaps"),
            handoff(job_agent, tool_name_override="handoff_to_job", tool_description_override="Handoff to JobAgent for job roles and salaries"),
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

        # Check if handoff occurred - with better error handling
        try:
            if hasattr(result, 'final_agent') and result.final_agent and result.final_agent != agent:
                # Show handoff message
                agent_info = {
                    "SkillAgent": {
                        "emoji": "📚",
                        "description": "I'll create a detailed skill roadmap to help you succeed in your chosen field!"
                    },
                    "JobAgent": {
                        "emoji": "💼",
                        "description": "I'll help you explore job roles, salaries, and career preparation strategies!"
                    }
                }
                
                # Safely get agent name
                agent_name = getattr(result.final_agent, 'name', str(result.final_agent))
                info = agent_info.get(agent_name, {"emoji": "🤖", "description": "I'll help you with your request!"})
                
                await cl.Message(
                    content=f"{info['emoji']} **Switching to {agent_name}**\n\n{info['description']}",
                    author="System"
                ).send()
        except Exception as handoff_error:
            print(f"Handoff message error: {handoff_error}")
            # Continue without showing handoff message

        msg.content = final
        await msg.update()

        history.append({"role": "assistant", "content": final})
        cl.user_session.set("chat_history", history)

    except Exception as e:
        msg.content = f"❌ Error: {str(e)}"
        await msg.update()
        print(f"Error: {e}")
