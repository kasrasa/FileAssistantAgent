import os
import asyncio
from pydantic import BaseModel
import openai
from agents import (
    Agent, Runner, function_tool, input_guardrail,
    GuardrailFunctionOutput, InputGuardrailTripwireTriggered,
    OutputGuardrailResult, output_guardrail, 
    OutputGuardrailTripwireTriggered, RunContextWrapper,
    TResponseInputItem
)


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()
BASE_DIR = "C:/Users/ksada/Desktop/experiments/FileAssistantAgent/outputs"

# 1) Define the guardrail output schema
class FileContentOutput(BaseModel):
    is_file_content_safe: bool
    reasoning: str

# 2) Input guardrail: validate content is safe
@input_guardrail()
async def prompt_guardrail(ctx: RunContextWrapper[None], agent: Agent, prompt) -> GuardrailFunctionOutput:
    result = await Runner.run(safety_agent, prompt, context=ctx)
    safe = result.final_output.is_file_content_safe
    if not safe:
        raise InputGuardrailTripwireTriggered(result.final_output.reasoning)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not safe
    )


class GeneratedContentSafety(BaseModel):
    is_safe: bool
    reasoning: str

@output_guardrail()
async def generated_content_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    content: str | list[TResponseInputItem]
) -> OutputGuardrailResult:
    result = await Runner.run(safety_agent, content, context=ctx)
    safe = result.final_output.is_safe
    
    if not safe:
        raise OutputGuardrailTripwireTriggered(result.final_output)
    
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not safe
    )

# 3) Tool: generate content
@function_tool
def generate_content(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system", 
            "content": "Never include sensitive or harmful content"
        }, {
            "role": "user", 
            "content": prompt
        }],
        temperature=0.7,
    )
    return resp.choices[0].message.content

# 4) Tool: create & populate file
@function_tool
def create_populate_file(file_name: str, content: str) -> str:

    directory = BASE_DIR
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(BASE_DIR,file_name)

    try:
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(content)
        return f"File '{file_name}' created and populated successfully."
    except OSError as e:
        return f"Error creating file: {e}. Check path/permissions."

# 5) Build the agent
safety_agent = Agent(
    name="SafetyChecker",
    instructions="Analyze content safety only - no tools",
    tools=[]
)


create_populate_agent = Agent(
    name="FileAssistant",
    instructions=(
        "You have two tools: create_populate_file(file_name, content) "
        "and generate_content(prompt). Use generate_content to produce "
        "the text, then call create_populate_file to write it. "
        "If content_guardrail flags unsafe content, reply '<TRIPWIRE_TRIGGERED>'."
    ),
    tools=[generate_content, create_populate_file]
)

# 6) Driver
async def main():
    prompt = input("Pick any topic you’d like to know more about: ")
    result = await Runner.run(create_populate_agent, prompt)

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
