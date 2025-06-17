from agents import Agent, Runner, function_tool
import openai
import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()
BASE_DIR = os.path.abspath(os.getcwd())

@function_tool
def generate_content(topic: str):
    moderation_response = client.moderations.create(
        input=topic,
    )
    response_flagged = moderation_response.results[0]
    if response_flagged.flagged:
        flagged_categories = [category for category in response_flagged.categories if response_flagged.categories[category]]
        raise ValueError(f"The topic is not appropriate for the content generation. Flagged categories: {flagged_categories}")
    
    
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=f"you are a helpful assistant that generates content for a given topic. generate sophisticated and relevant content for the topic.",
        input=topic,
        )
    return response.output_text


@function_tool
def create_populate_file(file_name: str, content: str):
    try:
        abs_path = os.path.abspath(file_name)
        if not abs_path.startswith(BASE_DIR + os.sep):
            return "Error: Invalid path. Please provide a path within the working directory."
        
        directory = os.path.dirname(abs_path) or BASE_DIR
        os.makedirs(directory, exist_ok=True)

        with open(abs_path, 'w', encoding="utf-8") as f:
            f.write(content)
        return f"File {file_name} created and populated successfully."
    except OSError as e:
        return f"Error creating file: {e}. please check the file path or permissions and try again."


agent = Agent(name="FileAssistant",
              instructions=(
                "You have two tools: create_populate_file(file_name, content), and "
                "generate_content(prompt). Respond by choosing the appropriate tool calls to satisfy the user's request."
            ),
              tools=[create_populate_file, generate_content]
             )

if __name__ == "__main__":
    print("AI File Agent ready. Enter instructions like: 'create test.txt and write Hello world in it'. to quit, type 'exit'")
    while True:
        cmd = input("\n> ")
        if cmd.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        result = Runner.run_sync(agent, cmd)
        if result.final_output.startswith("Error:"):
            print(result.final_output)
            continue
        print(result.final_output)
        
