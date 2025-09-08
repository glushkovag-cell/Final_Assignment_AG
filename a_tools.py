from smolagents import Tool, FinalAnswerTool
from langchain_community.agent_toolkits.load_tools import load_tools

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

final_answer = FinalAnswerTool()