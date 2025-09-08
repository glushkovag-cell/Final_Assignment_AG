import os
from smolagents import Tool

os.environ["SERPAPI_API_KEY"] = "06da1dedae2ed008f4d95e857c123fb1a3a6b20abd3a88de357d37a586fb70f3"

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

search_tool = Tool.from_langchain(load_tools(["serpapi"]))
