from smolagents import Tool, FinalAnswerTool, WebSearchTool

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

search_tool = WebSearchTool()

final_answer = FinalAnswerTool()