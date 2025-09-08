import os
from smolagents import CodeAgent, InferenceClientModel
from a_tools import search_tool, final_answer, image_generation_tool

extra_instructions = """

For your final answer, always use:
final_answer(<answer>)
Do NOT include any HTML tags. Return only Python code in your final output.

Example:
city_name = "St. Petersburg"
final_answer(city_name)
"""

class AG_Agent:
    def __init__(self):
        a_model = InferenceClientModel(
            model_id='meta-llama/Llama-3.3-70B-Instruct',
            max_tokens=1024,
            api_key=os.environ.get("HF_API_KEY"),
        )
        self.model = a_model
        self.tools = [search_tool, final_answer, image_generation_tool]
        self.agent = CodeAgent(tools=self.tools, model=self.model, stream_outputs=True, instructions=extra_instructions)
        self.context = []
        self.max_context_len = 20
        self.max_steps = 15

    def ask(self, question):
        # Refresh context and add it to prompt
        self.context.append({"role": "user", "content": question})
        if len(self.context) > self.max_context_len:
            self.context = self.context[-self.max_context_len:]
        # Combine story for prompt (optional)
        prompt = ""
        for msg in self.context:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Agent: {msg['content']}\n"
        prompt += f"User: {question}\nAgent:"
        # Run agent with created prompt
        response = self.agent.run(prompt, max_steps=self.max_steps)
        # Add answer to history for context
        self.context.append({"role": "assistant", "content": response})
        if len(self.context) > self.max_context_len:
            self.context = self.context[-self.max_context_len:]
        return response

