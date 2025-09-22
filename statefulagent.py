import os
from smolagents import CodeAgent, InferenceClientModel
from a_tools import search_tool, final_answer, image_generation_tool

class StatefulAgent:
    def __init__(self, model_id = 'Qwen/Qwen2.5-Coder-32B-Instruct', context_len = 0, structured_code = False):
        a_model = InferenceClientModel(
            model_id=model_id,
            max_tokens=1024,
            api_key=os.environ.get("HF_API_KEY"),
        )
        self.model = a_model
        self.tools = [search_tool, final_answer, image_generation_tool]
        self.agent = CodeAgent(tools=self.tools,
                               model=self.model,
                               stream_outputs=True,
                               code_block_tags="markdown",
                               use_structured_outputs_internally=structured_code,
                               additional_authorized_imports=['requests', 'bs4','pandas','numpy',
                                                              'json','datetime','geopandas','shapely'])
        self.context = []
        self.max_context_len = context_len
        self.max_steps = 20

    def ask(self, question):
        # Refresh context and add it to prompt
        self.context.append({"role": "user", "content": question})
        if len(self.context) > self.max_context_len:
            self.context = self.context[-self.max_context_len:]
        # Combine history for prompt (optional)
        prompt = ""
        for msg in self.context:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Agent: {msg['content']}\n"
        if self.max_steps > 0:
            prompt += f"User: {question}\nAgent:"
        # Run agent with created prompt
        response = self.agent.run(prompt, max_steps=self.max_steps)
        # Add answer to history for context
        if self.max_steps > 0:
            self.context.append({"role": "assistant", "content": response})
        if len(self.context) > self.max_context_len:
            self.context = self.context[-self.max_context_len:]
        return response

