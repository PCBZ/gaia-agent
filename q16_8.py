from gaia.web_solver import WebSolver
from gaia.base import openrouter_llm

print(WebSolver(16).resolve(llm=openrouter_llm()))
print(WebSolver(8).resolve(llm=openrouter_llm()))