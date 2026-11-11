"""Optional CrewAI compatibility layer.

Uses genuine ``crewai`` decorators and base classes when ``crewai`` is installed,
while allowing standalone pipeline runs and offline unit tests to import all
tools cleanly when ``crewai`` is absent from a lean environment.
"""

from functools import wraps

try:
    from crewai import Agent, Crew, Process, Task
    from crewai.project import CrewBase, agent, crew, task
    from crewai.tools import BaseTool, tool
    CREWAI_AVAILABLE = True
except ImportError:
    from pydantic import BaseModel, ConfigDict

    CREWAI_AVAILABLE = False

    class BaseTool(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        name: str = ""
        description: str = ""

        def run(self, *args, **kwargs):
            return self._run(*args, **kwargs)

    def tool(_name=None):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            wrapper.name = _name or fn.__name__
            wrapper.func = fn
            return wrapper

        if callable(_name):
            fn = _name
            _name = fn.__name__
            return decorator(fn)
        return decorator

    def CrewBase(cls):
        return cls

    def agent(fn):
        return fn

    def task(fn):
        return fn

    def crew(fn):
        return fn

    class _Stub:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class Process:
        sequential = "sequential"

    Agent = _Stub
    Crew = _Stub
    Task = _Stub
