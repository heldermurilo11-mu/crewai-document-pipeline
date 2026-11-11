"""Isolated subprocess helper entrypoints for supervisor tests."""

from pathlib import Path
import subprocess
import sys
import time


def result(value):
    return {"value": value}


def blocked():
    time.sleep(60)


def chatter():
    for _ in range(5000):
        print("x" * 1000)
    return {"complete": True}


def failure():
    raise ValueError("controlled failure")


def descendants(marker):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(marker).write_text(str(child.pid), encoding="utf-8")
    time.sleep(60)
