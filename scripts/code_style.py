# run_commands.py

import os
import subprocess

commands = [
    "black . --config pyproject.toml",
    "isort .",
]

for command in commands:
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    if output:
        print("Output:", output.decode())
    if error:
        print("Error:", error.decode())
