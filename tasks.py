import os

from invoke import task

WINDOWS = os.name == "nt"
PROJECT_NAME = "call_of_birds_autobird"
PYTHON_VERSION = "3.12"


def _pty() -> bool:
    return not WINDOWS




# Git helpers
@task
def git(ctx, message):
    ctx.run("git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")


