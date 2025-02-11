from typing import Annotated

import typer
from rich import print

# logger = logging.getLogger(__name__)
from proto_docs.logger import logger

app = typer.Typer()


@app.callback()
def main():
    logger.debug("Calling main() function")
    logger.debug("Exiting main() function")


@app.command()
def commander_1(arg_1: Annotated[int, typer.Option()] = 1):
    logger.info("Calling commander_1 function")
    print(f"Another command with {arg_1=}")
    logger.info("Exiting commander_1 function")
