import invoke


@invoke.task
def run(c):  # type: ignore
    invoke.run("python3 train.py --load_model --epochs=0")
