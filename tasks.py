import invoke


@invoke.task
def test(c):  # type: ignore
    invoke.run("poetry run python3 -m unittest discover tests")


@invoke.task
def lint(c):  # type: ignore
    flake_options = [
        "--ignore=E203,W503",
        "--max-line-length=88",
        "--show-source",
        "--statistics",
    ]
    invoke.run("poetry run flake8 src/ " + " ".join(flake_options))
    mypy_options = [
        "--follow-imports=silent",
        "--ignore-missing-imports",
        "--show-column-numbers",
        "--no-warn-return-any",
        "--no-implicit-optional",
        "--disallow-untyped-calls",
        "--no-pretty",
        "--disallow-untyped-defs",
        "--allow-redefinition",
    ]
    invoke.run("poetry run mypy src/ " + " ".join(mypy_options))
    black_options = [
        "--check",
        "--diff",
        "--color",
    ]
    invoke.run("poetry run black src/ " + " ".join(black_options))
