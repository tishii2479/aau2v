import invoke


@invoke.task
def train_ml(c):  # type: ignore
    params = [
        "--epochs=10",
        "--model_name=attentive",
        "--cache_dir=cache/ml/",
    ]
    invoke.run("python3 train.py " + " ".join(params))


@invoke.task
def run_ml(c):  # type: ignore
    params = [
        "--load_model",
        "--epochs=0",
        "--model_name=attentive",
        "--cache_dir=cache/ml/",
    ]
    invoke.run("python3 train.py " + " ".join(params))


@invoke.task
def run_ml_debug(c):  # type: ignore
    params = [
        "--load_model",
        "--epochs=0",
        "--model_name=attentive",
        "--cache_dir=cache/ml-debug/",
    ]
    invoke.run("python3 train.py " + " ".join(params))


@invoke.task
def run_doc2vec(c):  # type: ignore
    params = [
        "--load_model",
        "--epochs=0",
        "--model_name=doc2vec",
        "--cache_dir=cache/ml-doc2vec/",
    ]
    invoke.run("python3 train.py " + " ".join(params))


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
    invoke.run("poetry run flake8 . " + " ".join(flake_options))
    mypy_options = [
        "--follow-imports=silent",
        "--ignore-missing-imports",
        "--show-column-numbers",
        "--warn-return-any",
        "--no-implicit-optional",
        "--disallow-untyped-calls",
        "--no-pretty",
        "--disallow-untyped-defs",
        "--allow-redefinition",
    ]
    invoke.run("poetry run mypy . " + " ".join(mypy_options))
    black_options = [
        "--check",
        "--diff",
        "--color",
    ]
    invoke.run("poetry run black . " + " ".join(black_options))
