import invoke


@invoke.task
def run(c):  # type: ignore
    params = [
        "--load_model",
        "--epochs=0",
        "--model_name=attentive",
        "--model_path=weights/model_hm_meta.pt",
    ]
    invoke.run("python3 train.py " + " ".join(params))


@invoke.task
def run_doc2vec(c):  # type: ignore
    params = [
        "--load_model",
        "--epochs=0",
        "--model_name=doc2vec",
        "--model_path=weights/doc2vec_hm.pt",
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
