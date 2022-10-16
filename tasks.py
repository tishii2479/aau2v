import invoke


@invoke.task
def run(c):  # type: ignore
    invoke.run(
        "python3 train.py --load_model --epochs=0 --model_path=weights/model_hm_meta.pt"
    )


@invoke.task
def run_doc2vec(c):  # type: ignore
    invoke.run(
        "python3 train.py  --model_name=doc2vec --model_path=weights/doc2vec_hm.pt"
    )
