from data import create_hm_data
from doc2vec import AttentiveDoc2Vec


def main() -> None:
    num_cluster = 10

    d_model = 100
    batch_size = 64
    epochs = 5
    lr = 0.0005

    raw_sequences, item_name_dict = create_hm_data()

    doc2vec = AttentiveDoc2Vec(
        raw_sequences=raw_sequences, d_model=d_model,
        batch_size=batch_size, epochs=epochs, lr=lr, model='model',
        model_path='weights/model.pt', word2vec_path='weights/word2vec_hm.model',
        verbose=True)
    _ = doc2vec.train()

    doc2vec.top_items(num_cluster=num_cluster, item_name_dict=item_name_dict, show_fig=True)


if __name__ == '__main__':
    main()
