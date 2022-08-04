from argparse import ArgumentParser, Namespace

from data import create_20newsgroup_data, create_hm_data  # noqa
from model import AttentiveDoc2Vec


def main() -> None:
    def parse_args() -> Namespace:
        parser = ArgumentParser()
        parser.add_argument('--num_cluster', type=int, default=10)
        parser.add_argument('--d_model', type=int, default=100)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--lr', type=float, default=0.0005)
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--load_model', action='store_true')
        return parser.parse_args()

    args = parse_args()

    raw_sequences, item_name_dict = create_20newsgroup_data(max_data_size=1000)

    doc2vec = AttentiveDoc2Vec(
        raw_sequences=raw_sequences, d_model=args.d_model,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, model='attentive',
        model_path='weights/model_20news.pt', word2vec_path='weights/word2vec_hm.model',
        verbose=args.verbose, load_model=args.load_model)
    _ = doc2vec.train()

    doc2vec.top_items(num_cluster=args.num_cluster, item_name_dict=item_name_dict, show_fig=True)
    _ = doc2vec.calc_coherence(num_cluster=args.num_cluster)


if __name__ == '__main__':
    main()
