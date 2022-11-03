from data import create_movielens_data

(
    train_raw_sequences,
    test_raw_sequences,
    user_metadata,
    movie_metadata,
) = create_movielens_data()

print(train_raw_sequences["1"])
print(test_raw_sequences["1"])
print(user_metadata["5965"])
print(movie_metadata["1"])
