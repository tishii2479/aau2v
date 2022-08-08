class TrainerConfig:
    model_name: str
    epochs: int
    batch_size: int


class ModelConfig:
    d_model: int
    window_size: int
    negative_sample_size: int
    lr: float
