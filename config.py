class Config:
    data_dir = "data/"
    vec_file = "data/vec.txt"
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 10  # gradient clipping
    num_layers = 2
    num_steps = 101  # this value is one more than max number of words in sentence
    hidden_size = 20
    word_embedding_size = 100
    max_epoch = 30
    max_max_epoch = 80
    keep_prob = 0.5  # The probability that each element is kept through dropout layer
    lr_decay = 1.0
    batch_size = 16
    vocab_size = 16119 - 4
    num_keywords = 5
    save_freq = 10  # The step (counted by the number of iterations) at which the model is saved to hard disk.
    model_dir = "model/"  # the path of model that need to save or load

    # parameter for generation
    len_of_generation = 16  # The number of characters by generated
    save_time = 20  # load save_time saved models
    is_sample = True  # true means using sample, if not using argmax
    BeamSize = 2
