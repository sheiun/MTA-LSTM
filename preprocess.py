import os
import pickle

import numpy as np
import tensorflow as tf

from config import Config



def read_wordvec(config) -> (list, list):
    with open(config.vec_file, "r") as fvec:
        word_ls = []
        vec_ls = []
        fvec.readline()

        word_ls.append("PAD")
        vec_ls.append([0] * config.word_embedding_size)
        word_ls.append("START")
        vec_ls.append([0] * config.word_embedding_size)
        word_ls.append("END")
        vec_ls.append([0] * config.word_embedding_size)
        word_ls.append("UNK")
        vec_ls.append([0] * config.word_embedding_size)
        config.vocab_size += 4
        for line in fvec:
            line = line.split()
            word = line[0]
            vec = [float(i) for i in line[1:]]
            if len(vec) == config.word_embedding_size:
                word_ls.append(word)
                vec_ls.append(vec)
            else:
                print(f"{len(vec)} not same wtih {config.word_embedding_size}")
        assert len(word_ls) == config.vocab_size
        word_vec = np.array(vec_ls, dtype=np.float32)

        pickle.dump(
            word_vec, open("word_vec.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL
        )
        pickle.dump(
            word_ls, open("word_voc.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL
        )

    return word_ls, word_vec


def read_data(config) -> list:
    training_data = []
    with open(os.path.join(config.data_dir, "train.txt"), "r") as ftext:
        for line in ftext:
            tmp = line.split()
            idx = tmp.index("</d>")
            doc = tmp[:idx]
            keywords = tmp[idx + 1 :]
            assert len(keywords) == 5

            training_data.append((doc, keywords))
    return training_data


def data_iterator(training_data, batch_size, num_steps):
    epoch_size = len(training_data) // batch_size
    for i in range(epoch_size):
        batch_data = training_data[i * batch_size : (i + 1) * batch_size]
        raw_data = []
        key_words = []
        for it in batch_data:
            raw_data.append(it[0])
            tmp = []
            for wd in it[1]:
                tmp.append(word_to_idx[wd])
            key_words.append(tmp)

        data = np.zeros((len(raw_data), num_steps + 1), dtype=np.int64)
        for i in range(len(raw_data)):
            doc = raw_data[i]
            tmp = [1]
            for wd in doc:
                if wd in vocab:
                    tmp.append(word_to_idx[wd])
                else:
                    tmp.append(3)
            tmp.append(2)
            tmp = np.asarray(tmp, dtype=np.int)
            data[i][: tmp.shape[0]] = tmp

        key_words = np.array(key_words, dtype=np.int64)

        x = data[:, 0:num_steps]
        y = data[:, 1:]
        mask = np.float32(x != 0)
        yield (x, y, mask, key_words)


config = Config()
print("loading the training data...")
vocab, _ = read_wordvec(config)

data = read_data(config)

word_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_word = {i: ch for i, ch in enumerate(vocab)}

print(f"data has {len(data)} document, size of word vocabular: {len(vocab)}.")

writer = tf.python_io.TFRecordWriter("coverage_data")
data_ls = []
for step, (x, y, mask, key_words) in enumerate(
    data_iterator(data, config.batch_size, config.num_steps)
):
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
            feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                "input_data": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=x.reshape(-1).astype("int64"))
                ),
                "target": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=y.reshape(-1).astype("int64"))
                ),
                "mask": tf.train.Feature(
                    float_list=tf.train.FloatList(
                        value=mask.reshape(-1).astype("float")
                    )
                ),
                "key_words": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=key_words.reshape(-1).astype("int64")
                    )
                ),
            }
        )
    )
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)
print("total step: ", step)
