import numpy as np
import os


def create_train_valid_split(output_fn, n_sample, train_factor=9, seed=1234):
    np.random.seed(seed)

    data_ids = np.arange(0, n_sample)
    n_train = int(n_sample * train_factor / (train_factor + 1))
    n_valid = n_sample - n_train
    train_inds = np.sort(np.random.choice(data_ids, n_train, replace=False))
    valid_inds = np.sort(np.setdiff1d(data_ids, train_inds))

    np.savez(output_fn, train_inds=train_inds, valid_inds=valid_inds)


def load_split_file(split_fn):
    split_data = np.load(split_fn)
    train_inds = split_data['train_inds']
    valid_inds = split_data['valid_inds']
    return train_inds, valid_inds


if __name__ == '__main__':
    output_dir = os.path.join('data', 'pop909_mel')
    os.makedirs(output_dir, exist_ok=True)

    output_fn = os.path.join(output_dir, 'split.npz')
    create_train_valid_split(output_fn, 909, train_factor=9, seed=1234)

    t_id, v_id = load_split_file(output_fn)
    print(f'n_train={len(t_id)}, n_valid={len(v_id)}')
    print(t_id)
    print(v_id)
