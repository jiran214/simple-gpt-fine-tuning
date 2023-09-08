#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/7 17:22
# @Author  : 雷雨
# @File    : utils.py
# @Desc    :
import json


def dump_json(dataset, data_path):
    with open(data_path, 'w', encoding='utf-8') as f:
        chunk = 10
        lines = []
        for data in dataset:
            lines.append(json.dumps(data) + '\n')
            if len(lines) >= chunk:
                f.writelines(lines)
                lines.clear()
        if lines:
            f.writelines(lines)


def load_json(data_path):
    # Load the dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)
    return dataset


def get_cost(dataset, convo_lens):
    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096

    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)
    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")


def split_dataset(dataset, dataset_path, train_file_name, validation_file_name, scale=5/1):
    """
    split dataset
    :param dataset_path:
    :param dataset:
    :param train_file_name:
    :param validation_file_name:
    :param scale:  n_train / n_validation
    :example
        dataset_path = '/Users/xiye/PycharmProjects/simple-gpt-fine-tuning/dataset'
        dataset = load_json(dataset_path + "/toy_chat_fine_tuning.jsonl")
        split_dataset(dataset, dataset_path, 'train.jsonl', 'test.jsonl', 2/1)
    :return:
    """
    if not dataset_path.endswith('/'):
        dataset_path = dataset_path + '/'
    n_dataset = len(dataset)
    n_train = int(n_dataset * scale / (scale+1))
    # n_validation = n_dataset - n_train
    dump_json(dataset[:n_train], dataset_path + train_file_name)
    dump_json(dataset[n_train:], dataset_path + validation_file_name)
