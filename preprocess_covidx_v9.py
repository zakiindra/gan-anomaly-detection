# split into pos and neg train
import os

def extract_file_label(source_file, target_dir, save_pos_file, save_neg_file):
    pos = []
    neg = []
    with open(source_file, "r") as f:
        for line in f:
            x = line.strip().split(" ")
            if x[2] == "positive":
                pos.append(x[1])
            elif x[2] == "negative":
                neg.append(x[1])
            else:
                print(x)

    with open(save_pos_file, "w") as f:
        f.write("\n".join([os.path.join(target_dir, filename) for filename in pos]))
    with open(save_neg_file, "w") as f:
        f.write("\n".join([os.path.join(target_dir, filename) for filename in neg]))
    return pos, neg


train_source_file = "datasets/covid/train.txt"
test_source_file = "datasets/covid/test.txt"
train_pos_file = "datasets/covid/train_pos.txt"
train_neg_file = "datasets/covid/train_neg.txt"
test_pos_file = "datasets/covid/test_pos.txt"
test_neg_file = "datasets/covid/test_neg.txt"
train_target_dir = "datasets/covid/train"
test_target_dir = "datasets/covid/test"

pos_train, neg_train = extract_file_label(train_source_file, train_target_dir, train_pos_file, train_neg_file)
pos_test, neg_test = extract_file_label(test_source_file, test_target_dir, test_pos_file, test_neg_file)
