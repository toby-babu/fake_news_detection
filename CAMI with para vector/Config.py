UNKNOWN = "UNK"
ROOT = "ROOT"
NULL = "NULL"
NONEXIST = -1

hidden_size = 200
learning_rate = 0.1
validation_step = 200


embedding_dim = 100
filter_sizes=[3,4,5]
num_filters=128
dropout_keep_prob=0.5
l2_reg_lambda=0.0

# Training parameters
batch_size=8
num_epochs=5000
evaluate_every=100
stop_step=5000
display_step = 10
num_classes = 2
no_intervals_event = 20
# Misc Parameters
allow_soft_placement=True
log_device_placement=False