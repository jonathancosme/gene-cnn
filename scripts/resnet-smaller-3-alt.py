#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import keras_nlp
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import plot_model
from keras.utils.layer_utils import count_params
from pathlib import Path
import pandas as pd
import wandb
from wandb.keras import WandbCallback


# In[ ]:


policy = keras.mixed_precision.Policy("mixed_float16")
keras.mixed_precision.set_global_policy(policy)


# In[ ]:


output_folder_path = '/mnt/DATA/nfs/data/gene-cnn'
dataset_name = 'dset-10pD-100pL'


# In[ ]:


project_name = "gene-cnn"
run_name = 'resnet-smaller-3-alt'
description = 'resnet smaller 3; 1x3 w/ stride=1 initial filter; no maxpool; 10% data; 100% labels; species only'


# In[ ]:


data_splits = {
    'train': 0.8,
    'val': 0.1,
    'test': 0.1,
}

label_file_id_col = 'id'
label_file_label_cols = [
    { 'name_col': 'Species', 'code_col': 'species_code'},
    { 'name_col': 'Genus', 'code_col': 'genus_code'},
]

labels_to_use = 'species'

max_input_len = 150

BATCH_SIZE = 1024
EPOCHS = 3
learning_rate = .0003

tf_logs_dir = 'tf_logs/'



# In[ ]:


max_seq_len = max_input_len
first_filter_size = 1
res_block_reps = [1, 1, 1, 1]
activation = 'relu'
gen_filter_size = 3
bn_momen = 0.6
LAST_ACTIVATION = 'softmax'

CENTER = True
SCALE = False
EPSILON = 1e-8


# In[ ]:


wandb.config = {
    "description": description,
    "output_folder_path": output_folder_path,
    "dataset_name": dataset_name,
    "label_file_id_col": label_file_id_col,
    "label_file_label_cols": label_file_label_cols,
    "max_input_len": max_input_len,
    "learning_rate": learning_rate,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "first_filter_size": first_filter_size,
    "res_block_reps": res_block_reps,
    "activation": activation,
    "gen_filter_size": gen_filter_size,
    "bn_momen": bn_momen,
    "LAST_ACTIVATION": LAST_ACTIVATION,
    "CENTER": CENTER,
    "SCALE": SCALE,
    "EPSILON": EPSILON,
}


# In[ ]:


data_splits = {x:data_splits[x] for x in sorted(data_splits, key=data_splits.get, reverse=True)}


# In[ ]:


unsplit_data_filepath = Path(output_folder_path) / dataset_name / f"{dataset_name}_unsplit.parquet"
data_label_filepath = Path(output_folder_path) / dataset_name / f"{dataset_name}_labels.csv"
vocab_filepath = Path(output_folder_path) / dataset_name / f"{dataset_name}_vocab.txt"
data_splits_filepaths = {}
for key, val in data_splits.items():
    data_splits_filepaths[key] = Path(output_folder_path) / dataset_name / f"{dataset_name}_{key}.parquet"
data_splits_filepaths_csv = {}
for key, val in data_splits.items():
    data_splits_filepaths_csv[key] = Path(output_folder_path) / dataset_name / f"{dataset_name}_{key}.csv"


# In[ ]:


assert (labels_to_use == 'genus') or (labels_to_use == 'species') or (labels_to_use == 'both'), f"\nvariable 'labels_to_use' must be 'genus' 'species' or 'both' \n currently set as {labels_to_use}"


# In[ ]:


raw_vocab_list = open(vocab_filepath, 'r').read().split('\n')
token_vocab_list = list(range(len(raw_vocab_list)))


# In[ ]:


raw_vocab_list


# In[ ]:


vocab_size = len(raw_vocab_list) - 1
vocab_size


# In[ ]:


label_cols = [x['code_col'] for x in label_file_label_cols]
all_classes = (pd.read_csv(data_label_filepath)[label_cols].max() + 1).tolist()
if labels_to_use == 'species':
    N_CLASSES = [all_classes[0]]
elif labels_to_use == 'genus':
    N_CLASSES = [all_classes[0]]
else:
    N_CLASSES = all_classes

N_CLASSES


# In[ ]:


wandb.config['N_CLASSES'] = N_CLASSES
wandb.config['vocab_size'] = vocab_size


# In[ ]:


all_cols = pd.read_csv(data_splits_filepaths_csv['train'], nrows=5).columns.tolist()

input_cols = all_cols.copy()
for label_col in label_cols:
    input_cols.remove(label_col)


# In[ ]:


# Load data
train_ds = tf.data.experimental.CsvDataset(
    str(data_splits_filepaths_csv['train']), [tf.float32] * len(input_cols) + [tf.int32] * len(label_cols), header=True, 
).batch(BATCH_SIZE)
val_ds = tf.data.experimental.CsvDataset(
    str(data_splits_filepaths_csv['val']), [tf.float32] * len(input_cols) + [tf.int32] * len(label_cols), header=True, 
).batch(BATCH_SIZE)


# In[ ]:


print(val_ds.unbatch().batch(4).take(1).get_single_element())


# In[ ]:


if labels_to_use == 'species':
    @tf.function
    def preprocess(*all_cols):
        inps = tf.transpose(all_cols[:(len(label_cols) * -1)])
        inps = tf.expand_dims(inps, 1)
        inps = tf.expand_dims(inps, -1)
        labs = all_cols[(len(label_cols) * -1)::]
        lab1 = tf.one_hot(labs[0], depth=N_CLASSES[0])
        return inps, lab1
elif labels_to_use == 'genus':
    @tf.function
    def preprocess(*all_cols):
        inps = tf.transpose(all_cols[:(len(label_cols) * -1)])
        inps = tf.expand_dims(inps, 1)
        inps = tf.expand_dims(inps, -1)
        labs = all_cols[(len(label_cols) * -1)::]
        lab2 = tf.one_hot(labs[1], depth=N_CLASSES[0])
        return inps, lab2
else:
    @tf.function
    def preprocess(*all_cols):
        inps = tf.transpose(all_cols[:(len(label_cols) * -1)])
        inps = tf.expand_dims(inps, 1)
        inps = tf.expand_dims(inps, -1)
        labs = all_cols[(len(label_cols) * -1)::]
        lab1 = tf.one_hot(labs[0], depth=N_CLASSES[0])
        lab2 = tf.one_hot(labs[1], depth=N_CLASSES[1])
        return inps, (lab1, lab2)


# In[ ]:


tf_options = tf.data.Options()
tf_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy(2)


# In[ ]:


# We use prefetch() to pre-compute preprocessed batches on the fly on our CPU.
processed_train_ds = train_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE).with_options(tf_options)
processed_val_ds = val_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE).with_options(tf_options)


# In[ ]:


# Preview a single input example.
print(processed_val_ds.take(1).get_single_element())


# In[ ]:


inputs = tf.keras.Input(dtype=tf.float32, shape=(1, max_input_len, 1,))
outputs = tf.math.multiply(inputs, 1/vocab_size)


# In[ ]:


optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config['learning_rate'])
wandb.config['optimizer'] = optimizer


# In[ ]:


mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy() 


# In[ ]:


with mirrored_strategy.scope():

    assert (gen_filter_size % 2) != 0, f"gen_filter_size must be an odd number. currently set to {gen_filter_size}"
    pad_size = int((max_seq_len - (max_seq_len - (gen_filter_size - 1))) / 2)

    outputs = tf.keras.layers.Conv2D(first_filter_size, (1, gen_filter_size), padding='same', strides=1, activation=None)(inputs)
    outputs = tf.keras.layers.BatchNormalization(momentum=bn_momen, scale=SCALE, center=CENTER, epsilon=EPSILON)(outputs)
    outputs = tf.keras.layers.Activation(activation)(outputs)

    cur_filter_size = first_filter_size 
    for br_i, block_rep in enumerate(res_block_reps):
        # print(block_rep)
        for r_i in np.arange(block_rep):
            # print(f"\t{r_i}")


            # print(outputs.shape)
            if (r_i == 0) & (br_i == 0):
                outputs = tf.keras.layers.Conv2D(cur_filter_size * 4, (1, 1), padding='same', strides=1, activation=None)(outputs)
                outputs = tf.keras.layers.BatchNormalization(momentum=bn_momen, scale=SCALE, center=CENTER, epsilon=EPSILON)(outputs)
                skip_layer = outputs
            elif r_i == 0:
                cur_filter_size *= 2
                outputs = tf.keras.layers.Conv2D(cur_filter_size * 4, (1, 1), padding='same', strides=1, activation=None)(outputs)
                outputs = tf.keras.layers.BatchNormalization(momentum=bn_momen, scale=SCALE, center=CENTER, epsilon=EPSILON)(outputs)
                skip_layer = outputs
            else:
                skip_layer = outputs


            # print(outputs.shape)
            outputs = tf.keras.layers.Conv2D(cur_filter_size, (1, 1), padding='same', strides=1, activation=None)(outputs)
            outputs = tf.keras.layers.BatchNormalization(momentum=bn_momen, scale=SCALE, center=CENTER, epsilon=EPSILON)(outputs)
            outputs = tf.keras.layers.Activation(activation)(outputs)

            # print(outputs.shape)
            outputs = tf.keras.layers.Conv2D(cur_filter_size, (1, gen_filter_size), strides=1, activation=None)(skip_layer)
            outputs = tf.keras.layers.ZeroPadding2D((0, pad_size))(outputs)
            outputs = tf.keras.layers.BatchNormalization(momentum=bn_momen, scale=SCALE, center=CENTER, epsilon=EPSILON)(outputs)
            outputs = tf.keras.layers.Activation(activation)(outputs)

            # print(outputs.shape)
            outputs = tf.keras.layers.Conv2D(cur_filter_size * 4, (1, 1), padding='same', strides=1, activation=None)(outputs)
            outputs = tf.keras.layers.BatchNormalization(momentum=bn_momen, scale=SCALE, center=CENTER, epsilon=EPSILON)(outputs)

            # print(outputs.shape)
            outputs = tf.keras.layers.Add()([outputs, skip_layer])
            outputs = tf.keras.layers.Activation(activation)(outputs)
            # print(outputs.shape)


    outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
    
    prediction_head = []
    for n_class in N_CLASSES:
        prediction_head.append(keras.layers.Dense(n_class, activation=LAST_ACTIVATION)(outputs))
    
    model = tf.keras.Model(inputs, prediction_head)
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["categorical_accuracy",],
    )


# In[ ]:


model.summary()


# In[ ]:


trainable_param_count = count_params(model.trainable_weights)
non_trainable_param_count = count_params(model.non_trainable_weights)


# In[ ]:


trainable_param_count


# In[ ]:


wandb.config['trainable_param_count'] = trainable_param_count
wandb.config['non_trainable_param_count'] = non_trainable_param_count
wandb.config['total_param_count'] = non_trainable_param_count + trainable_param_count


# In[ ]:


earlystopping_cb = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=EPOCHS,
                        restore_best_weights=True
                    )

tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=tf_logs_dir,
    histogram_freq=1,
)


# In[ ]:


configs = wandb.config


# In[ ]:


run = wandb.init(project=project_name,
          name=run_name,
          )


# In[ ]:


# train model
with run:
    model.fit(
        processed_train_ds, validation_data=processed_val_ds, 
        epochs=EPOCHS,
        callbacks=[
            earlystopping_cb, 
            tensorboard_cb, 
            WandbCallback(),
        ]
    )
    # model.evaluate(processed_val_ds)
    wandb.config.update(configs)


# In[ ]:




