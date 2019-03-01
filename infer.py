from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import numpy as np
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
from nltk import tokenize

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("bert_config_file", None,
                    "The config json file corresponding to the pre-trained BERT model. "
                    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("input_file", None,
                    "The input text file.")

flags.DEFINE_string("output_dir", None,
                    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string("init_checkpoint", None,
                    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_lower_case", True,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")


class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class  InputFeatures(object):
    """A  single  set  of  features  of  data."""

    def  __init__(self,  unique_id,  tokens,  input_ids,  input_mask,  input_type_ids):
        self.unique_id  =  unique_id
        self.tokens  =  tokens
        self.input_ids  =  input_ids
        self.input_mask  =  input_mask
        self.input_type_ids  =  input_type_ids
def read_sentences(input_file):
    with  tf.gfile.GFile(input_file,  "r")  as  reader:
        text  =  tokenization.convert_to_unicode(reader.read())
        sentences = tokenize.sent_tokenize(text)
    return sentences

def convert_sentences_to_examples(sentences):
    examples  =  []
    for unique_id, sentence in enumerate(sentences):
        examples.append(InputExample(unique_id=unique_id,  text_a=sentence,  text_b=None))
    return  examples

def  convert_examples_to_features(examples,  seq_length,  tokenizer):
    """Loads  a  data  file  into  a  list  of  `InputBatch`s."""

    features  =  []
    for  (ex_index,  example)  in  enumerate(examples):
        tokens_a  =  tokenizer.tokenize(example.text_a)

        tokens_b  =  None
        if  example.text_b:
            tokens_b  =  tokenizer.tokenize(example.text_b)

        if  tokens_b:
            _truncate_seq_pair(tokens_a,  tokens_b,  seq_length  -  3)
        else:
            if  len(tokens_a)  >  seq_length  -  2:
                tokens_a  =  tokens_a[0:(seq_length  -  2)]
        tokens  =  []
        input_type_ids  =  []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for  token  in  tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if  tokens_b:
            for  token  in  tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids  =  tokenizer.convert_tokens_to_ids(tokens)
        input_mask  =  [1]  *  len(input_ids)
        while  len(input_ids)  <  seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert  len(input_ids)  ==  seq_length
        assert  len(input_mask)  ==  seq_length
        assert  len(input_type_ids)  ==  seq_length

        if  ex_index  <  5:
            tf.logging.info("***  Example  ***")
            tf.logging.info("unique_id:  %s"  %  (example.unique_id))
            tf.logging.info("tokens:  %s"  %  "  ".join(
                    [tokenization.printable_text(x)  for  x  in  tokens]))
            tf.logging.info("input_ids:  %s"  %  "  ".join([str(x)  for  x  in  input_ids]))
            tf.logging.info("input_mask:  %s"  %  "  ".join([str(x)  for  x  in  input_mask]))
            tf.logging.info(
                    "input_type_ids:  %s"  %  "  ".join([str(x)  for  x  in  input_type_ids]))

        features.append(
                InputFeatures(
                        unique_id=example.unique_id,
                        tokens=tokens,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        input_type_ids=input_type_ids))
    return  features
def  model_fn_builder(bert_config,  init_checkpoint, ):
    """Returns  `model_fn`  closure  for  TPUEstimator."""

    def  model_fn(features,  labels,  mode,  params):    #  pylint:  disable=unused-argument
        """The  `model_fn`  for  TPUEstimator."""

        unique_ids  =  features["unique_ids"]
        input_ids  =  features["input_ids"]
        input_mask  =  features["input_mask"]
        input_type_ids  =  features["input_type_ids"]

        model  =  modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids,
                use_one_hot_embeddings=False)

        if  mode  !=  tf.estimator.ModeKeys.PREDICT:
            raise  ValueError("Only  PREDICT  modes  are  supported:  %s"  %  (mode))

        tvars  =  tf.trainable_variables()
        scaffold_fn  =  None
        (assignment_map,
          initialized_variable_names)  =  modeling.get_assignment_map_from_checkpoint(
                  tvars,  init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint,  assignment_map)

        tf.logging.info("****  Trainable  Variables  ****")
        for  var  in  tvars:
            init_string  =  ""
            if  var.name  in  initialized_variable_names:
                init_string  =  ",  *INIT_FROM_CKPT*"
            tf.logging.info("    name  =  %s,  shape  =  %s%s",  var.name,  var.shape,
                                            init_string)
        output_layer  =  model.get_pooled_output()
        predictions  =  {"unique_id":  unique_ids, "output_layer": output_layer}
        output_spec  =  tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,  predictions=predictions,  scaffold_fn=scaffold_fn)
        return  output_spec
    return  model_fn

def  input_fn_builder(features,  seq_length):
    """Creates  an  `input_fn`  closure  to  be  passed  to  TPUEstimator."""

    all_unique_ids  =  []
    all_input_ids  =  []
    all_input_mask  =  []
    all_input_type_ids  =  []

    for  feature  in  features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def  input_fn(params):
        """The  actual  input  function."""
        batch_size  =  params["batch_size"]
        num_examples  =  len(features)
        d  =  tf.data.Dataset.from_tensor_slices({
                "unique_ids":
                        tf.constant(all_unique_ids,  shape=[num_examples],  dtype=tf.int32),
                "input_ids":
                        tf.constant(
                                all_input_ids,  shape=[num_examples,  seq_length],
                                dtype=tf.int32),
                "input_mask":
                        tf.constant(
                                all_input_mask,
                                shape=[num_examples,  seq_length],
                                dtype=tf.int32),
                "input_type_ids":
                        tf.constant(
                                all_input_type_ids,
                                shape=[num_examples,  seq_length],
                                dtype=tf.int32),
        })

        d  =  d.batch(batch_size=batch_size,  drop_remainder=False)
        return  d
    return  input_fn
def encode_sentences(sentences):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    is_per_host  =  tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config  =  tf.contrib.tpu.RunConfig(
            master=None,
            tpu_config=tf.contrib.tpu.TPUConfig(
                    num_shards=1,
                    per_host_input_for_training=is_per_host))

    examples  =  convert_sentences_to_examples(sentences)
    features  =  convert_examples_to_features(examples=examples,  seq_length=FLAGS.max_seq_length,  tokenizer=tokenizer)
    model_fn  =  model_fn_builder(bert_config=bert_config, init_checkpoint=FLAGS.init_checkpoint)
    estimator  =  tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            predict_batch_size=FLAGS.batch_size)
    input_fn  =  input_fn_builder(features=features,  seq_length=FLAGS.max_seq_length)
    embeddings = [None] * len(examples)
    for result in  estimator.predict(input_fn,  yield_single_examples=True):
        embeddings[result["unique_id"]] = result["output_layer"]
    return embeddings

def main(self):
    tf.logging.set_verbosity(tf.logging.INFO)
    sentences  =  read_sentences(FLAGS.input_file)
    embeddings = encode_sentences(sentences)
    #for  i, result  in  enumerate(embeddings):
    #    print (i, len(result))
    x = tf.constant(np.array(embeddings))

    w = tf.Variable(tf.random_uniform([5, len(embeddings)]))
    ww = tf.nn.softmax(w)
    centers = tf.matmul(ww, x)
    n = tf.Variable(tf.random_uniform([len(embeddings), 5]))
    nn = tf.log(1 + tf.exp(n))
    y = tf.matmul(nn, centers)
    loss = tf.nn.l2_loss(x - y)
    init = tf.initialize_all_variables()
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)
    with tf.Session() as sess:
        sess.run(init)
        for step in range(10000):
            sess.run(train)
            print ("step:", step, "loss:", sess.run(loss))
        ww_v = sess.run(ww)
        idx = np.argmax(ww_v, axis=1)
        for i in sorted(set(idx)):
            print (sentences[i])
if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
