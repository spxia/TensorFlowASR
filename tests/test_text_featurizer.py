import os
import tensorflow as tf
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRSliceDataset
from tensorflow_asr.utils import setup_environment, setup_strategy

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#txf = CharFeaturizer(None)
#b = txf.extract("fkaff aksfbfnak kcjhoiu")
#print (b)



config_dir = "tests/config_aishell.yml"              
config = Config(config_dir, learning=True)

speech_featurizer = TFSpeechFeaturizer(config.speech_config)
text_featurizer = CharFeaturizer(config.decoder_config)


train_dataset = ASRSliceDataset(
    data_paths=config.learning_config.dataset_config.train_paths,
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    augmentations=config.learning_config.augmentations,
    stage="train", cache=False, shuffle=True, sort=False
)

train_data = train_dataset.create(2)

train_data_loader = strategy.experimental_distribute_dataset(train_data)


train_iterator = iter(train_data_loader)
while True:
    batch=next(train_iterator)
    features, input_length, labels, label_length, prediction, prediction_length = batch
    #print ("features")
    #print (features)
    #print ("input_length")
    #print (input_length)
    print ("labels")
    print (labels)
    #print ("label_length")
    #print (label_length)
    #print ("prediction")
    #print (prediction)
    #print ("prediction_length")
    #print (prediction_length)

"""
a = text_featurizer.extract("一丁七万 丈 oo A 一")
print(a)
"""