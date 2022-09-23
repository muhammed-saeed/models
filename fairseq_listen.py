from fairseq.models.transformer import TransformerModel
pcm2en_large_model_path="/home/mohammed_yahia3/models/"
pcm2en_small_model_path='/home/CE/musaeed/kd-distiller/checkpoints/pcm_en_1_layer_512_embedddings_512_ffnn/'
pcm2en_small_model_preprocess='/home/CE/musaeed/auto-annotator/kd-distiller/JW300_with_bible_joined_embeddings/FAKE_pcm_en.tokenized.pcm-en'
pcm2en_large_model_preprocess="/home/mohammed_yahia3/models/BT_pcm_en.tokenized.pcm-en"
pcm2en =  TransformerModel.from_pretrained(pcm2en_large_model_path
  ,
  checkpoint_file='checkpoint_last.pt',
  data_name_or_path=pcm2en_large_model_preprocess,
  bpe='sentencepiece',
  sentencepiece_model='/home/mohammed_yahia3/models/pcm__vocab_4000.model'
)
