 BATCH_SIZE=128
 BEAM=5
 SEED=1
 SCORING=bleu
 CHECKPOINT_PATH="/home/mohammed_yahia3/models/checkpoint_best.pt" 

fairseq-generate "/home/mohammed_yahia3/models/BT_pcm_en.tokenized.pcm-en" \
    --batch-size $BATCH_SIZE \
    --beam $BEAM \
    --path $CHECKPOINT_PATH \
    --seed $SEED \
    --scoring bleu > "/home/mohammed_yahia3/checker.txt"
