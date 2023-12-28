BASE_MODEL=vinai/PhoGPT-7B5
python evaluation.py \
   --model_name_or_path $BASE_MODEL \
   --mode dev --mask_embedding_sentence \
   --load_kbit 4 --icl_examples_file 274_templates.txt