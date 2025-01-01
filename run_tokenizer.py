import sentencepiece as spm

# Training a SentencePiece model

spm.SentencePieceTrainer.Train(
    "--input=corpus.txt "
    "--model_prefix=bpe_model "
    "--vocab_size=37000 "
    "--character_coverage=0.999 "
    "--model_type=bpe "
    "--pad_id=-1 "
    "--unk_id=0 "
    "--bos_id=1 "
    "--eos_id=2 "
    "--user_defined_symbols=<cls>,<sep>,<mask> "
    "--normalization_rule_name=identity"
)

# Test
sp = spm.SentencePieceProcessor()
sp.load('bpe_model.model')

tokens = sp.encode_as_pieces("This is a test sentence.")
print(tokens)
