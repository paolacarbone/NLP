# NLP

Evaluates polymer corpora with material science BERT models

Arguments:
"-m", "--model        Choice of BERT model.
                      options:
                      materialsbert: MaterialsBERT by Ramprasaad Group
                      matbert_cased: MatBERT by Ceder group
                      matbert_uncased: MatBERT by Ceder group
                      scibert_cased: SciBERT by AllenAI
                      scibert_uncased: SciBERT by AllenAI
                      bert_uncased: BERT base uncased
                      bert_cased: BERT base cased
                      matscibert: MatSciBERt by T. Gupta et al

-c", "--corpus"       Choice of corpus to use for training and testing
                      options:
                      polyie: PolymerIE dataset
                      polymerabstracts
                      matscholar: MatScholar dataset
                      pcmsp: PC-MSP dataset

"-o", "--output"      Output directory to save the evaluation results

"-s", "--model_save"  Output directory to save the model

Example input:
python bert_eval_main.py -m matbert_cased -o matbert_cased_polymerabstracts_eval.json -s matbert_cased_pa -c polymerabstracts
