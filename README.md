# NLP

Evaluates polymer corpora with material science BERT models.

The user selects their desired model and corpus (options are listed below). The corpus is loaded and processed to prepare for Named Entity Recognition (NER). THis includes assigning consistent labels according to BIO scheme and truncating long texts. The selected model is loaded from the Huggingface repository except the "MatBERT" models which are loaded from Github. 

Afterwards, an NER head is initialised and trained on the training set of the selected corpus. After each training epoch evaluation is carried out using the validation/evaluation set of the corpus. This calculates the per-entity and micro precision, recall and F1 scores are calculated. The iteration which achieved the highest micro-F1 is saved. After training, the training loss, evaluation loss and F1 scores per epoch are plotted to analyse model performance per epoch.

After running, the program should have a BERT model with a trained NER head, evaluation values for the best performing model and a plot of losses and F1 scores per epoch.

Files included: python script to load models/corpora and perform evaluation and the training/evaluation/test sets of each corpus.

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
