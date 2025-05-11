# NLP

The goal of this program is to train a BERT model to perfrom Named Entity Recognition (NER) on polymer literature. To that end, a variety of pre-trained models and training/evaluation labelled datasets (corpora) are included. A pre-trained model is fine-tuned for the downstream task of NER using one of the corpora's training data. The evaluation dataset is included to test the effectiveness of the fine-tuned model. Each model is fine-tuned on each corpus and evaluation is performed in order to compare the effectiveness of each model with the material contained in each corpus.

The user selects their desired model and corpus (options are listed below). The corpus is loaded and processed to prepare for NER. THis includes assigning consistent labels according to BIO scheme and truncating long texts. The selected model is loaded from the Huggingface repository except the "MatBERT" models which are loaded from Github. 

Afterwards, an NER head is initialised and trained on the training set of the selected corpus. After each training epoch evaluation is carried out using the validation/evaluation set of the corpus. This calculates the per-entity and micro precision, recall and F1 scores are calculated. The iteration which achieved the highest micro-F1 is saved. After training, the training loss, evaluation loss and F1 scores per epoch are plotted to analyse model performance per epoch.

After running, the program should have a BERT model with a trained NER head, evaluation values for the best performing model and a plot of losses and F1 scores per epoch.

Files included: python script to load models/corpora and perform evaluation and the training/evaluation/test sets of each corpus.

Arguments:

"-m", "--model        Choice of BERT model.

                      options:
                      
                      materialsbert: MaterialsBERT by Ramprasaad Group. A model trained on biomedical text and then fine-tuned on polymer literature [1]

                      scibert_cased: SciBERT by AllenAI. Pre-trained on general scientific literature (mainly biomedicine and computer science) [2]
                      
                      scibert_uncased: SciBERT by AllenAI [2].
                      
                      matbert_cased: MatBERT by Ceder group. Pre-trained on material science literature [3].
                      
                      matbert_uncased: MatBERT by Ceder group [3].
                      
                      bert_uncased: BERT base uncased. Pre-trained on general English text [4].
                      
                      bert_cased: BERT base cased [4].
                      
                      matscibert: MatSciBERt by T. Gupta et al. Fine-tuning of SciBERT on material science literature [5].

-c", "--corpus"       Choice of corpus to use for training and testing

                      options:
                      
                      polyie: PolymerIE dataset. Polymer fuel cell full articles [6].
                      
                      polymerabstracts. General polymer abstracts [1].
                      
                      matscholar: MatScholar dataset. Abstracts on material science literature from 1900-2016. Numerical values are unlabelled [7].
                      
                      pcmsp: PC-MSP dataset. Full papers of polycrystalline synthesis [8].

"-o", "--output"      Output directory to save the evaluation results

"-s", "--model_save"  Output directory to save the model

Example input:

python bert_eval_main.py -m matbert_cased -o matbert_cased_polymerabstracts_eval.json -s matbert_cased_pa -c polymerabstracts

Evaluation results:

<img src="https://github.com/user-attachments/assets/1704077b-4b60-414a-aa0e-98dd5cc46a84" alt="eval_table" width="500">

Each model is evaluated using each corpus and the score is calculated as the F1 score in order to account for both the correctness and coverage of the extracted information.

F1 is calculated thusly:

![F1 score](https://latex.codecogs.com/png.image?\dpi{120}&space;F1=\frac{2\cdot(Precision\cdot&space;Recall)}{Precision+Recall})

![Precision](https://latex.codecogs.com/png.image?\dpi{120}&space;Precision=\frac{\text{True%20Positives}}{\text{True%20Positives}%20+%20\text{False%20Positives}})

![Recall](https://latex.codecogs.com/png.image?\dpi{120}&space;Recall=\frac{\text{True%20Positives}}{\text{True%20Positives}%20+%20\text{False%20Negatives}})


References:

[1] Shetty, P., Rajan, A.C., Kuenneth, C. et al. A general-purpose material property data extraction pipeline from large polymer corpora using natural language processing. npj Comput Mater 9, 52 (2023). https://doi.org/10.1038/s41524-023-01003-w

[2]Iz Beltagy, Kyle Lo, and Arman Cohan. 2019. SciBERT: A Pretrained Language Model for Scientific Text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3615–3620, Hong Kong, China. Association for Computational Linguistics.

[3] Trewartha, A. et al. (2022) ‘Quantifying the advantage of domain-specific pre-training on named entity recognition tasks in materials science’, Patterns, 3(4), p. 100488. doi:10.1016/j.patter.2022.100488. 

[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

[5] Gupta, T., Zaki, M., Krishnan, N.M.A. et al. MatSciBERT: A materials domain language model for text mining and information extraction. npj Comput Mater 8, 102 (2022). https://doi.org/10.1038/s41524-022-00784-w

[6] Jerry Cheung, Yuchen Zhuang, Yinghao Li, Pranav Shetty, Wantian Zhao, Sanjeev Grampurohit, Rampi Ramprasad, and Chao Zhang. 2024. POLYIE: A Dataset of Information Extraction from Polymer Material Scientific Literature. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2370–2385, Mexico City, Mexico. Association for Computational Linguistics.

[7] Weston, L. et al. (2019) ‘Named entity recognition and normalization applied to large-scale information extraction from the materials science literature’, Journal of Chemical Information and Modeling, 59(9), pp. 3692–3702. doi:10.1021/acs.jcim.9b00470. 

[8] Xianjun Yang, Ya Zhuo, Julia Zuo, Xinlu Zhang, Stephen Wilson, and Linda Petzold. 2022. PcMSP: A Dataset for Scientific Action Graphs Extraction from Polycrystalline Materials Synthesis Procedure Text. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 6033–6046, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
